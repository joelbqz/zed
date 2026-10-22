use anyhow::{Context as _, Result};
use derive_more::{Deref, DerefMut};
use etagere::BucketedAtlasAllocator;
use gpui::{
    AtlasBackend, AtlasKey, AtlasState, AtlasTextureId, AtlasTextureKind, AtlasTextureList,
    AtlasTile, Bounds, BuiltTexture, DevicePixels, PlatformAtlas, Point, Size, TextureBuilder,
};
use metal::Device;
use parking_lot::Mutex;
use std::{borrow::Cow, sync::Arc};

pub struct MetalAtlas(Mutex<AtlasState<MetalAtlasTextures>>);

impl MetalAtlas {
    pub(crate) fn new(device: Device, is_apple_gpu: bool) -> Self {
        MetalAtlas(Mutex::new(AtlasState::new(MetalAtlasTextures {
            device: AssertSend(device),
            is_apple_gpu,
            monochrome_textures: Default::default(),
            polychrome_textures: Default::default(),
        })))
    }

    /// Returns the GPU texture backing `id`, or `None` once every tile in it
    /// has been removed. A scene can still reference such a texture when a
    /// cached view replays a paint from before the image was dropped, so
    /// callers must skip those sprites rather than assume the texture exists.
    pub(crate) fn metal_texture(&self, id: AtlasTextureId) -> Option<metal::Texture> {
        Some(self.0.lock().backend.texture(id)?.metal_texture.clone())
    }
}

struct MetalAtlasTextures {
    device: AssertSend<Device>,
    is_apple_gpu: bool,
    monochrome_textures: AtlasTextureList<MetalAtlasTexture>,
    polychrome_textures: AtlasTextureList<MetalAtlasTexture>,
}

impl PlatformAtlas for MetalAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        self.0.lock().get_or_insert_with(key, build)
    }

    fn remove(&self, key: &AtlasKey) {
        self.0.lock().remove(key);
    }

    fn texture_builder(&self) -> Option<Arc<dyn TextureBuilder>> {
        let state = self.0.lock();
        Some(Arc::new(MetalTextureBuilder {
            device: (*state.backend.device).clone(),
            is_apple_gpu: state.backend.is_apple_gpu,
        }))
    }

    fn insert_built(&self, key: AtlasKey, texture: BuiltTexture) -> Result<Option<AtlasTile>> {
        let Ok(built) = texture.0.downcast::<BuiltMetalTexture>() else {
            return Ok(None);
        };
        if key.texture_kind() != AtlasTextureKind::Polychrome {
            return Ok(None);
        }
        self.0
            .lock()
            .insert_built(key, |backend| Ok(backend.push_built(*built)))
    }
}

/// Makes polychrome textures of their own (images) on any thread, for [`MetalAtlas::insert_built`].
struct MetalTextureBuilder {
    device: Device,
    is_apple_gpu: bool,
}

struct BuiltMetalTexture {
    texture: metal::Texture,
    size: Size<DevicePixels>,
}

impl TextureBuilder for MetalTextureBuilder {
    fn build(&self, size: Size<DevicePixels>, bytes: &[u8]) -> Option<BuiltTexture> {
        let (width, height) = (size.width.0, size.height.0);
        if width <= 0
            || height <= 0
            || width > MAX_ATLAS_SIZE.width.0
            || height > MAX_ATLAS_SIZE.height.0
            || bytes.len() < width as usize * height as usize * 4
        {
            return None;
        }
        let texture = new_texture(
            &self.device,
            self.is_apple_gpu,
            size,
            AtlasTextureKind::Polychrome,
        );
        texture.replace_region(
            metal::MTLRegion::new_2d(0, 0, width as u64, height as u64),
            0,
            bytes.as_ptr() as *const _,
            width as u64 * 4,
        );
        Some(BuiltTexture(Box::new(BuiltMetalTexture { texture, size })))
    }
}

/// Max texture size on all modern Apple GPUs. Anything bigger than that crashes in validateWithDevice.
const MAX_ATLAS_SIZE: Size<DevicePixels> = Size {
    width: DevicePixels(16384),
    height: DevicePixels(16384),
};

fn new_texture(
    device: &Device,
    is_apple_gpu: bool,
    size: Size<DevicePixels>,
    kind: AtlasTextureKind,
) -> metal::Texture {
    let texture_descriptor = metal::TextureDescriptor::new();
    texture_descriptor.set_width(size.width.into());
    texture_descriptor.set_height(size.height.into());
    let pixel_format;
    let usage;
    match kind {
        AtlasTextureKind::Monochrome => {
            pixel_format = metal::MTLPixelFormat::A8Unorm;
            usage = metal::MTLTextureUsage::ShaderRead;
        }
        AtlasTextureKind::Polychrome => {
            pixel_format = metal::MTLPixelFormat::BGRA8Unorm;
            usage = metal::MTLTextureUsage::ShaderRead;
        }
        AtlasTextureKind::Subpixel => unreachable!(),
    }
    texture_descriptor.set_pixel_format(pixel_format);
    texture_descriptor.set_usage(usage);
    // Shared memory mode can be used only on Apple GPU families
    // https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared
    texture_descriptor.set_storage_mode(if is_apple_gpu {
        metal::MTLStorageMode::Shared
    } else {
        metal::MTLStorageMode::Managed
    });
    device.new_texture(&texture_descriptor)
}

impl AtlasBackend for MetalAtlasTextures {
    fn insert(
        &mut self,
        kind: AtlasTextureKind,
        size: Size<DevicePixels>,
        bytes: &[u8],
    ) -> Result<AtlasTile> {
        let tile = self.allocate(size, kind).context("failed to allocate")?;
        let texture = self
            .texture(tile.texture_id)
            .context("allocated tile refers to a missing texture")?;
        texture.upload(tile.bounds, bytes);
        Ok(tile)
    }

    fn remove(&mut self, tile: AtlasTile) {
        let id = tile.texture_id;

        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };

        let Some(texture_slot) = textures
            .textures
            .iter_mut()
            .find(|texture| texture.as_ref().is_some_and(|v| v.id == id))
        else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                textures.free_list.push(id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }
}

impl MetalAtlasTextures {
    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = match texture_kind {
                AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
                AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
                AtlasTextureKind::Subpixel => unreachable!(),
            };

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind);
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut MetalAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        let size = min_size.min(&MAX_ATLAS_SIZE).max(&DEFAULT_ATLAS_SIZE);
        let metal_texture = new_texture(&self.device, self.is_apple_gpu, size, kind);
        self.push_metal_texture(metal_texture, size, kind)
    }

    /// A texture built elsewhere (see [`MetalTextureBuilder`]) as a page of its own, all of
    /// it one tile.
    fn push_built(&mut self, built: BuiltMetalTexture) -> Option<AtlasTile> {
        let kind = AtlasTextureKind::Polychrome;
        self.push_metal_texture(built.texture, built.size, kind)
            .allocate(built.size)
    }

    fn push_metal_texture(
        &mut self,
        metal_texture: metal::Texture,
        size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut MetalAtlasTexture {
        let texture_list = match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };

        let index = texture_list.free_list.pop();

        let atlas_texture = MetalAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            allocator: etagere::BucketedAtlasAllocator::new(size_to_etagere(size)),
            metal_texture: AssertSend(metal_texture),
            live_atlas_keys: 0,
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list.textures.get_mut(ix)
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list.textures.last_mut()
        }
        .unwrap()
        .as_mut()
        .unwrap()
    }

    fn texture(&self, id: AtlasTextureId) -> Option<&MetalAtlasTexture> {
        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };
        textures.textures.get(id.index as usize)?.as_ref()
    }
}

struct MetalAtlasTexture {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    metal_texture: AssertSend<metal::Texture>,
    live_atlas_keys: u32,
}

impl MetalAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            bounds: Bounds {
                origin: point_from_etagere(allocation.rectangle.min),
                size,
            },
            padding: 0,
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn upload(&self, bounds: Bounds<DevicePixels>, bytes: &[u8]) {
        let region = metal::MTLRegion::new_2d(
            bounds.origin.x.into(),
            bounds.origin.y.into(),
            bounds.size.width.into(),
            bounds.size.height.into(),
        );
        self.metal_texture.replace_region(
            region,
            0,
            bytes.as_ptr() as *const _,
            bounds.size.width.to_bytes(self.bytes_per_pixel()) as u64,
        );
    }

    fn bytes_per_pixel(&self) -> u8 {
        use metal::MTLPixelFormat::*;
        match self.metal_texture.pixel_format() {
            A8Unorm | R8Unorm => 1,
            RGBA8Unorm | BGRA8Unorm => 4,
            _ => unimplemented!(),
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&mut self) -> bool {
        self.live_atlas_keys == 0
    }
}

fn size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    etagere::Size::new(size.width.into(), size.height.into())
}

fn point_from_etagere(value: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels::from(value.x),
        y: DevicePixels::from(value.y),
    }
}

#[derive(Deref, DerefMut)]
struct AssertSend<T>(T);

unsafe impl<T> Send for AssertSend<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::PlatformAtlas;
    use std::borrow::Cow;

    fn create_atlas() -> Option<MetalAtlas> {
        let device = metal::Device::system_default()?;
        Some(MetalAtlas::new(device, true))
    }

    fn make_image_key(image_id: usize, frame_index: usize) -> AtlasKey {
        AtlasKey::Image(gpui::RenderImageParams {
            image_id: gpui::ImageId(image_id),
            frame_index,
        })
    }

    fn insert_tile(atlas: &MetalAtlas, key: AtlasKey, size: Size<DevicePixels>) -> AtlasTile {
        atlas
            .get_or_insert_with(key, &mut || {
                let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
                Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
            })
            .expect("allocation should succeed")
            .expect("callback returns Some")
    }

    #[test]
    fn test_remove_clears_stale_keys_from_tiles_by_key() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };

        let key_a = make_image_key(1, 0);
        let key_b = make_image_key(2, 0);
        let key_c = make_image_key(3, 0);

        let tile_a = insert_tile(&atlas, key_a.clone(), small);
        let tile_b = insert_tile(&atlas, key_b.clone(), small);
        let tile_c = insert_tile(&atlas, key_c.clone(), small);

        assert_eq!(tile_a.texture_id, tile_b.texture_id);
        assert_eq!(tile_b.texture_id, tile_c.texture_id);

        // Remove A: texture still has B and C, so it stays.
        // The key for A must be removed from tiles_by_key.
        atlas.remove(&key_a);

        // Remove B: texture still has C.
        atlas.remove(&key_b);

        // Remove C: texture becomes unreferenced and is deleted.
        atlas.remove(&key_c);

        // Re-inserting A must allocate a fresh tile on a new texture,
        // NOT return a stale tile referencing the deleted texture.
        let tile_a2 = insert_tile(&atlas, key_a, small);

        // The texture must actually exist — this would panic before the fix.
        assert!(atlas.metal_texture(tile_a2.texture_id).is_some());
    }

    #[test]
    fn test_metal_texture_is_none_after_last_tile_removed() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let key = make_image_key(1, 0);
        let tile = insert_tile(
            &atlas,
            key.clone(),
            Size {
                width: DevicePixels(64),
                height: DevicePixels(64),
            },
        );
        assert!(atlas.metal_texture(tile.texture_id).is_some());

        // A scene built before the removal may still carry `tile`; looking its
        // texture up must report the gap instead of panicking.
        atlas.remove(&key);
        assert!(atlas.metal_texture(tile.texture_id).is_none());
    }

    #[test]
    fn test_remove_deallocates_tile_space_for_reuse() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        let big = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };

        let keeper_key = make_image_key(1, 0);
        let big_key_a = make_image_key(2, 0);
        let big_key_b = make_image_key(3, 0);

        let keeper_tile = insert_tile(&atlas, keeper_key, small);
        let tile_a = insert_tile(&atlas, big_key_a.clone(), big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);
        let tile_b = insert_tile(&atlas, big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
    }

    #[test]
    fn test_remove_nonexistent_key_is_noop() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let key = make_image_key(999, 0);
        atlas.remove(&key);
    }

    #[test]
    fn a_built_texture_goes_in_as_a_page_of_its_own() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let size = Size {
            width: DevicePixels(3000),
            height: DevicePixels(2000),
        };
        let bytes = vec![7u8; 3000 * 2000 * 4];
        let builder = atlas.texture_builder().expect("Metal builds textures");
        let built = builder.build(size, &bytes).expect("builds");
        let key = make_image_key(42, 0);
        let tile = atlas
            .insert_built(key.clone(), built)
            .unwrap()
            .expect("takes it");
        assert_eq!(tile.bounds.origin, Point::default());
        assert_eq!(tile.bounds.size, size);
        // Asked for again, it's the same tile, never copied in.
        let again = atlas
            .get_or_insert_with(key.clone(), &mut || panic!("already there"))
            .unwrap()
            .unwrap();
        assert_eq!(again, tile);
        // Another image packs into a page of its own rather than this one.
        let other = insert_tile(
            &atlas,
            make_image_key(43, 0),
            Size {
                width: DevicePixels(64),
                height: DevicePixels(64),
            },
        );
        assert_ne!(other.texture_id, tile.texture_id);
        atlas.remove(&key);
        assert!(atlas.metal_texture(tile.texture_id).is_none());
        // Too big for the GPU, or short of pixels: not built.
        let huge = Size {
            width: DevicePixels(20000),
            height: DevicePixels(10),
        };
        assert!(builder.build(huge, &vec![0; 20000 * 10 * 4]).is_none());
        assert!(builder.build(size, &bytes[..100]).is_none());
    }
}
