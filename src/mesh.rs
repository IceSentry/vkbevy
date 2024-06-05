use std::time::Instant;

use ash::vk;
use bevy::prelude::*;
use bytemuck::cast_slice;

use crate::vk_bevy_instance::{Buffer, VkBevyInstance};

#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Vertex {
    pub pos: [f32; 4],
    pub color: [f32; 4],
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct MeshDraw {
    pub offset: [f32; 2],
    pub scale: [f32; 2],
}

unsafe impl bytemuck::Zeroable for MeshDraw {}
unsafe impl bytemuck::Pod for MeshDraw {}

#[derive(Component, Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

#[derive(Component, Deref)]
pub struct VertexBuffer(pub Buffer);
#[derive(Component, Deref)]
pub struct IndexBuffer(pub Buffer);

#[derive(Component)]
pub struct PreparedMesh;

pub fn prepare_mesh(
    mut commands: Commands,
    mut vk_bevy: ResMut<VkBevyInstance>,
    meshes: Query<(Entity, &Mesh), Without<PreparedMesh>>,
) {
    for (entity, mesh) in &meshes {
        info!("preparing mesh");
        let start = Instant::now();

        let vertex_buffer_data = cast_slice(&mesh.vertices);
        let index_buffer_data = cast_slice(&mesh.indices);

        let mut entity_cmd = commands.entity(entity);

        let mut scratch_buffer = vk_bevy
            .create_buffer(
                128 * 1024 * 1024,
                vk::BufferUsageFlags::TRANSFER_SRC,
                gpu_allocator::MemoryLocation::CpuToGpu,
            )
            .unwrap();

        let vertex_buffer = vk_bevy
            .create_buffer(
                128 * 1024 * 1024,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )
            .unwrap();
        vk_bevy.upload_buffer(
            vk_bevy.command_buffers[0],
            &mut scratch_buffer,
            &vertex_buffer,
            vertex_buffer_data,
        );
        entity_cmd.insert(VertexBuffer(vertex_buffer));

        let index_buffer = vk_bevy
            .create_buffer(
                128 * 1024 * 1024,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )
            .unwrap();
        vk_bevy.upload_buffer(
            vk_bevy.command_buffers[0],
            &mut scratch_buffer,
            &index_buffer,
            index_buffer_data,
        );
        entity_cmd.insert(IndexBuffer(index_buffer));

        entity_cmd.insert(PreparedMesh);
        info!("mesh prepared in {}ms", start.elapsed().as_millis());
    }
}
