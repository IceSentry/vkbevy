use std::time::Instant;

use ash::vk::{self, PresentModeKHR};
use bevy::{
    a11y::AccessibilityPlugin,
    app::AppExit,
    input::InputPlugin,
    log::LogPlugin,
    prelude::*,
    window::{PrimaryWindow, WindowResized},
    winit::{WinitPlugin, WinitWindows},
};
use mesh::{prepare_mesh, IndexBuffer, Mesh, Vertex, VertexBuffer};
use vk_bevy_instance::{Pipeline, Program, VkBevyInstance};

mod mesh;
mod shader;
mod vk_bevy_instance;

fn main() {
    App::new()
        .add_plugins((
            MinimalPlugins,
            WindowPlugin {
                primary_window: Some(Window {
                    title: "vkbevy".into(),
                    ..default()
                }),
                ..default()
            },
            AccessibilityPlugin,
            WinitPlugin::default(),
            InputPlugin,
            LogPlugin::default(),
        ))
        .add_systems(Startup, (setup, spawn_triangle_mesh).chain())
        .add_systems(Update, (resize, update).chain())
        .add_systems(Update, (prepare_mesh, exit_on_esc))
        .run();
}

fn exit_on_esc(key_input: Res<ButtonInput<KeyCode>>, mut exit_events: EventWriter<AppExit>) {
    if key_input.just_pressed(KeyCode::Escape) {
        exit_events.send_default();
    }
}

#[derive(Resource)]
pub struct MeshPipeline(pub Pipeline);
#[derive(Resource, Deref)]
pub struct MeshProgram(pub Program);

fn setup(
    mut commands: Commands,
    windows: Query<Entity, With<PrimaryWindow>>,
    winit_windows: NonSendMut<WinitWindows>,
) {
    let winit_window = windows
        .get_single()
        .ok()
        .and_then(|window_id| winit_windows.get_window(window_id))
        .expect("Failed to get winit window");

    let mut vk_bevy = VkBevyInstance::init(winit_window, PresentModeKHR::IMMEDIATE)
        .expect("Failed to initialize VkBevyInstance");

    let vertex_shader = vk_bevy.load_shader("assets/shaders/triangle.vert.glsl");
    let fragment_shader = vk_bevy.load_shader("assets/shaders/triangle.frag.glsl");

    let pipeline_rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo {
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        line_width: 1.0,
        polygon_mode: vk::PolygonMode::FILL,
        ..Default::default()
    };

    let pipeline_layout = vk_bevy
        .create_pipeline_layout()
        .expect("Failed to create pipeline layout");

    let pipeline = vk_bevy
        .create_graphics_pipeline(
            &pipeline_layout,
            vk_bevy.render_pass,
            &[vertex_shader.create_info(), fragment_shader.create_info()],
            vk::PrimitiveTopology::TRIANGLE_LIST,
            pipeline_rasterization_state_create_info,
            &[vk::VertexInputBindingDescription {
                binding: 0,
                stride: std::mem::size_of::<Vertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }],
            &[
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: {
                        unsafe {
                            let b: Vertex = std::mem::zeroed();
                            std::ptr::addr_of!(b.pos) as isize - std::ptr::addr_of!(b) as isize
                        }
                    } as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: {
                        unsafe {
                            let b: Vertex = std::mem::zeroed();
                            std::ptr::addr_of!(b.color) as isize - std::ptr::addr_of!(b) as isize
                        }
                    } as u32,
                },
            ],
        )
        .expect("Failed to create graphics pipeline");

    commands.insert_resource(MeshPipeline(pipeline));

    commands.insert_resource(vk_bevy);
}

fn spawn_triangle_mesh(mut commands: Commands) {
    commands.spawn(Mesh {
        vertices: vec![
            Vertex {
                pos: [-1.0, 1.0, 0.0, 1.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            Vertex {
                pos: [1.0, 1.0, 0.0, 1.0],
                color: [0.0, 0.0, 1.0, 1.0],
            },
            Vertex {
                pos: [0.0, -1.0, 0.0, 1.0],
                color: [1.0, 0.0, 0.0, 1.0],
            },
        ],
        indices: vec![0, 1, 2],
    });
}

#[allow(clippy::too_many_lines)]
fn update(
    vk_bevy: Res<VkBevyInstance>,
    mesh_pipeline: Res<MeshPipeline>,
    mut windows: Query<&mut Window>,
    meshes: Query<(&Mesh, &VertexBuffer, &IndexBuffer)>,
    mut frame_gpu_avg: Local<f64>,
    mut frame_cpu_avg: Local<f64>,
) {
    let begin_frame = Instant::now();

    let mut window = windows.single_mut();
    let width = window.physical_width();
    let height = window.physical_height();
    if width == 0 || height == 0 {
        // If the window is minimized the width and height will be 0. Skip rendering in this case.
        return;
    }

    // BEGIN

    let (image_index, command_buffer) = vk_bevy.begin_frame();

    // BEGIN RENDER PASS

    // info!("begin render pass");

    let clear_color = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.3, 0.3, 0.3, 1.0],
        },
    };
    let render_pass_begin_info = vk::RenderPassBeginInfo::default()
        .render_pass(vk_bevy.render_pass)
        .framebuffer(vk_bevy.framebuffers[image_index as usize])
        .render_area(vk::Rect2D::default().extent(vk::Extent2D {
            width: vk_bevy.swapchain_width,
            height: vk_bevy.swapchain_height,
        }))
        .clear_values(std::slice::from_ref(&clear_color));
    unsafe {
        vk_bevy.device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin_info,
            vk::SubpassContents::INLINE,
        );
    }

    // DRAW
    {
        // info!("draw");

        assert_eq!(width, vk_bevy.swapchain_width);
        assert_eq!(height, vk_bevy.swapchain_height);

        vk_bevy.set_viewport(command_buffer, width, height);

        vk_bevy.bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            &mesh_pipeline.0,
        );

        for (mesh, vb, ib) in &meshes {
            vk_bevy.bind_vertex_buffers(command_buffer, 0, &[vb.vk_buffer()], &[0]);
            vk_bevy.bind_index_buffer(command_buffer, ib, 0, vk::IndexType::UINT32);
            vk_bevy.draw_indexed(command_buffer, mesh.indices.len() as u32, 1, 0, 0, 1);
        }
    }

    // END RENDER PASS

    unsafe {
        // info!("end render pass");
        vk_bevy.device.cmd_end_render_pass(command_buffer);
    }

    // END

    vk_bevy.end_frame(image_index, command_buffer);

    {
        let (frame_gpu_begin, frame_gpu_end) = vk_bevy.get_frame_time();
        *frame_gpu_avg = *frame_gpu_avg * 0.95 + (frame_gpu_end - frame_gpu_begin) * 0.05;
        let frame_cpu = begin_frame.elapsed().as_secs_f64() * 1000.0;
        *frame_cpu_avg = *frame_cpu_avg * 0.95 + frame_cpu * 0.05;

        window.title = format!(
            "cpu: {:.2} ms gpu: {:.2} ms",
            *frame_cpu_avg, *frame_gpu_avg,
        );
    }
}

fn resize(
    windows: Query<&Window>,
    mut events: EventReader<WindowResized>,
    mut vk_bevy: ResMut<VkBevyInstance>,
) {
    for event in events.read() {
        if let Ok(window) = windows.get(event.window) {
            let width = window.physical_width();
            let height = window.physical_height();
            if width == 0 || height == 0 {
                continue;
            }
            if width != vk_bevy.swapchain_width || height != vk_bevy.swapchain_height {
                // FIXME: this will break with multiple windows
                vk_bevy.recreate_swapchain(width, height);
            }
        }
    }
}
