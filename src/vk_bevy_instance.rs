use std::{
    borrow::Cow,
    ffi::{c_char, CStr, CString},
    sync::{Arc, Mutex},
};

use crate::shader::{compile_shader, create_shader_module, parse_spirv, Shader};
use anyhow::bail;
use ash::{
    ext::debug_utils,
    khr::{surface, swapchain},
    vk::{self},
    Device, Entry, Instance,
};
use bevy::prelude::*;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
    MemoryLocation,
};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub struct Buffer {
    vk_buffer: Arc<Mutex<vk::Buffer>>,
    allocation: Arc<Mutex<Option<Allocation>>>,
    pub size: u64,
}

impl Buffer {
    pub fn write(&mut self, data: &[u8]) {
        if let Some(allocation) = self.allocation.lock().unwrap().as_mut() {
            let slice = allocation.mapped_slice_mut().unwrap();
            slice[..data.len()].copy_from_slice(data);
        }
    }

    #[must_use]
    pub fn vk_buffer(&self) -> vk::Buffer {
        *self.vk_buffer.lock().unwrap()
    }
}

#[derive(Clone)]
pub struct PipelineLayout {
    vk_pipeline_layout: Arc<Mutex<vk::PipelineLayout>>,
}

impl PipelineLayout {
    #[must_use]
    pub fn vk_pipeline_layout(&self) -> vk::PipelineLayout {
        *self.vk_pipeline_layout.lock().unwrap()
    }
}

pub struct Pipeline {
    pub layout: PipelineLayout,
    vk_pipeline: Arc<Mutex<vk::Pipeline>>,
}

impl Pipeline {
    #[must_use]
    pub fn vk_pipeline(&self) -> vk::Pipeline {
        *self.vk_pipeline.lock().unwrap()
    }
}

#[derive(Clone)]
pub struct Program {
    pub layout: PipelineLayout,
}

#[allow(unused)]
pub struct Image {
    vk_image: Arc<Mutex<vk::Image>>,
    vk_image_view: Arc<Mutex<vk::ImageView>>,
    allocation: Arc<Mutex<Option<Allocation>>>,
}

impl Image {
    #[allow(unused)]
    #[must_use]
    pub fn vk_image(&self) -> vk::Image {
        *self.vk_image.lock().unwrap()
    }

    #[allow(unused)]
    #[must_use]
    pub fn vk_image_view(&self) -> vk::ImageView {
        *self.vk_image_view.lock().unwrap()
    }
}

#[derive(Resource)]
pub struct VkBevyInstance {
    pub instance: Instance,
    pub device: Device,
    pub physical_device: vk::PhysicalDevice,
    pub debug_utils_loader: debug_utils::Instance,
    pub debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    pub swapchain_loader: swapchain::Device,
    pub swapchain_khr: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_width: u32,
    pub swapchain_height: u32,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub present_mode: vk::PresentModeKHR,
    pub surface_loader: surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub acquire_semaphore: vk::Semaphore,
    pub release_semaphore: vk::Semaphore,
    pub present_queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub render_pass: vk::RenderPass,
    pub pipeline_cache: vk::PipelineCache,
    pipeline_layouts: Vec<Arc<Mutex<vk::PipelineLayout>>>,
    pipelines: Vec<Arc<Mutex<vk::Pipeline>>>,
    shader_modules: Vec<Arc<Mutex<vk::ShaderModule>>>,
    #[allow(unused)]
    images: Vec<Arc<Mutex<vk::Image>>>,
    #[allow(unused)]
    image_views: Vec<Arc<Mutex<vk::ImageView>>>,
    allocator: std::mem::ManuallyDrop<Allocator>,
    allocations: Vec<Arc<Mutex<Option<Allocation>>>>,
    buffers: Vec<Arc<Mutex<vk::Buffer>>>,
    query_pool: vk::QueryPool,
}

impl VkBevyInstance {
    pub fn init(
        winit_window: &winit::window::Window,
        present_mode: vk::PresentModeKHR,
    ) -> anyhow::Result<Self> {
        let entry = Entry::linked();
        let instance =
            create_instance(&entry, "Cendre", winit_window).expect("Failed to create instance");

        let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
        let debug_utils_messenger = init_debug_utils_messenger(&debug_utils_loader)
            .expect("Failed to init debug utils messenger");

        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                winit_window.display_handle()?.as_raw(),
                winit_window.window_handle()?.as_raw(),
                None,
            )
            .expect("Failed to create surface")
        };
        let surface_loader = surface::Instance::new(&entry, &instance);

        let (physical_device, queue_family_index) =
            select_physical_device(&instance, &surface_loader, surface).expect("No GPU found");

        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        assert!(props.limits.timestamp_compute_and_graphics == 1);

        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index as u32)
            .queue_priorities(&[1.0]);

        let device_extension_names = vec![swapchain::NAME.as_ptr()];

        let mut physical_device_buffer_device_address_features =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default();

        let features = vk::PhysicalDeviceFeatures {
            shader_clip_distance: 1,
            ..Default::default()
        };

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names)
            .enabled_features(&features)
            .push_next(&mut physical_device_buffer_device_address_features);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        };

        info!("device created");

        let surface_format = get_surface_format(&surface_loader, physical_device, surface)
            .expect("Failed to get a surface format");

        info!("surface format: {surface_format:?}");

        let render_pass = create_render_pass(&device, surface_format).unwrap();

        info!("render pass created");

        let swapchain_loader = swapchain::Device::new(&instance, &device);

        let swapchain_width = winit_window.inner_size().width;
        let swapchain_height = winit_window.inner_size().height;

        let swapchain_khr = create_swapchain_khr(
            &swapchain_loader,
            &surface_loader,
            surface,
            surface_format,
            physical_device,
            swapchain_width,
            swapchain_height,
            None,
            present_mode,
        )
        .expect("Failed to create swapchain");
        info!("swapchain created");

        let (swapchain_images, swapchain_image_views, framebuffers) = create_swapchain_resources(
            &device,
            &swapchain_loader,
            swapchain_khr,
            render_pass,
            surface_format,
            swapchain_width,
            swapchain_height,
        );
        info!("swapchain resources created");

        let create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(128);
        let query_pool = unsafe { device.create_query_pool(&create_info, None).unwrap() };

        let command_pool = create_command_pool(&device, queue_family_index as u32);

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers = unsafe { device.allocate_command_buffers(&allocate_info).unwrap() };

        let acquire_semaphore = create_semaphore(&device).expect("Failed to create semaphore");
        let release_semaphore = create_semaphore(&device).expect("Failed to create semaphore");
        let present_queue = unsafe { device.get_device_queue(queue_family_index as u32, 0) };

        let create_info = vk::PipelineCacheCreateInfo::default();
        let pipeline_cache = unsafe { device.create_pipeline_cache(&create_info, None).unwrap() };

        let buffer_device_address =
            physical_device_buffer_device_address_features.buffer_device_address == 1;
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: default(),
            buffer_device_address,
            allocation_sizes: default(),
        })
        .unwrap();

        Ok(Self {
            instance,
            device,
            physical_device,
            debug_utils_loader,
            debug_utils_messenger,
            swapchain_loader,
            swapchain_khr,
            swapchain_images,
            swapchain_image_views,
            swapchain_width,
            swapchain_height,
            framebuffers,
            present_mode,
            surface_loader,
            surface,
            surface_format,
            acquire_semaphore,
            release_semaphore,
            present_queue,
            command_pool,
            command_buffers,
            render_pass,
            pipeline_cache,
            pipeline_layouts: vec![],
            pipelines: vec![],
            shader_modules: vec![],
            images: vec![],
            image_views: vec![],
            allocator: std::mem::ManuallyDrop::new(allocator),
            allocations: vec![],
            buffers: vec![],
            query_pool,
        })
    }

    pub fn create_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> anyhow::Result<Buffer> {
        let vk_info = vk::BufferCreateInfo::default().size(size).usage(usage);
        let buffer = unsafe { self.device.create_buffer(&vk_info, None)? };
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let allocation = self.allocator.allocate(&AllocationCreateDesc {
            name: &format!("usage: {usage:?} size: {size} "),
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        // Bind memory to the buffer
        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        };

        let buffer_raw = Arc::new(Mutex::new(buffer));
        let allocation_raw = Arc::new(Mutex::new(Some(allocation)));

        self.buffers.push(buffer_raw.clone());
        self.allocations.push(allocation_raw.clone());

        Ok(Buffer {
            vk_buffer: buffer_raw,
            allocation: allocation_raw,
            size,
        })
    }

    pub fn upload_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        scratch_buffer: &mut Buffer,
        buffer: &Buffer,
        data: &[u8],
    ) {
        assert!(scratch_buffer.size >= data.len() as u64);
        scratch_buffer.write(data);

        unsafe {
            self.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .expect("Failed to reset command_pool");
        }

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
        }

        let region = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(data.len() as u64);
        unsafe {
            self.device.cmd_copy_buffer(
                command_buffer,
                scratch_buffer.vk_buffer(),
                buffer.vk_buffer(),
                std::slice::from_ref(&region),
            );
        }

        unsafe {
            let copy_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .size(vk::WHOLE_SIZE)
                .buffer(buffer.vk_buffer());
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[copy_barrier],
                &[],
            );
        }

        unsafe {
            self.device.end_command_buffer(command_buffer).unwrap();
        }

        unsafe {
            let submits =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
            self.device
                .queue_submit(self.present_queue, &[submits], vk::Fence::null())
                .unwrap();
            self.device.queue_wait_idle(self.present_queue).unwrap();
        }
    }

    pub fn load_shader(&mut self, path: &str) -> Shader {
        let spv = compile_shader(path).expect("Failed to compile shader");

        let parse_info = parse_spirv(&spv).expect("Failed to parse spirv");

        let vk_shader_module = create_shader_module(&self.device, spv)
            .expect("Failed to create shader module from spirv");
        let vk_shader_module = Arc::new(Mutex::new(vk_shader_module));
        self.shader_modules.push(vk_shader_module.clone());
        let entry_point = CString::new(parse_info.entry_point).unwrap();
        Shader {
            vk_shader_module,
            entry_point,
            stage: parse_info.stage,
            storage_buffer_mask: parse_info.storage_buffer_mask,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_graphics_pipeline(
        &mut self,
        pipeline_layout: &PipelineLayout,
        render_pass: vk::RenderPass,
        stages: &[vk::PipelineShaderStageCreateInfo],
        primitive_topology: vk::PrimitiveTopology,
        rasterization_state: vk::PipelineRasterizationStateCreateInfo,
        vertex_input_binding_descriptions: &[vk::VertexInputBindingDescription],
        vertex_input_attribute_descriptions: &[vk::VertexInputAttributeDescription],
    ) -> anyhow::Result<Pipeline> {
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(vertex_input_attribute_descriptions)
            .vertex_binding_descriptions(vertex_input_binding_descriptions);

        let input_assembly_state =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(primitive_topology);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default();

        let color_attachment_state = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_attachment_state));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(*pipeline_layout.vk_pipeline_layout.lock().unwrap())
            .render_pass(render_pass);
        let graphics_pipelines = match unsafe {
            self.device.create_graphics_pipelines(
                self.pipeline_cache,
                std::slice::from_ref(&create_info),
                None,
            )
        } {
            Ok(pipelines) => pipelines,
            // For some reason the Err is a tuple and doesn't work with anyhow
            Err((_, result)) => {
                error!("{result:?}");
                bail!("Failed to create graphics pipelines")
            }
        };
        let pipeline = Arc::new(Mutex::new(graphics_pipelines[0]));
        self.pipelines.push(pipeline.clone());
        Ok(Pipeline {
            vk_pipeline: pipeline,
            layout: pipeline_layout.clone(),
        })
    }

    #[must_use]
    pub fn begin_frame(&self) -> (u32, vk::CommandBuffer) {
        #[cfg(feature = "trace")]
        let _span = bevy::utils::tracing::info_span!("begin frame").entered();

        let (image_index, _) = unsafe {
            self.swapchain_loader
                .acquire_next_image(
                    self.swapchain_khr,
                    std::u64::MAX,
                    self.acquire_semaphore,
                    vk::Fence::null(),
                )
                .expect("Failed to acquire next image")
        };

        unsafe {
            self.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .expect("Failed to reset command_pool");
        }

        let command_buffer = self.command_buffers[0];

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
        }

        unsafe {
            self.device
                .cmd_reset_query_pool(command_buffer, self.query_pool, 0, 128);
            self.device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                0,
            );
        }

        unsafe {
            let render_begin_barrier = image_barrier(
                self.swapchain_images[image_index as usize],
                vk::AccessFlags::empty(),
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[render_begin_barrier],
            );
        }
        (image_index, command_buffer)
    }

    pub fn end_frame(&self, image_index: u32, command_buffer: vk::CommandBuffer) {
        #[cfg(feature = "trace")]
        let _span = bevy::utils::tracing::info_span!("end frame").entered();

        unsafe {
            let render_end_barrier = image_barrier(
                self.swapchain_images[image_index as usize],
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::empty(),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[render_end_barrier],
            );
        }

        unsafe {
            self.device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                1,
            );
        }

        unsafe {
            #[cfg(feature = "trace")]
            let _span = bevy::utils::tracing::info_span!("end command buffer").entered();
            self.device.end_command_buffer(command_buffer).unwrap();
        }

        {
            #[cfg(feature = "trace")]
            let _span = bevy::utils::tracing::info_span!("submit").entered();
            self.submit();
        }
        {
            #[cfg(feature = "trace")]
            let _span = bevy::utils::tracing::info_span!("present").entered();
            self.present(image_index);
        }

        unsafe {
            #[cfg(feature = "trace")]
            let _span = bevy::utils::tracing::info_span!("wait idle").entered();
            self.device.device_wait_idle().unwrap();
        }
    }

    #[must_use]
    pub fn get_frame_time(&self) -> (f64, f64) {
        let mut data: [i64; 2] = [0, 0];
        unsafe {
            self.device
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut data,
                    vk::QueryResultFlags::TYPE_64,
                )
                .unwrap();
        }

        let props = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        let period = f64::from(props.limits.timestamp_period);
        let to_ms = 1e-6;
        let frame_gpu_begin = (data[0] as f64) * period * to_ms;
        let frame_gpu_end = (data[1] as f64) * period * to_ms;
        (frame_gpu_begin, frame_gpu_end)
    }

    pub fn submit(&self) {
        let acquire_semaphores = [self.acquire_semaphore];
        let release_semaphores = [self.release_semaphore];
        unsafe {
            let submits = [vk::SubmitInfo::default()
                .wait_semaphores(&acquire_semaphores)
                .wait_dst_stage_mask(std::slice::from_ref(
                    &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                ))
                .command_buffers(&self.command_buffers)
                .signal_semaphores(&release_semaphores)];
            self.device
                .queue_submit(self.present_queue, &submits, vk::Fence::null())
                .unwrap();
        }
    }

    pub fn present(&self, image_index: u32) {
        let release_semaphores = [self.release_semaphore];
        unsafe {
            let present_info = vk::PresentInfoKHR::default()
                .swapchains(std::slice::from_ref(&self.swapchain_khr))
                .image_indices(std::slice::from_ref(&image_index))
                .wait_semaphores(&release_semaphores);
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
                .expect("Failed to queue present");
        }
    }

    pub fn set_viewport(&self, command_buffer: vk::CommandBuffer, width: u32, height: u32) {
        let viewport = vk::Viewport {
            width: width as f32,
            height: height as f32,
            max_depth: 1.0,
            ..default()
        };
        let scissor = vk::Rect2D {
            extent: vk::Extent2D { width, height },
            ..default()
        };

        unsafe {
            self.device
                .cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));
        };
        unsafe {
            self.device
                .cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));
        };
    }

    pub fn bind_pipeline(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline_bind_point: vk::PipelineBindPoint,
        pipeline: &Pipeline,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                pipeline_bind_point,
                pipeline.vk_pipeline(),
            );
        }
    }

    pub fn bind_vertex_buffers(
        &self,
        command_buffer: vk::CommandBuffer,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(command_buffer, first_binding, buffers, offsets);
        }
    }

    pub fn bind_index_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: &Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        unsafe {
            self.device.cmd_bind_index_buffer(
                command_buffer,
                buffer.vk_buffer(),
                offset,
                index_type,
            );
        }
    }

    pub fn draw_indexed(
        &self,
        command_buffer: vk::CommandBuffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            self.device.cmd_draw_indexed(
                command_buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }

    pub fn recreate_swapchain(&mut self, width: u32, height: u32) {
        println!("recreate swap chain");
        let new_swapchain_khr = create_swapchain_khr(
            &self.swapchain_loader,
            &self.surface_loader,
            self.surface,
            self.surface_format,
            self.physical_device,
            width,
            height,
            Some(self.swapchain_khr),
            self.present_mode,
        )
        .expect("Failed to create swapchain");
        let (swapchain_images, swapchain_image_views, framebuffers) = create_swapchain_resources(
            &self.device,
            &self.swapchain_loader,
            new_swapchain_khr,
            self.render_pass,
            self.surface_format,
            width,
            height,
        );

        unsafe { self.device.device_wait_idle().unwrap() };
        self.destroy_swapchain();

        self.swapchain_width = width;
        self.swapchain_height = height;

        self.swapchain_khr = new_swapchain_khr;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_views = swapchain_image_views;
        self.framebuffers = framebuffers;
    }

    fn destroy_swapchain(&mut self) {
        unsafe {
            for framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }

            for image_view in &self.swapchain_image_views {
                self.device.destroy_image_view(*image_view, None);
            }

            self.swapchain_loader
                .destroy_swapchain(self.swapchain_khr, None);
        }
    }

    pub fn create_pipeline_layout(
        &mut self,
        shaders: &[&Shader],
    ) -> anyhow::Result<PipelineLayout> {
        let set_layout = self.create_set_layout(shaders)?;

        let create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout));
        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&create_info, None)? };

        let vk_pipeline_layout = Arc::new(Mutex::new(pipeline_layout));
        self.pipeline_layouts.push(vk_pipeline_layout.clone());

        Ok(PipelineLayout { vk_pipeline_layout })
    }

    fn create_set_layout(
        &mut self,
        shaders: &[&Shader],
    ) -> anyhow::Result<vk::DescriptorSetLayout> {
        let mut storage_mask = 0;
        for shader in shaders {
            storage_mask |= shader.storage_buffer_mask;
        }
        let mut bindings = vec![];
        for i in 0..32 {
            if storage_mask & (1 << i) > 0 {
                let mut stage_flags = vk::ShaderStageFlags::empty();
                for shader in shaders {
                    if shader.storage_buffer_mask & (1 << i) > 0 {
                        stage_flags |= shader.stage;
                    }
                }
                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .stage_flags(stage_flags),
                );
            }
        }
        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        Ok(unsafe {
            self.device
                .create_descriptor_set_layout(&create_info, None)?
        })
    }
}

fn create_instance(
    entry: &Entry,
    app_name: &str,
    winit_window: &winit::window::Window,
) -> anyhow::Result<ash::Instance> {
    let app_name = CString::new(app_name.to_string())?;

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(0)
        .engine_name(&app_name)
        .engine_version(0)
        .api_version(vk::make_api_version(0, 1, 1, 0));

    let debug_layers_raw: Vec<*const c_char> =
        [unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") }]
            .iter()
            .map(|raw_name: &&CStr| raw_name.as_ptr())
            .collect();

    let mut extension_names =
        ash_window::enumerate_required_extensions(winit_window.display_handle()?.as_raw())?
            .to_vec();
    {
        extension_names.push(debug_utils::NAME.as_ptr());
    }

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .flags(vk::InstanceCreateFlags::default())
        .enabled_layer_names(&debug_layers_raw)
        .enabled_extension_names(&extension_names);

    Ok(unsafe { entry.create_instance(&create_info, None)? })
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    let message_format =
        format!("{message_type:?} [{message_id_name} ({message_id_number})]:\n{message}\n");

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => trace!("{message_format}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => info!("{message_format}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => warn!("{message_format}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!("{message_format}");
            panic!("VULKAN VALIDATION ERROR");
        }
        _ => panic!("Unknown message severity"),
    }

    vk::FALSE
}

fn init_debug_utils_messenger(
    debug_utils_loader: &debug_utils::Instance,
) -> anyhow::Result<vk::DebugUtilsMessengerEXT> {
    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));
    Ok(unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_info, None)? })
}

fn get_queue_family_index(
    instance: &Instance,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Option<usize> {
    for (index, props) in
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
            .iter()
            .enumerate()
    {
        let supports_graphic_and_surface = props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && unsafe {
                surface_loader
                    .get_physical_device_surface_support(physical_device, index as u32, surface)
                    .is_ok()
            };
        if supports_graphic_and_surface {
            return Some(index);
        }
    }
    None
}

fn select_physical_device(
    instance: &Instance,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> Option<(vk::PhysicalDevice, usize)> {
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate devices")
    };

    let mut fallback = None;
    for physical_device in &physical_devices {
        let props = unsafe { instance.get_physical_device_properties(*physical_device) };

        if props.api_version < vk::API_VERSION_1_1 {
            continue;
        }

        let Some(queue_family_index) =
            get_queue_family_index(instance, surface_loader, surface, *physical_device)
        else {
            continue;
        };

        if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            let device_name = unsafe { c_char_buf_to_string(props.device_name.as_ptr()) };
            info!("Using discrete GPU {:?}", device_name);
            return Some((*physical_device, queue_family_index));
        }

        if fallback.is_none() {
            fallback = Some((*physical_device, queue_family_index));
        }
    }
    if let Some((physical_device, _)) = fallback {
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe { c_char_buf_to_string(props.device_name.as_ptr()) };
        info!("Using fallback device {:?}", device_name);
    }
    fallback
}

fn get_surface_format(
    surface_loader: &surface::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> anyhow::Result<vk::SurfaceFormatKHR> {
    let formats =
        unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface)? };
    if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
        Ok(vk::SurfaceFormatKHR::default().format(vk::Format::R8G8B8A8_UNORM))
    } else if let Some(format) = formats.iter().find(|format| {
        format.format == vk::Format::R8G8B8A8_UNORM || format.format == vk::Format::B8G8R8A8_UNORM
    }) {
        Ok(*format)
    } else {
        Ok(formats[0])
    }
}

fn create_render_pass(
    device: &Device,
    surface_format: vk::SurfaceFormatKHR,
) -> anyhow::Result<vk::RenderPass> {
    let dependencies = [vk::SubpassDependency {
        src_subpass: vk::SUBPASS_EXTERNAL,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ..Default::default()
    }];
    let color_attachment_refs = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let subpass = vk::SubpassDescription::default()
        .color_attachments(&color_attachment_refs)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

    let attachment = vk::AttachmentDescription::default()
        .format(surface_format.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&dependencies);
    Ok(unsafe { device.create_render_pass(&create_info, None)? })
}

#[allow(clippy::too_many_arguments)]
fn create_swapchain_khr(
    swapchain_loader: &swapchain::Device,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    old_swapchain: Option<vk::SwapchainKHR>,
    present_mode: vk::PresentModeKHR,
) -> anyhow::Result<vk::SwapchainKHR> {
    let surface_capabilities = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
    };

    let mut desired_image_count = surface_capabilities.min_image_count + 1;
    if surface_capabilities.max_image_count > 0
        && desired_image_count > surface_capabilities.max_image_count
    {
        desired_image_count = surface_capabilities.max_image_count;
    }

    let surface_resolution = match surface_capabilities.current_extent.width {
        std::u32::MAX => vk::Extent2D { width, height },
        _ => surface_capabilities.current_extent,
    };

    let pre_transform = if surface_capabilities
        .supported_transforms
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        surface_capabilities.current_transform
    };

    let composite_alpha = match surface_capabilities.supported_composite_alpha {
        vk::CompositeAlphaFlagsKHR::OPAQUE
        | vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
        | vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED => {
            surface_capabilities.supported_composite_alpha
        }
        _ => vk::CompositeAlphaFlagsKHR::INHERIT,
    };

    let present_modes = unsafe {
        surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
    };
    let present_mode = present_modes
        .iter()
        .copied()
        .find(|&mode| mode == present_mode)
        .unwrap_or(vk::PresentModeKHR::FIFO);

    let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(desired_image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(surface_resolution)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(pre_transform)
        .composite_alpha(composite_alpha)
        .present_mode(present_mode);
    if let Some(old_swapchain) = old_swapchain {
        swapchain_create_info.old_swapchain = old_swapchain;
    }
    let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };
    Ok(swapchain)
}

fn create_image_view(
    device: &Device,
    format: vk::Format,
    image: vk::Image,
) -> anyhow::Result<vk::ImageView> {
    let aspect_mask = if format == vk::Format::D32_SFLOAT {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    };
    let create_view_info = vk::ImageViewCreateInfo::default()
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(vk::ComponentMapping {
            r: vk::ComponentSwizzle::R,
            g: vk::ComponentSwizzle::G,
            b: vk::ComponentSwizzle::B,
            a: vk::ComponentSwizzle::A,
        })
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect_mask)
                .layer_count(1)
                .level_count(1),
        )
        .image(image);
    Ok(unsafe { device.create_image_view(&create_view_info, None)? })
}

fn create_frame_buffer(
    device: &Device,
    render_pass: vk::RenderPass,
    image_view: vk::ImageView,
    width: u32,
    height: u32,
) -> anyhow::Result<vk::Framebuffer> {
    println!("create_frame_buffer {} {}", width, height);
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(std::slice::from_ref(&image_view))
        .width(width)
        .height(height)
        .layers(1);
    Ok(unsafe { device.create_framebuffer(&create_info, None)? })
}

fn create_swapchain_resources(
    device: &Device,
    swapchain_loader: &swapchain::Device,
    swapchain_khr: vk::SwapchainKHR,
    render_pass: vk::RenderPass,
    surface_format: vk::SurfaceFormatKHR,
    width: u32,
    height: u32,
) -> (Vec<vk::Image>, Vec<vk::ImageView>, Vec<vk::Framebuffer>) {
    let mut swapchain_image_views = vec![];
    let mut framebuffers = vec![];
    let swapchain_images = unsafe {
        swapchain_loader
            .get_swapchain_images(swapchain_khr)
            .expect("Failed to get swapchain images")
    };
    for image in &swapchain_images {
        let image_view = create_image_view(device, surface_format.format, *image)
            .expect("Failed to create image view for swapchain image");
        let fb = create_frame_buffer(device, render_pass, image_view, width, height)
            .expect("Failed to create frame buffer");
        swapchain_image_views.push(image_view);
        framebuffers.push(fb);
    }
    (swapchain_images, swapchain_image_views, framebuffers)
}

fn create_command_pool(device: &Device, queue_family_index: u32) -> vk::CommandPool {
    let pool_create_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);
    unsafe { device.create_command_pool(&pool_create_info, None).unwrap() }
}

fn create_semaphore(device: &Device) -> anyhow::Result<vk::Semaphore> {
    let semaphore_create_info = vk::SemaphoreCreateInfo::default();
    Ok(unsafe { device.create_semaphore(&semaphore_create_info, None)? })
}

fn image_barrier<'a>(
    image: vk::Image,
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> vk::ImageMemoryBarrier<'a> {
    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .layer_count(vk::REMAINING_ARRAY_LAYERS);

    vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access_mask)
        .old_layout(old_layout)
        .dst_access_mask(dst_access_mask)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
}

// Helper function to convert c_char buffer to a str
unsafe fn c_char_buf_to_string<'a>(buf: *const c_char) -> Cow<'a, str> {
    unsafe { CStr::from_ptr(buf) }.to_string_lossy()
}
