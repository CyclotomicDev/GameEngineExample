use anyhow::{Result,anyhow};
use winit::window::{Window, WindowBuilder};


pub struct GraphicsHandler {
    vulkan_handler: vulkan::VulkanHandler,
}

impl GraphicsHandler {
    pub fn new(window: &Window) -> Result<Self> {
        let vulkan_handler = vulkan::VulkanHandler::new(window)?;
        Ok(Self {vulkan_handler})
    }

    pub fn render(&mut self, window: &Window) -> Result<()> {
        self.vulkan_handler.render(window)?;
        Ok(())
    }


}


/// For holding low-level (unsafe) Vulkan
mod vulkan {
    // General imports
    use winit::window::{Window, WindowBuilder};
    use anyhow::{anyhow, Context, Ok, Result};
    use thiserror::Error;
    use log::*;

    use vulkanalia::loader::{LibloadingLoader, LIBRARY};
    use vulkanalia::{bytecode, window as vk_window};
    use vulkanalia::prelude::v1_0::*;
    use vulkanalia::vk::{ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension, PipelineColorBlendAttachmentState};

    use std::collections::HashSet;
    use std::ffi::CStr;
    use std::os::raw::c_void;

    
    // Compatibility
    use vulkanalia::Version;
    const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

    // Validation layers
    const VALIDATION_ENABLED: bool =
        cfg!(debug_assertions);

    const VALIDATION_LAYER: vk::ExtensionName =
        vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

    // Extensions
    const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

    extern "system" fn debug_callback(
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        type_: vk::DebugUtilsMessageTypeFlagsEXT,
        data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _: *mut c_void,
    ) -> vk::Bool32 {
        let data = unsafe {
            *data
        };
        let message = unsafe {
            CStr::from_ptr(data.message).to_string_lossy()
        };

        if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
            error!("({:?}) {}", type_, message);
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
            warn!("({:?}) {}", type_, message);
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
            debug!("({:?}) {}", type_, message);
        } else {
            trace!("({:?}) {}", type_, message);
        }

        vk::FALSE
    }

   // Note: drop executed in order first to last
    pub struct VulkanHandler {
        device: DeviceWrapper, // Requires instance
        instance: InstanceWrapper,
    }

    impl VulkanHandler {
        pub fn new(window: &Window) -> Result<Self> {
            let instance = InstanceWrapper::new(window)?;
            let device = DeviceWrapper::new(&instance, window)?;
            Ok(Self {instance, device})
        }

        pub fn render(&mut self, window: &Window) -> Result<()> {
            let image_available_semaphore = self.device.command.image_available_semaphore;
            let render_finsished_semaphore = self.device.command.render_finished_semaphore;
            
            let swapchain = self.device.swapchain.swapchain;

            // 1. Aquire image from swapchain
            let image_index = unsafe {
                self.device.logical_device
                    .acquire_next_image_khr(
                        swapchain, 
                        u64::MAX, 
                        image_available_semaphore, 
                        vk::Fence::null(),
                    )?
                    .0 as usize
            };

            // 2. Execute command buffer with image as attachment in the framebuffer

            let wait_semaphores = &[image_available_semaphore];
            let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = &[self.device.command.command_buffers[image_index as usize]];
            let signal_semaphores = &[render_finsished_semaphore];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(wait_stages)
                .command_buffers(command_buffers)
                .signal_semaphores(signal_semaphores);

            unsafe {
                self.device.logical_device.queue_submit(self.device.graphics_queue, &[submit_info], vk::Fence::null())?  
            };

            let swapchains = &[swapchain];
            let image_indices = &[image_index as u32];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(signal_semaphores)
                .swapchains(swapchains)
                .image_indices(image_indices);

            unsafe {
                self.device.logical_device.queue_present_khr(self.device.present_queue, &present_info)?;
            }

            Ok(())
        }
    
        
    }

    impl Drop for VulkanHandler {
        fn drop(&mut self) {
            
        }
    }

    struct InstanceWrapper {
        entry: Entry,
        messenger: Option<vk::DebugUtilsMessengerEXT>,
        surface: vk::SurfaceKHR,
        instance: Instance,
    }

    impl InstanceWrapper {
        // Safety not yet guarenteed
        fn new(window: &Window) -> Result<Self> {

            // Entry setup

            let entry = {
                // Opens Vulkan dll: safety based on trust of the Vulkan dll
                let loader = unsafe {
                    LibloadingLoader::new(LIBRARY)?
                };

                // Unknown safety requirements
                unsafe {
                    Entry::new(loader).map_err(|b| anyhow!("{}",b).context("Entry point failure"))?
                }
            };
            
            // Instance setup
            let (instance, messenger) = {

                // Gives information about setup - only useful for well-known engines for optimization
                let application_info = vk::ApplicationInfo::builder()
                    .application_name(b"G03Apha\0")
                    .application_version(vk::make_version(1, 0, 0))
                    .engine_name(b"G03\0")
                    .engine_version(vk::make_version(1, 0, 0))
                    .api_version(vk::make_version(1, 0, 0));

                // Safety if entry initialized
                let available_layers = unsafe {
                    entry
                        .enumerate_instance_layer_properties()?
                        .iter()
                        .map(|l| l.layer_name)
                        .collect::<HashSet<_>>()
                };

                if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
                    return Err(anyhow!("Validation layer requested, but not supported"));
                }

                let layers = if VALIDATION_ENABLED {
                    vec![VALIDATION_LAYER.as_ptr()]
                } else {
                    Vec::new()
                };


                let mut extensions = 
                    vk_window::get_required_instance_extensions(window)
                        .iter()
                        .map(|e| e.as_ptr())
                        .collect::<Vec<_>>();
                
                if VALIDATION_ENABLED {
                    extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
                }

                // Required by Vulkan SDK on macOS since 1.3.216
                let flags = if 
                    cfg!(target_os = "macos") &&
                    entry.version()? >= PORTABILITY_MACOS_VERSION
                {
                    info!("Enabling extensions for macOS portability");
                    extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name.as_ptr());
                    extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
                    vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
                } else {
                    vk::InstanceCreateFlags::empty()
                };

                let mut info = vk::InstanceCreateInfo::builder()
                    .application_info(&application_info)
                    .enabled_layer_names(&layers)
                    .enabled_extension_names(&extensions)
                    .flags(flags);

                let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                    .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
                    .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                    .user_callback(Some(debug_callback));

                if VALIDATION_ENABLED {
                    info = info.push_next(&mut debug_info);
                }

                // Unknown safety requirements!!!
                let instance = unsafe {
                    entry.create_instance(&info, None)?
                }; 

                // Debug setup

                let messenger = if VALIDATION_ENABLED {
                    // Safety requires valid instance
                    Some(unsafe {
                        instance.create_debug_utils_messenger_ext(&debug_info, None)?
                    })
                } else {
                    None
                };

                (instance, messenger)                
            };
            
            // Safety: window must be valid
            let surface = unsafe {
                vk_window::create_surface(&instance, window, window)?
            };

            

            Ok(Self {entry, instance, messenger, surface})
        }
    }

    impl Drop for InstanceWrapper {
        fn drop(&mut self) {
            unsafe{
                self.instance.destroy_surface_khr(self.surface, None);
            }

            if let Some(messenger) = self.messenger {
                // Safety requires existance of messenger
                unsafe {
                    self.instance.destroy_debug_utils_messenger_ext(messenger,None)
                };
            }

            info!("Before instance destroyed");

            // Safety:
            // All child objects must be destroyed BEFORE
            // No AllocationCallbacks provided, so value None MUST match creation
            unsafe {
                self.instance.destroy_instance(None);
            }
        }
    }

    #[derive(Debug, Error)]
    #[error("Missing {0}")]
    struct SuitabilityError(&'static str);
    struct DeviceWrapper {
        graphics_queue: vk::Queue,
        present_queue: vk::Queue,
        command: CommandWrapper,
        framebuffers: Vec<vk::Framebuffer>,
        pipeline: PipelineWrapper,
        swapchain: SwapchainWrapper, // Requires 'logical_device'
        physical_device: vk::PhysicalDevice,
        logical_device: Device,
    }

    impl DeviceWrapper {
        fn new(instance: &InstanceWrapper, window: &Window) -> Result<Self> {
            unsafe {
                for physical_device in instance.instance.enumerate_physical_devices()? {
                    let properties = instance.instance.get_physical_device_properties(physical_device);

                    if let Err(error) = DeviceWrapper::check_physical_device(instance, physical_device) {
                        warn!("Skipping physical device (`{}`): {}", properties.device_name, error);
                    } else {
                        info!("Selected physical device (`{}`)", properties.device_name);
                        let (logical_device, graphics_queue, present_queue) = DeviceWrapper::create_logical_device(instance, physical_device)?;
                        let swapchain = SwapchainWrapper::new(instance, physical_device, &logical_device, window)?;
                        let pipeline = PipelineWrapper::new(&logical_device, swapchain.swapchain_extent, swapchain.swapchain_format)?;
                        let framebuffers = DeviceWrapper::create_framebuffers(&logical_device, &swapchain, &pipeline.render_pass)?;
                        let command = CommandWrapper::new(&instance, physical_device, &logical_device, &framebuffers, &swapchain, &pipeline)?;
                        return Ok(Self { physical_device, logical_device, graphics_queue, present_queue, swapchain, pipeline, framebuffers, command });
                    }
                }
            }

            Err(anyhow!("Failed to find suitable physical device."))
        }

        fn check_physical_device(instance: &InstanceWrapper, physical_device: vk::PhysicalDevice) -> Result<()> {
            {
                let instance = &instance.instance;
                // Safety: physical_device must be valid
                let properties = unsafe {
                    instance
                .get_physical_device_properties(physical_device)
                };
                if properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
                    return Err(anyhow!(SuitabilityError("Only discrete GPUs are supported")));
                }

                let features = unsafe {
                    instance
                        .get_physical_device_features(physical_device)
                };
                if features.geometry_shader != vk::TRUE {
                    return Err(anyhow!(SuitabilityError("Missing geometry shader support")));
                }
            }

            DeviceWrapper::check_physical_device_extensions(instance, physical_device)?;


            let support = SwapchainSupport::new(instance, physical_device)?;
            if support.formats.is_empty() || support.present_modes.is_empty() {
                return Err(anyhow!(SuitabilityError("Insufficient swapchain support")));
            }

            Ok(())
        }
    
        fn create_logical_device(instance: &InstanceWrapper, physical_device: vk::PhysicalDevice) -> Result<(Device, vk::Queue, vk::Queue)> {
            let indices = QueueFamilyIndices::new(instance, physical_device)?;

            let device = {
                // From 0.0 to 1.0, determines how scheduling of command buffer execution works
                let queue_priorities = &[1.0];
                let queue_infos = 
                {
                    let mut unique_indices = HashSet::new();
                    unique_indices.insert(indices.graphics);
                    unique_indices.insert(indices.present);

                    unique_indices
                        .iter()
                        .map(|i| {
                            vk::DeviceQueueCreateInfo::builder()
                                .queue_family_index(*i)
                                .queue_priorities(queue_priorities)
                        })
                        .collect::<Vec<_>>()
                };


                // Specifies validation layers to enable (for debugging)
                let layers = if VALIDATION_ENABLED {
                    vec![VALIDATION_LAYER.as_ptr()]
                } else {
                    vec![]
                };

                let mut extensions = DEVICE_EXTENSIONS
                    .iter()
                    .map(|n| n.as_ptr())
                    .collect::<Vec<_>>();
                // Required by Vulkan SDK on macOS since 1.3.216.
                if cfg!(target_os = "macos") && instance.entry.version()? >= PORTABILITY_MACOS_VERSION {
                    extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
                }

                let features = vk::PhysicalDeviceFeatures::builder();

                let info = vk::DeviceCreateInfo::builder()
                    .queue_create_infos(&queue_infos)
                    .enabled_layer_names(&layers)
                    .enabled_extension_names(&extensions)
                    .enabled_features(&features);

                unsafe {
                    instance.instance.create_device(physical_device, &info, None)?
                }
            };

            let graphics_queue = unsafe {
                device.get_device_queue(indices.graphics, 0)
            };

            let present_queue = unsafe {
                device.get_device_queue(indices.present, 0)
            };

            Ok((device, graphics_queue, present_queue))

        }
        
        fn check_physical_device_extensions(instance: &InstanceWrapper,physical_device: vk::PhysicalDevice) -> Result<()> {
            let instance = &instance.instance;
            
            let extensions = unsafe {
                instance
                    .enumerate_device_extension_properties(physical_device, None)?
                    .iter()
                    .map(|e| e.extension_name)
                    .collect::<HashSet<_>>()
            };

            if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
                Ok(())
            } else {
                Err(anyhow!(SuitabilityError("Missing required device extensions.")))
            }
        }
    
        fn create_framebuffers(logical_device: &Device, swapchain: &SwapchainWrapper, render_pass: &RenderPassWeapper) -> Result<Vec<vk::Framebuffer>> {
            let framebuffers = 
                swapchain.swapchain_image_views
                .iter()
                .map(|i| {
                    let attachments = &[*i];
                    let create_info = vk::FramebufferCreateInfo::builder()
                        .render_pass(render_pass.render_pass)
                        .attachments(attachments)
                        .width(swapchain.swapchain_extent.width)
                        .height(swapchain.swapchain_extent.height)
                        .layers(1);

                    unsafe {
                        logical_device.create_framebuffer(&create_info, None)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;

            Ok(framebuffers)
        }
   
        
    }

    impl Drop for DeviceWrapper {
        fn drop(&mut self) {
            
            self.command.destroy(&self.logical_device);
            self.pipeline.destroy(&self.logical_device);
            self.swapchain.destroy(&self.logical_device);

            unsafe {
                
                self.framebuffers
                    .iter()
                    .for_each(|f| self.logical_device.destroy_framebuffer(*f, None));

                self.logical_device.destroy_device(None);
            }
        }
    }

    struct QueueFamilyIndices {
        graphics: u32,
        present: u32,
    }

    impl QueueFamilyIndices {
        fn new(instance: &InstanceWrapper, physical_device: vk::PhysicalDevice) -> Result<Self> {
            let InstanceWrapper { entry: _, messenger: _, surface, instance } = instance;
            
            let properties = unsafe {
                instance
                    .get_physical_device_queue_family_properties(physical_device)
            };

            let graphics = properties
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|i| i as u32);

            let mut present = None;
            for (index, properties) in properties.iter().enumerate() {
                unsafe {
                    if instance.get_physical_device_surface_support_khr(physical_device, index as u32, *surface)? {
                        present = Some(index as u32);
                        break;
                    }
                }
            }

            if let (Some(graphics), Some(present)) = (graphics, present) {
                Ok(Self { graphics, present })
            } else {
                Err(anyhow!(SuitabilityError("Missing required queue families")))
            }
        }
    }

    #[derive(Clone, Debug)]
    struct SwapchainSupport {
        capabilities: vk::SurfaceCapabilitiesKHR,
        formats: Vec<vk::SurfaceFormatKHR>,
        present_modes: Vec<vk::PresentModeKHR>,
    }

    impl SwapchainSupport {
        fn new(instance: &InstanceWrapper, physical_device: vk::PhysicalDevice) -> Result<Self> {
            let InstanceWrapper { entry: _, messenger: _, surface, instance } = instance;
            Ok(Self { 
                capabilities: unsafe {
                    instance.
                        get_physical_device_surface_capabilities_khr(physical_device, *surface)?
                }, 
                formats: unsafe {
                    instance.
                        get_physical_device_surface_formats_khr(physical_device, *surface)?
                }, 
                present_modes:  unsafe {
                    instance.
                        get_physical_device_surface_present_modes_khr(physical_device, *surface)?
                } })
        }
    
        // Format for communicating with display; attempt to pick display format closest to intent
        fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
            formats
                .iter()
                .cloned()
                .find(|f| {
                    f.format == vk::Format::B8G8R8A8_SRGB
                        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .unwrap_or_else(|| formats[0])
        }

        // Attempt to pick best presentation (how frames are refreshed)
        fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
            present_modes
                .iter()
                .cloned()
                .find(|m| *m == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO)
        }
        
        // Allows to set resolution (to same as window)
        fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
            if capabilities.current_extent.width != u32::MAX {
                capabilities.current_extent
            } else {
                let size = window.inner_size();
                let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));
                vk::Extent2D::builder()
                    .width(clamp(
                        capabilities.min_image_extent.width,
                        capabilities.max_image_extent.width,
                        size.width,
                    ))
                    .height(clamp(
                        capabilities.min_image_extent.height,
                        capabilities.max_image_extent.height,
                        size.height,
                    ))
                    .build()
            }
        }
    }

    struct SwapchainWrapper {
        swapchain_format: vk::Format,
        swapchain_extent: vk::Extent2D,
        swapchain: vk::SwapchainKHR,
        swapchain_images: Vec<vk::Image>,
        swapchain_image_views: Vec<vk::ImageView>,
    }

    impl SwapchainWrapper {
        fn new(instance: &InstanceWrapper, physical_device: vk::PhysicalDevice, logical_device: &Device, window: &Window) -> Result<Self> {

            let physical_device = physical_device;

            let support = SwapchainSupport::new(&instance, physical_device)?;
            let surface_format = SwapchainSupport::get_swapchain_surface_format(&support.formats);
            let present_mode = SwapchainSupport::get_swapchain_present_mode(&support.present_modes);
            let swapchain_extent = SwapchainSupport::get_swapchain_extent(window, support.capabilities);

            let swapchain = {
                
                let indices = QueueFamilyIndices::new(&instance, physical_device)?;
                

                

                let image_count = {
                    let image_count = support.capabilities.min_image_count + 1;
                    if support.capabilities.max_image_count != 0
                        && image_count > support.capabilities.max_image_count 
                    {
                        support.capabilities.max_image_count
                    } else {
                        image_count
                    }
                };

                let mut queue_family_indices = vec![];
                let image_sharing_mode = if indices.graphics != indices.present {
                    queue_family_indices.push(indices.graphics);
                    queue_family_indices.push(indices.present);
                    vk::SharingMode::CONCURRENT
                } else {
                    vk::SharingMode::EXCLUSIVE
                };

                let info = vk::SwapchainCreateInfoKHR::builder()
                    .surface(instance.surface)
                    .min_image_count(image_count)
                    .image_format(surface_format.format)
                    .image_color_space(surface_format.color_space)
                    .image_extent(swapchain_extent)
                    .image_array_layers(1)
                    .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(image_sharing_mode)
                    .queue_family_indices(&queue_family_indices)
                    .pre_transform(support.capabilities.current_transform)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(present_mode)
                    .clipped(true)
                    .old_swapchain(vk::SwapchainKHR::null());

                unsafe {
                        logical_device.create_swapchain_khr(&info, None)?
                }
            }; 

            let swapchain_images = unsafe {
                logical_device.get_swapchain_images_khr(swapchain)?
            };

            let swapchain_image_views = {
                swapchain_images
                    .iter()
                    .map(|i| {
                        SwapchainWrapper::create_image_view(surface_format.format, logical_device, *i, vk::ImageAspectFlags::COLOR)
                    })
                    .collect::<Result<Vec<_>,_>>()?
            };

            Ok(Self { swapchain_format: surface_format.format, swapchain_extent, swapchain, swapchain_images, swapchain_image_views })
        }

        fn create_image_view(swapchain_format: vk::Format, logical_device: &Device, image: vk::Image, aspects: vk::ImageAspectFlags) -> Result<vk::ImageView> {
            let components = vk::ComponentMapping::builder()
                .r(vk::ComponentSwizzle::IDENTITY)
                .g(vk::ComponentSwizzle::IDENTITY)
                .b(vk::ComponentSwizzle::IDENTITY)
                .a(vk::ComponentSwizzle::IDENTITY);

            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let info = vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::_2D)
                .format(swapchain_format)
                .components(components)
                .subresource_range(subresource_range);

            let image_view = unsafe {
                logical_device.create_image_view(&info, None)?
            };

            Ok(image_view)

        }

        fn destroy(&mut self, logical_device: &Device) {
            unsafe {
                self.swapchain_image_views
                    .iter()
                    .for_each(|v| logical_device.destroy_image_view(*v, None));

                logical_device.destroy_swapchain_khr(self.swapchain, None);
            }
        }
    }

    struct PipelineWrapper {
        pipeline_layout: vk::PipelineLayout,
        render_pass: RenderPassWeapper,
        pipeline: vk::Pipeline,
    }

    impl PipelineWrapper {
        fn new(logical_device: &Device, swapchain_extent: vk::Extent2D, swapchain_format: vk::Format) -> Result<Self> {
            let vert = include_bytes!("../shaders/vert.spv");
            let frag = include_bytes!("../shaders/frag.spv");

            let vert_shader_module = PipelineWrapper::create_shader_module(logical_device, &vert[..])?;
            let frag_shader_module = PipelineWrapper::create_shader_module(logical_device, &frag[..])?;

            let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader_module)
                .name(b"main\0");

            let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader_module)
                .name(b"main\0");

            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();

            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);

            let viewport = vk::Viewport::builder()
                .x(0.0)
                .y(0.0)
                .width(swapchain_extent.width as f32)
                .height(swapchain_extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);

            let scissor = vk::Rect2D::builder()
                .offset(vk::Offset2D {x: 0, y: 0})
                .extent(swapchain_extent);

            let viewports = &[viewport];
            let scissors = &[scissor];
            let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
                .viewports(viewports)
                .scissors(scissors);

            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false);
                
            let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::_1);
            
            let attachment = vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::all())
                .blend_enable(false);

            let attachments = &[attachment];
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(attachments)
                .blend_constants([0.0, 0.0, 0.0, 0.0]);
            
            let pipeline_layout ={
                let layout_info = vk::PipelineLayoutCreateInfo::builder();
                unsafe {
                    logical_device.create_pipeline_layout(&layout_info, None)?
                }
            };

            let render_pass = RenderPassWeapper::new(swapchain_format, logical_device)?;

            let stages = &[vert_stage, frag_stage];
            let info = vk::GraphicsPipelineCreateInfo::builder()
                .stages(stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .color_blend_state(&color_blend_state)
                .layout(pipeline_layout)
                .render_pass(render_pass.render_pass)
                .subpass(0);

            let pipeline = unsafe {
                logical_device.create_graphics_pipelines(
                    vk::PipelineCache::null(), &[info], None)?.0[0]
            };

            // Clean up shader_modules at end
            unsafe {
                logical_device.destroy_shader_module(vert_shader_module, None);
                logical_device.destroy_shader_module(frag_shader_module, None);
            }

            Ok(Self { pipeline_layout, render_pass, pipeline})
        }

        fn create_shader_module(logical_device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
            use vulkanalia::bytecode::Bytecode;

            let bytecode = Bytecode::new(bytecode).context("Shader provided has incorrect bytecode")?;

            let info = vk::ShaderModuleCreateInfo::builder()
                .code_size(bytecode.code_size())
                .code(bytecode.code());

            let shader_module = unsafe {
                logical_device.create_shader_module(&info, None)?
            };
            
            Ok(shader_module)
        }
    
        fn destroy(&mut self, logical_device: &Device) {
            self.render_pass.destroy(logical_device);
            unsafe {
                logical_device.destroy_pipeline_layout(self.pipeline_layout, None);
                logical_device.destroy_pipeline(self.pipeline, None);
            }
        }
    }

    struct RenderPassWeapper {
        render_pass: vk::RenderPass,
    }

    impl RenderPassWeapper {
        fn new(swapchain_format: vk::Format, logical_device: &Device) -> Result<Self> {

            let render_pass = unsafe {
                // Attachments
                let color_attachment = vk::AttachmentDescription::builder()
                    .format(swapchain_format)
                    .samples(vk::SampleCountFlags::_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);


                // Subpasses
                let color_attachment_ref = vk::AttachmentReference::builder()
                    .attachment(0)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

                let color_attachments = &[color_attachment_ref];
                let subpass = vk::SubpassDescription::builder()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .color_attachments(color_attachments);

                // Dependencies
                let dependency = vk::SubpassDependency::builder()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

                let attachments = &[color_attachment];
                let subpasses = &[subpass];
                let dependencies = &[dependency];
                let info = vk::RenderPassCreateInfo::builder()
                    .attachments(attachments)
                    .subpasses(subpasses)
                    .dependencies(dependencies);


                logical_device.create_render_pass(&info, None)?  
            };

            Ok(Self { render_pass })
        }
 
        fn destroy(&mut self, logical_device: &Device) {
            unsafe {
                logical_device.destroy_render_pass(self.render_pass, None);
            }
        }
    }

    struct CommandWrapper {
        command_pool: vk::CommandPool,
        command_buffers: Vec<vk::CommandBuffer>,
        // Semaphores to control syncronization
        image_available_semaphore: vk::Semaphore,
        render_finished_semaphore: vk::Semaphore,
    }

    impl CommandWrapper {
        fn new(instance: &InstanceWrapper, physical_device: vk::PhysicalDevice, logical_device: &Device, framebuffers: &Vec<vk::Framebuffer>, swapchain: &SwapchainWrapper, pipeline: &PipelineWrapper) -> Result<Self> {
            let command_pool = CommandWrapper::create_command_pool(instance, physical_device, logical_device)?;
            let command_buffers  =CommandWrapper::create_command_buffers(logical_device, command_pool, framebuffers, swapchain, pipeline)?;

            let (image_available_semaphore, render_finished_semaphore) = unsafe {
                let semaphore_info = vk::SemaphoreCreateInfo::builder();

                let image_available_semaphore = logical_device.create_semaphore(&semaphore_info, None)?;
                let render_finished_semaphore = logical_device.create_semaphore(&semaphore_info, None)?;

                (image_available_semaphore, render_finished_semaphore)
            };

            Ok(Self { command_pool, command_buffers, image_available_semaphore, render_finished_semaphore })
        }

        fn create_command_pool(instance: &InstanceWrapper, physical_device: vk::PhysicalDevice, logical_device: &Device) -> Result<vk::CommandPool> {
            let indices = QueueFamilyIndices::new(instance, physical_device)?;

            let info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::empty())
                .queue_family_index(indices.graphics);

            let command_pool = unsafe {
                logical_device.create_command_pool(&info, None)?
            };

            Ok(command_pool)
        }
    
        fn create_command_buffers(logical_device: &Device, command_pool: vk::CommandPool, framebuffers: &Vec<vk::Framebuffer>, swapchain: &SwapchainWrapper, pipeline: &PipelineWrapper) -> Result<Vec<vk::CommandBuffer>> {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(framebuffers.len() as u32);

            let command_buffers = unsafe {
                logical_device.allocate_command_buffers(&allocate_info) ?
            };

            for (i, command_buffer) in command_buffers.iter().enumerate() {
                let inheritance = vk::CommandBufferInheritanceInfo::builder();

                let info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::empty())
                    .inheritance_info(&inheritance);

                unsafe {
                    logical_device.begin_command_buffer(*command_buffer, &info)?
                };

                let render_area = vk::Rect2D::builder()
                    .offset(vk::Offset2D::default())
                    .extent(swapchain.swapchain_extent);

                let color_clear_value = vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                };

                let clear_values = &[color_clear_value];
                let info = vk::RenderPassBeginInfo::builder()
                    .render_pass(pipeline.render_pass.render_pass)
                    .framebuffer(framebuffers[i])
                    .render_area(render_area)
                    .clear_values(clear_values);

                unsafe {
                    logical_device.cmd_begin_render_pass(*command_buffer, &info, vk::SubpassContents::INLINE);
                    logical_device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
                    logical_device.cmd_draw(*command_buffer, 3, 1, 0, 0);
                    logical_device.cmd_end_render_pass(*command_buffer);
                    logical_device.end_command_buffer(*command_buffer)?;
                }
                
            }

            Ok(command_buffers)
        }

        fn destroy(&mut self, logical_device: &Device) {
            
            unsafe {
                // Semaphores must match ones created

                logical_device.destroy_semaphore(self.render_finished_semaphore, None);
                logical_device.destroy_semaphore(self.image_available_semaphore, None);

                logical_device.destroy_command_pool(self.command_pool, None);
            }
        }
    }
    
}