use anyhow::Result;
use winit::window::Window;
//use control::control::{InstructionBuffer,Instruction, Layer};
use layers::Layer;


// Warning: minimized not handles properly, may need resructuring
pub struct GraphicsHandler {
    vulkan_handler: vulkan::VulkanHandler,
    minimized: bool,
}

impl GraphicsHandler {
    pub fn new(window: &Window) -> Result<Self> {
        let vulkan_handler = vulkan::VulkanHandler::new(window)?;
        let minimized = false;
        Ok(Self {vulkan_handler, minimized})
    }

    pub fn render(&mut self, window: &Window) -> Result<()> {
        if !self.minimized { // Does not render while minimized
            self.vulkan_handler.render(window)?;
        } 
        
        Ok(())
    }

    pub fn set_resized(&mut self) {
        self.vulkan_handler.set_resized();
    }

    pub fn set_minisized(&mut self, status: bool) {
        self.minimized = status;
    }

    pub fn inc(&mut self) {
        if self.vulkan_handler.model_num < 4 {
            self.vulkan_handler.model_num += 1;
        }
    }
        
    pub fn dec(&mut self) {
        if self.vulkan_handler.model_num > 1 {
            self.vulkan_handler.model_num -= 1;
        }
    }
}

impl Layer for GraphicsHandler {
    
}


/// For holding low-level (unsafe) Vulkan
mod vulkan {
    // General imports
    use winit::window::Window;
    use anyhow::{anyhow, Context, Result};
    use thiserror::Error;
    use log::*;
    use std::sync::Arc;

    use vulkanalia::loader::{LibloadingLoader, LIBRARY};
    use vulkanalia::window as vk_window;
    use vulkanalia::prelude::v1_0::*;
    // use vulkanalia::vk::{BufferCreateInfoBuilder, ExtDebugUtilsExtension, Image, KhrSurfaceExtension, KhrSwapchainExtension};
    use vulkanalia::vk::{ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension};

    use cgmath::{point3, Deg};
    type Mat4 = cgmath::Matrix4<f32>;

    use std::collections::{HashSet, HashMap};
    use std::hash::{Hash, Hasher};
    use std::io::BufReader;
    use std::ffi::CStr;
    use std::os::raw::c_void;
    use std::ops::{Deref, DerefMut};
    use std::time::Instant;
    use std::fs::File;

    const MAX_FRAMES_IN_FLIGHT: usize = 2;


    
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
        recreate: Option<RecreateWrapper>, // Handles objects that may be recreated
        model: Model,
        frames: Vec<Frame>,
        descriptor_set_layout: DescriptorSetLayoutWrapper,
        command_pool: CommandPoolWrapper,
        device: Arc<DeviceWrapper>, // Destroyed after all other objects
        instance: InstanceWrapper, // Destroyed after device
        
        current_frame: usize,
        resized: bool,
        start: Instant,
        pub model_num: usize,
    }

    impl VulkanHandler {
        pub fn new(window: &Window) -> Result<Self> {
            let instance = InstanceWrapper::new(window)?;
            let device = Arc::new(DeviceWrapper::new(&instance)?);
            let command_pool = CommandPoolWrapper::new(&instance, device.clone())?;
            let descriptor_set_layout = DescriptorSetLayoutWrapper::new(device.clone())?;
            // let texture = TextureImageWrapper::new(&instance, device.clone(), &command_pool)?;

            let frames = (0..MAX_FRAMES_IN_FLIGHT)
                .map(|_| Frame::new(&instance, device.clone()))
                .collect::<Result<Vec<Frame>>>()?;

            let model = Model::new(&instance, device.clone(), &command_pool)?;
            let recreate = Some(RecreateWrapper::new(device.clone(), &instance, window, &command_pool, &descriptor_set_layout, &model)?);

            Ok(Self {instance, device, frames, resized: false, recreate, command_pool, descriptor_set_layout, start: Instant::now(), model, current_frame: 0, model_num: 2})
        }

        pub fn render(&mut self, window: &Window) -> Result<()> {


            let recreate = self.recreate.as_mut().ok_or(anyhow!("Recreate missing"))?;
 
            // Wait for work on previous frame in slot to be completed

            
            unsafe {
                self.device.logical_device.wait_for_fences(
                    &[self.frames[self.current_frame].fence], 
                    true, 
                    u64::MAX,
                )?
            };

            // Reset frame resources

            self.frames[self.current_frame].reset()?;
            

            

            // Aquire image from swapchain

            let result = unsafe {
                self.device.acquire_next_image_khr(
                    recreate.swapchain.swapchain, 
                    u64::MAX,
                    self.frames[self.current_frame].image_available_semaphore,
                    vk::Fence::null()
                )
            };

            // Check if swapchain needs to be recreated (e.g. window resize)

            let image_index = match result {
                Ok((image_index, _)) => image_index as usize,
                Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return  self.recreate_swapchain(window),
                Err(e) => return  Err(anyhow!(e)),
            };
            // Wait for previous image in same swapchain index to be finished

            let image_in_flight = recreate.swapchain.swapchain_fences[self.current_frame];


            if !image_in_flight.is_null() {
                unsafe {
                    self.device.wait_for_fences(
                        &[image_in_flight],
                        true, 
                        u64::MAX,
                    )?;
                }
            }

            

            // Associate swapchain with current frame

            recreate.swapchain.swapchain_fences[image_index] = self.frames[self.current_frame].fence;

            // Update push constants and unifornm buffer + render

            // recreate.framebuffers[image_index].update(&recreate.swapchain, &recreate.pipeline, &recreate.descriptor_sets ,&self.model, image_index, self.start, &self.frames[self.current_frame])?;

            self.frames[self.current_frame].update(recreate, image_index, self.start, &self.model, self.model_num)?;

            VulkanHandler::update_uniform_buffer( self.device.clone(), &recreate.uniform_buffers.buffers[image_index], &recreate.swapchain)?;

            // Execute command buffer with image as attachment in the framebuffer

            
            let signal_semaphores = &[self.frames[self.current_frame].render_finished_semaphore];

            {
                let wait_semaphores = &[self.frames[self.current_frame].image_available_semaphore];
                let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                let command_buffers = &[self.frames[self.current_frame].command_buffers[0]];
                let submit_info = vk::SubmitInfo::builder()
                    .wait_semaphores(wait_semaphores)
                    .wait_dst_stage_mask(wait_stages)
                    .command_buffers(command_buffers)
                    .signal_semaphores(signal_semaphores);

                unsafe {
                    self.device.reset_fences(&[self.frames[self.current_frame].fence])?;
                };

                unsafe {
                    self.device.logical_device.queue_submit(
                        self.device.graphics_queue, 
                        &[submit_info], 
                        self.frames[self.current_frame].fence)?  
                };
            }

            let result = {
                let swapchains = &[recreate.swapchain.swapchain];
                let image_indices = &[image_index as u32];
                let present_info = vk::PresentInfoKHR::builder()
                    .wait_semaphores(signal_semaphores)
                    .swapchains(swapchains)
                    .image_indices(image_indices);

                unsafe {
                    self.device.queue_present_khr(self.device.present_queue, &present_info)
                }
            };

            // Handled after queue_present_khr to ensure semaphores in consitent state
            let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
                || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

            if self.resized || changed {
                self.resized = false;
                self.recreate_swapchain(window)?;
            } else if let Err(e) = result {
                return Err(anyhow!(e));
            }



            // Advance to next frame
            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

            Ok(())
        }

        pub fn set_resized(&mut self) {
            self.resized = true;
        }

        fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
            
            debug!("Recreate swapchain");

            unsafe {
                self.device.device_wait_idle()?
            };

            self.recreate = None;
            self.recreate = Some(RecreateWrapper::new(self.device.clone(), &self.instance, window, &self.command_pool, &self.descriptor_set_layout, &self.model)?);
            
            

            Ok(())
        }
    
        fn update_uniform_buffer(device: Arc<DeviceWrapper>, buffer: &BufferWrapper, swapchain: &SwapchainWrapper) -> Result<()> {

            let view = Mat4::look_at_rh(
                point3(6.0, 0.0, 2.0), 
                point3(0.0, 0.0, 0.0), 
                vec3(0.0, 0.0, 1.0),
            );

            let swapchain_extent = swapchain.swapchain_extent;

            let correction = Mat4::new(
                1.0 , 0.0, 0.0, 0.0, 
                0.0, -1.0, 0.0, 0.0, 
                0.0, 0.0, 1.0 / 2.0, 0.0, 
                0.0, 0.0, 1.0 / 2.0, 1.0,
            );

            // Transfer from OpenGL depth range [-1.0, 1.0] to Vulkan [0.0, 1.0]
            let proj = correction *
                cgmath::perspective(
                    Deg(45.0), 
                    swapchain_extent.width as f32 / swapchain_extent.height as f32, 
                    0.1, 
                    10.0,
                );

            let ubo = UniformBufferObject { view, proj };

            let memory = unsafe {
                device.map_memory(
                    buffer.buffer_memory, 
                    0, 
                    size_of::<UniformBufferObject>() as u64, 
                    vk::MemoryMapFlags::empty()
                )?
            };

            unsafe {
                memcpy(&ubo, memory.cast(), 1);

                device.unmap_memory(buffer.buffer_memory);
            }

            Ok(())
        }
    }

    impl Drop for VulkanHandler {
        fn drop(&mut self) {
            unsafe {
                self.device.logical_device.device_wait_idle().unwrap();
            }
        }
    }

    struct RecreateWrapper {
        depth_image: DepthImageWrapper,
        color_image: ColorImageWrapper,
        framebuffers: Vec<FramebufferWrapper>,
        uniform_buffers: UniformBuffers,
        descriptor_pool: DescriptorPoolWrapper,
        descriptor_sets: DescriptorSetsWrapper,
        pipeline: PipelineWrapper,
        swapchain: SwapchainWrapper,
        device: Arc<DeviceWrapper>,
    }

    impl RecreateWrapper {
        fn new(
            device: Arc<DeviceWrapper>, 
            instance: &InstanceWrapper, 
            window: &Window, 
            // vertex_buffer: &VertexBuffer,
            // index_buffer: &IndexBuffer,
            command_pool: &CommandPoolWrapper,
            descriptor_set_layout: &DescriptorSetLayoutWrapper,
            // texture: &TextureImageWrapper,
            model: &Model,
        ) -> Result<Self> {
            let swapchain = SwapchainWrapper::new(instance, device.clone(), window)?;
            let color_image = ColorImageWrapper::new(instance, device.clone(), &swapchain)?;
            let depth_image = DepthImageWrapper::new(instance, device.clone(), &swapchain)?;
            let uniform_buffers = UniformBuffers::new(instance, device.clone(), &swapchain)?;
            let pipeline = PipelineWrapper::new(device.clone(), swapchain.swapchain_extent, swapchain.swapchain_format, descriptor_set_layout, instance)?;
            let descriptor_pool = DescriptorPoolWrapper::new(device.clone(), &swapchain)?;
            let descriptor_sets = DescriptorSetsWrapper::new(device.clone(), descriptor_set_layout, &descriptor_pool, &swapchain, &uniform_buffers, model)?;
            let framebuffers = (0..swapchain.swapchain_images.len())
                .map(|i| FramebufferWrapper::new(device.clone(), &swapchain, &pipeline.render_pass, &depth_image, &color_image, i))
                .collect::<Result<Vec<_>>>()?;
            // command.images_in_flight.resize(swapchain.swapchain_images.len(), vk::Fence::null());

            Ok(Self {framebuffers, pipeline, swapchain, device, uniform_buffers, descriptor_pool, depth_image, color_image, descriptor_sets })
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

    impl Deref for InstanceWrapper {
        type Target = Instance;

        fn deref(&self) -> &Self::Target {
            &self.instance
        }
    }

    impl DerefMut for InstanceWrapper {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.instance
        }
    }

    #[derive(Debug, Error)]
    #[error("Missing {0}")]
    struct SuitabilityError(&'static str);
    struct DeviceWrapper {
        graphics_queue: vk::Queue,
        present_queue: vk::Queue,
        physical_device: vk::PhysicalDevice,
        logical_device: Device,
        msaa_samples: vk::SampleCountFlags,
    }

    impl DeviceWrapper {
        fn new(instance: &InstanceWrapper) -> Result<Self> {
            unsafe {
                for physical_device in instance.instance.enumerate_physical_devices()? {
                    let properties = instance.instance.get_physical_device_properties(physical_device);

                    if let Err(error) = DeviceWrapper::check_physical_device(instance, physical_device) {
                        warn!("Skipping physical device (`{}`): {}", properties.device_name, error);
                    } else {
                        info!("Selected physical device (`{}`)", properties.device_name);
                        let (logical_device, graphics_queue, present_queue) = DeviceWrapper::create_logical_device(instance, physical_device)?;
                        
                        let msaa_samples = DeviceWrapper::get_max_msaa_samples(instance, physical_device);

                        return Ok(Self { physical_device, logical_device, graphics_queue, present_queue, msaa_samples});
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
                if features.sampler_anisotropy != vk::TRUE {
                    return Err(anyhow!("No sampler anistropy support."));
                }
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

                let features = vk::PhysicalDeviceFeatures::builder()
                    .sampler_anisotropy(true);

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

        fn get_memory_type_index(&self, instance: &InstanceWrapper, properties: vk::MemoryPropertyFlags, requirements: vk::MemoryRequirements) -> Result<u32> {
            
            // Two arrays: memory_types and memory_heaps
            let memory = unsafe {
                instance.instance.get_physical_device_memory_properties(self.physical_device)
            };

            // Get a memory space which matches properties and requirements provided
            (0..memory.memory_type_count)
                .find(|i| {
                    let suitable = requirements.memory_type_bits & (1 << i) != 0;
                    let memory_type = memory.memory_types[*i as usize];
                    suitable && memory_type.property_flags.contains(properties)
                })
                .ok_or_else(|| anyhow!("Failed to find suitable memory type"))
        }

        
        fn get_max_msaa_samples(
            instance: &Instance,
            physical_device: vk::PhysicalDevice,
        ) -> vk::SampleCountFlags {
            let properties = unsafe {
                instance.get_physical_device_properties(physical_device)
            };
            let counts = properties.limits.framebuffer_color_sample_counts 
                & properties.limits.framebuffer_depth_sample_counts;
            [
                vk::SampleCountFlags::_64,
                vk::SampleCountFlags::_32,
                vk::SampleCountFlags::_16,
                vk::SampleCountFlags::_8,
                vk::SampleCountFlags::_4,
                vk::SampleCountFlags::_2,
            ]
            .iter()
            .cloned()
            .find(|c| counts.contains(*c))
            .unwrap_or(vk::SampleCountFlags::_1)
        }   
    }

    impl Drop for DeviceWrapper {
        fn drop(&mut self) {
            unsafe {
                
                
                self.logical_device.destroy_device(None);
            }
        }
    }

    impl Deref for DeviceWrapper {
        type Target = Device;

        fn deref(&self) -> &Self::Target {
            &self.logical_device
        }
    }

    impl DerefMut for DeviceWrapper {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.logical_device
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
        swapchain_image_views: Vec<ImageViewWrapper>,
        swapchain_fences: Vec<vk::Fence>,
        device: Arc<DeviceWrapper>,
    }

    impl SwapchainWrapper {
        fn new(instance: &InstanceWrapper, device: Arc<DeviceWrapper>, window: &Window) -> Result<Self> {

            info!("Swapchain recreated");

            let physical_device = device.physical_device;

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
                        device.create_swapchain_khr(&info, None)?
                }
            }; 

            let swapchain_images = unsafe {
                device.get_swapchain_images_khr(swapchain)?
            };

            let swapchain_image_views = SwapchainWrapper::create_swapchain_image_views(surface_format.format, &swapchain_images, &device)?;

            let swapchain_fences = (0..swapchain_images.len())
                .map(|_| vk::Fence::null())
                .collect();

            Ok(Self { swapchain_format: surface_format.format, swapchain_extent, swapchain, swapchain_images, swapchain_image_views, swapchain_fences, device })
        }

        fn create_swapchain_image_views(swapchain_format: vk::Format, swapchain_images: &Vec<vk::Image>, device: &Arc<DeviceWrapper>) -> Result<Vec<ImageViewWrapper>> {
            Ok(swapchain_images
                    .iter()
                    .map(|i| {
                        ImageViewWrapper::new(
                            device.clone(), 
                            *i, 
                            swapchain_format,
                            vk::ImageAspectFlags::COLOR,
                            1,
                        )
                        // SwapchainWrapper::create_image_view(swapchain_format, logical_device, *i, vk::ImageAspectFlags::COLOR)
                    })
                    .collect::<Result<Vec<_>,_>>()?)
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
    }

    impl Drop for SwapchainWrapper {
        fn drop(&mut self) {
            unsafe {
                /*
                self.swapchain_fences
                    .drain(..)
                    .for_each(|f| self.device.destroy_fence(f, None));
                */

                self.device.destroy_swapchain_khr(self.swapchain, None);
            }
        }
    }

    // Holds all resources rquired per frame
    struct Frame {
        command_pool: vk::CommandPool,
        command_buffers: Vec<vk::CommandBuffer>,
        secondary_command_buffers: Vec<vk::CommandBuffer>,

        // Semaphores to control GPU syncronization
        image_available_semaphore: vk::Semaphore,
        render_finished_semaphore: vk::Semaphore,

        // Fences to control CPU syncronization
        fence: vk::Fence,
        
        device: Arc<DeviceWrapper>,
    }
    
    impl Frame {
        fn new(
            instance: &InstanceWrapper,
            device: Arc<DeviceWrapper>,
        ) -> Result<Self>
        {
            let command_pool = {
                let indices = QueueFamilyIndices::new(instance, device.physical_device)?;

                let info = vk::CommandPoolCreateInfo::builder()
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .queue_family_index(indices.graphics);

                unsafe {
                    device.create_command_pool(&info, None)?
                }
            };

            let command_buffers = {
                let allocate_info = vk::CommandBufferAllocateInfo::builder()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);

                unsafe {
                    device.allocate_command_buffers(&allocate_info)?
                }
            };

            let (image_available_semaphore, render_finished_semaphore, fence) = unsafe {
                let semaphore_info = vk::SemaphoreCreateInfo::builder();
                let fence_info = vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::SIGNALED); // Want to have fences appear as if just complete frame at start rather than causing wait for non-existant frame

                let image_available_semaphore = device.create_semaphore(&semaphore_info, None)?;

                let render_finished_semaphore = device.create_semaphore(&semaphore_info, None)?;
            
                let in_flight_fence = device.create_fence(&fence_info, None)?;
                
                (image_available_semaphore, render_finished_semaphore, in_flight_fence)
            };

            Ok(Self { command_pool, command_buffers, secondary_command_buffers: vec![], image_available_semaphore, render_finished_semaphore, fence, device })
        }

        fn reset(
            &mut self
        ) -> Result<()> {
            unsafe {
                self.device.reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::RELEASE_RESOURCES)?;
            }

            Ok(())
        }

        fn update(
            &mut self,
            recreate: &RecreateWrapper,
            image_index: usize,
            start: Instant,
            object: &Model,
            model_num: usize,
        ) -> Result<()> {

            let command_buffer = self.command_buffers[0];


            let info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                self.device.begin_command_buffer(command_buffer, &info)?
            };

            let render_area = vk::Rect2D::builder()
                .offset(vk::Offset2D::default())
                .extent(recreate.swapchain.swapchain_extent);

            let color_clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            };

            let depth_clear_value = vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            };

            let clear_values = &[color_clear_value, depth_clear_value];
            let info = vk::RenderPassBeginInfo::builder()
                .render_pass(recreate.pipeline.render_pass)
                .framebuffer(recreate.framebuffers[image_index].framebuffer)
                .render_area(render_area)
                .clear_values(clear_values);

            unsafe {
                self.device.cmd_begin_render_pass(
                    command_buffer, 
                    &info, 
                    vk::SubpassContents::SECONDARY_COMMAND_BUFFERS
                );
                
                let secondary_command_buffers = (0..model_num)
                    .map(|i| 
                        self.update_secondary_command_buffer(
                            image_index, 
                            i,
                            recreate, 
                            object, 
                            start
                        )
                    )
                    .collect::<Result<Vec<_>,_>>()?;

                self.device.cmd_execute_commands(command_buffer, &secondary_command_buffers[..]);
                
                self.device.cmd_end_render_pass(command_buffer);
                self.device.end_command_buffer(command_buffer)?;
            }


            Ok(())
        }
        
        fn update_secondary_command_buffer(
            &mut self,
            image_index: usize,
            model_index: usize,
            // framebuffer: &FramebufferWrapper,
            // render_pass: vk::RenderPass,
            recreate: &RecreateWrapper,
            object: &Model,
            start: Instant,
        ) -> Result<vk::CommandBuffer> {
            // self.secondary_command_buffers.resize_with(image_index + 1, );
            let command_buffers = &mut self.secondary_command_buffers;
            while model_index >= command_buffers.len() {
                let allocate_info = vk::CommandBufferAllocateInfo::builder()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::SECONDARY)
                    .command_buffer_count(1);

                let command_buffer = unsafe {
                    self.device.allocate_command_buffers(&allocate_info)?
                };
                command_buffers.extend(command_buffer);
            }

            let command_buffer = command_buffers[model_index];

            // Model specific data

            let time = start.elapsed().as_secs_f32();

            let y = (((model_index % 2) as f32) * 2.5) - 1.25;
            let z = (((model_index / 2) as f32) * -2.0) + 1.0;

            let model = Mat4::from_translation(vec3(0.0, y, z)) *  Mat4::from_axis_angle(
                vec3(0.0, 0.0, 1.0), 
                Deg(90.0) * time,
            );

            let model_bytes = unsafe {
                &*std::slice::from_raw_parts(
                    &model as *const Mat4 as *const u8,
                    size_of::<Mat4>(),
                )
            };

            let opacity = (model_index + 1) as f32 * 0.25;
            let opacity_bytes = &opacity.to_ne_bytes()[..];



            let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
                .render_pass(recreate.pipeline.render_pass)
                .subpass(0)
                .framebuffer(recreate.framebuffers[image_index].framebuffer);

            let info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
                .inheritance_info(&inheritance_info);

            unsafe {
                self.device.begin_command_buffer(command_buffer, &info)?;

                self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, recreate.pipeline.pipeline);
                self.device.cmd_bind_vertex_buffers(command_buffer, 0, &[object.vertex_buffer.buffer.buffer], &[0]);
                self.device.cmd_bind_index_buffer(command_buffer, object.index_buffer.buffer.buffer, 0, vk::IndexType::UINT32);
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS, 
                    recreate.pipeline.pipeline_layout, 
                    0, 
                    &[recreate.descriptor_sets.descriptor_sets[image_index]], 
                    &[],
                );
                self.device.cmd_push_constants(
                    command_buffer, 
                    recreate.pipeline.pipeline_layout, 
                    vk::ShaderStageFlags::VERTEX, 
                    0, 
                    model_bytes,
                );
                self.device.cmd_push_constants(
                    command_buffer, 
                    recreate.pipeline.pipeline_layout, 
                    vk::ShaderStageFlags::FRAGMENT, 
                    64, 
                    opacity_bytes,
                );

                self.device.cmd_draw_indexed(
                    command_buffer, 
                    object.indices.len() as u32, 
                    1, 
                    0, 
                    0, 
                    0,
                );


                self.device.end_command_buffer(command_buffer)?;
            }

            Ok(command_buffer)

        }
        
    }


    impl Drop for Frame {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_semaphore(self.image_available_semaphore, None);
                self.device.destroy_semaphore(self.render_finished_semaphore, None);

                self.device.destroy_fence(self.fence, None);

                self.device.free_command_buffers(self.command_pool, &self.command_buffers);
                self.device.destroy_command_pool(self.command_pool, None);
            }
        }
    }

    struct DescriptorSetLayoutWrapper {
        descriptor_set_layout: vk::DescriptorSetLayout,
        device: Arc<DeviceWrapper>,
    }

    impl DescriptorSetLayoutWrapper {
        fn new(
            device: Arc<DeviceWrapper>,
        ) -> Result<Self> {
            let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX);

            let sampler_bindling = vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT);

            let bindings = &[ubo_binding, sampler_bindling];
            let info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(bindings);

            let descriptor_set_layout = unsafe {
                device.create_descriptor_set_layout(&info, None)?
            };

            Ok(Self { descriptor_set_layout, device })

        }
    }

    impl Drop for DescriptorSetLayoutWrapper {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            }
        }
    }

    impl Deref for DescriptorSetLayoutWrapper {
        type Target = vk::DescriptorSetLayout;

        fn deref(&self) -> &Self::Target {
            &self.descriptor_set_layout
        }
    }

    impl DerefMut for DescriptorSetLayoutWrapper {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.descriptor_set_layout
        }
    }

    struct DescriptorPoolWrapper {
        descriptor_pool: vk::DescriptorPool,
        device: Arc<DeviceWrapper>,
    }

    impl DescriptorPoolWrapper {
        fn new(device: Arc<DeviceWrapper>, swapchain: &SwapchainWrapper) -> Result<Self> {
            let ubo_size = vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(swapchain.swapchain_images.len() as u32);

            let sampler_size = vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(swapchain.swapchain_images.len() as u32);

            let pool_sizes = &[ubo_size, sampler_size];
            let info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(pool_sizes)
                .max_sets(swapchain.swapchain_images.len() as u32);

            let descriptor_pool = unsafe {
                device.create_descriptor_pool(&info, None)? 
            };

            Ok(Self { descriptor_pool, device })
        }
    }

    impl Drop for DescriptorPoolWrapper {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            }
        }
    }

    struct DescriptorSetsWrapper {
        descriptor_sets: Vec<vk::DescriptorSet>,
    }

    impl DescriptorSetsWrapper {
        fn new(
            device: Arc<DeviceWrapper>, 
            descriptor_set_layout: &DescriptorSetLayoutWrapper, 
            descriptor_pool: &DescriptorPoolWrapper,
            swapchain: &SwapchainWrapper,
            uniform_buffers: &UniformBuffers,
            // texture_image: &TextureImageWrapper,
            model: &Model,

        ) -> Result<Self> {
            let layouts = vec![descriptor_set_layout.descriptor_set_layout; swapchain.swapchain_images.len()];
            let info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool.descriptor_pool)
                .set_layouts(&layouts);

            let descriptor_sets = unsafe {
                device.allocate_descriptor_sets(&info)?
            };

            for i in 0..swapchain.swapchain_images.len() {
                let info = vk::DescriptorBufferInfo::builder()
                    .buffer(uniform_buffers.buffers[i].buffer)
                    .offset(0)
                    .range(size_of::<UniformBufferObject>() as u64);

                let buffer_info = &[info];
                let ubo_write = vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[i])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(buffer_info);

                let info = vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(model.image.image_view.image_view)
                    .sampler(model.sampler.sampler);

                let image_info = &[info];
                let sampler_write = vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[i])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(image_info);

                unsafe {
                    device.update_descriptor_sets(
                        &[ubo_write, sampler_write], 
                        &[] as &[vk::CopyDescriptorSet]);
                }
            }

            Ok(Self { descriptor_sets })                        
        }
    }

    struct PipelineWrapper {
        pipeline_layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        pipeline: vk::Pipeline,
        device: Arc<DeviceWrapper>
    }

    impl PipelineWrapper {
        fn new(
            device: Arc<DeviceWrapper>, 
            swapchain_extent: vk::Extent2D, 
            swapchain_format: vk::Format, 
            descriptor_set_layout: &DescriptorSetLayoutWrapper,
            instance: &InstanceWrapper,
         ) -> Result<Self> {
            let vert = include_bytes!("../shaders/shader.vert.spv");
            let frag = include_bytes!("../shaders/shader.frag.spv");

            let logical_device = &**device;

            let vert_shader_module = PipelineWrapper::create_shader_module(logical_device, &vert[..])?;
            let frag_shader_module = PipelineWrapper::create_shader_module(logical_device, &frag[..])?;

            let vert_stage = {vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader_module)
                .name(b"main\0")
            };

            let frag_stage = {vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader_module)
                .name(b"main\0")
            };

            let binding_descriptions = &[Vertex::binding_description()];
            let attribute_descriptions = Vertex::attribute_descriptions();

            let vertex_input_state = {vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(binding_descriptions)
                .vertex_attribute_descriptions(&attribute_descriptions)
            };


            let input_assembly_state = {vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false)
            };

            let viewport = {vk::Viewport::builder()
                .x(0.0)
                .y(0.0)
                .width(swapchain_extent.width as f32)
                .height(swapchain_extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0)
            };

            let scissor = {vk::Rect2D::builder()
                .offset(vk::Offset2D {x: 0, y: 0})
                .extent(swapchain_extent)
            };

            let viewports = &[viewport];
            let scissors = &[scissor];
            let viewport_state = {vk::PipelineViewportStateCreateInfo::builder()
                .viewports(viewports)
                .scissors(scissors)
            };

            let rasterization_state = {vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
            };
                
            let multisample_state = {vk::PipelineMultisampleStateCreateInfo::builder()
                .sample_shading_enable(false)
                .rasterization_samples(device.msaa_samples)
            };

            let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS)
                .depth_bounds_test_enable(false)
                .min_depth_bounds(0.0)
                .max_depth_bounds(1.0)
                .stencil_test_enable(false);
                // .front and .back required for use of stencil

            let attachment = vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::all())
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD);

            let attachments = &[attachment];
            let color_blend_state = {vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(attachments)
                .blend_constants([0.0, 0.0, 0.0, 0.0])
            };
            
            let pipeline_layout = {
                let vert_push_constant_range = vk::PushConstantRange::builder()
                    .stage_flags(vk::ShaderStageFlags::VERTEX)
                    .offset(0)
                    .size(64);

                let frag_push_constant_range = vk::PushConstantRange::builder()
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .offset(64)
                    .size(4);

                let set_layouts = &[**descriptor_set_layout];
                let push_constant_ranges = &[vert_push_constant_range, frag_push_constant_range];
                let layout_info = vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(set_layouts)
                    .push_constant_ranges(push_constant_ranges);

                unsafe {
                    logical_device.create_pipeline_layout(&layout_info, None)?
                }
            };

            // let render_pass = RenderPassWeapper::new(swapchain_format, logical_device)?;

            let render_pass = unsafe {
                // Attachments
                let color_attachment = vk::AttachmentDescription::builder()
                    .format(swapchain_format)
                    .samples(device.msaa_samples)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

                let depth_stencil_attachment = vk::AttachmentDescription::builder()
                    .format(DepthImageWrapper::get_depth_format(&device, instance)?)
                    .samples(device.msaa_samples)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

                // Resolve the initial multisampled attachment to habe sample count of 1
                let color_reolve_attachment = vk::AttachmentDescription::builder()
                    .format(swapchain_format)
                    .samples(vk::SampleCountFlags::_1)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
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


                let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
                    .attachment(1)
                    .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

                let color_resolve_attachment_ref = vk::AttachmentReference::builder()
                    .attachment(2)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

                let resolve_attachments = &[color_resolve_attachment_ref];

                let subpass = vk::SubpassDescription::builder()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .color_attachments(color_attachments)
                    .depth_stencil_attachment(&depth_stencil_attachment_ref)
                    .resolve_attachments(resolve_attachments);

                // Dependencies
                let dependency = vk::SubpassDependency::builder()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT 
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

                let attachments = &[
                    color_attachment, 
                    depth_stencil_attachment,
                    color_reolve_attachment,
                    ];
                let subpasses = &[subpass];
                let dependencies = &[dependency];
                let info = vk::RenderPassCreateInfo::builder()
                    .attachments(attachments)
                    .subpasses(subpasses)
                    .dependencies(dependencies);


                logical_device.create_render_pass(&info, None)?  
            };

            let stages = &[vert_stage, frag_stage];
            let info = {vk::GraphicsPipelineCreateInfo::builder()
                .stages(stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .depth_stencil_state(&depth_stencil_state)
                .color_blend_state(&color_blend_state)
                .layout(pipeline_layout)
                .render_pass(render_pass)
                .subpass(0)
            };

            let pipeline = unsafe {
                logical_device.create_graphics_pipelines(
                    vk::PipelineCache::null(), &[info], None)?.0[0]
            };



            // Clean up shader_modules at end
            unsafe {
                logical_device.destroy_shader_module(vert_shader_module, None);
                logical_device.destroy_shader_module(frag_shader_module, None);
            }

            Ok(Self { pipeline_layout, render_pass, pipeline, device})
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
    }

    impl Drop for PipelineWrapper {
        fn drop(&mut self) {
            // self.render_pass.destroy(logical_device);
            unsafe {
                self.device.destroy_render_pass(self.render_pass, None);
                self.device.destroy_pipeline_layout(self.pipeline_layout, None);
                self.device.destroy_pipeline(self.pipeline, None);
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

    // 
    struct CommandPoolWrapper {
        command_pool: vk::CommandPool,
        device: Arc<DeviceWrapper>,
    }

    impl CommandPoolWrapper {
        fn new(instance: &InstanceWrapper, device: Arc<DeviceWrapper>) -> Result<Self> {
            let indices = QueueFamilyIndices::new(instance, device.physical_device)?;

            let info = vk::CommandPoolCreateInfo::builder()
                // .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                .queue_family_index(indices.graphics);

            let command_pool = unsafe {
                device.create_command_pool(&info, None)?
            };

            Ok(Self { command_pool, device })
        }

    }

    impl Drop for CommandPoolWrapper {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_command_pool(self.command_pool, None);
            }
        }
    }
    
    impl Deref for CommandPoolWrapper{
        type Target = vk::CommandPool;

        fn deref(&self) -> &Self::Target {
            &self.command_pool
        }
    }

    impl DerefMut for CommandPoolWrapper {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.command_pool
        }
    }

    struct FramebufferWrapper {
        framebuffer: vk::Framebuffer,
        device: Arc<DeviceWrapper>,
    }

    impl FramebufferWrapper {
        fn new(
            device: Arc<DeviceWrapper>,
            swapchain: &SwapchainWrapper,
            render_pass: &vk::RenderPass, 
            depth_image: &DepthImageWrapper, 
            color_image: &ColorImageWrapper,
            index: usize,
        ) -> Result<Self> {
            
            let framebuffer = {
                let attachments = &[color_image.color_image_view.image_view, depth_image.depth_image_view.image_view, swapchain.swapchain_image_views[index].image_view];
                let create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(*render_pass)
                    .attachments(attachments)
                    .width(swapchain.swapchain_extent.width)
                    .height(swapchain.swapchain_extent.height)
                    .layers(1);

                unsafe {
                    device.create_framebuffer(&create_info, None)?
                }
            };
            

            Ok(Self { framebuffer, device })
        }

        fn update(
            &mut self,
            swapchain: &SwapchainWrapper, 
            pipeline: &PipelineWrapper,
            descriptor_sets: &DescriptorSetsWrapper,
            model: &Model,
            image_index: usize,
            start: Instant,
            frame: &Frame,
        ) -> Result<()> {

            

            
            
            Ok(())
        }
    }

    impl Drop for FramebufferWrapper {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_framebuffer(self.framebuffer, None);
            }
        }
    }

    struct OneTimeCommand {
        command_buffer: vk::CommandBuffer,
        device: Arc<DeviceWrapper>,
    }

    impl OneTimeCommand {
        /// Must be manually ended
        unsafe fn new(device: Arc<DeviceWrapper>, command_pool: &CommandPoolWrapper) -> Result<Self> {
            let info = vk::CommandBufferAllocateInfo::builder()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(command_pool.command_pool)
                .command_buffer_count(1);

            let command_buffer = unsafe {
                device.allocate_command_buffers(&info)?[0]
            };

            let info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            unsafe {
                device.begin_command_buffer(command_buffer, &info)?;
            }

            Ok(Self { command_buffer, device })
        }

        unsafe fn end(&mut self, command_pool: &CommandPoolWrapper) -> Result<()> {
            self.device.end_command_buffer(self.command_buffer)?;

            let command_buffers = &[self.command_buffer];
            let info = vk::SubmitInfo::builder()
                .command_buffers(command_buffers);

            self.device.queue_submit(self.device.graphics_queue, &[info], vk::Fence::null())?;
            self.device.queue_wait_idle(self.device.graphics_queue)?;

            self.device.free_command_buffers(command_pool.command_pool, command_buffers);

            Ok(())
        }
    }


    use std::mem::size_of;
    use cgmath::{vec2, vec3};

    type Vec2 = cgmath::Vector2<f32>;
    type Vec3 = cgmath::Vector3<f32>;

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    struct Vertex {
        pos: Vec3,
        color: Vec3,
        tex_coord: Vec2,
    }

    impl Vertex {
        const fn new(pos: Vec3, color: Vec3, tex_coord: Vec2) -> Self {
            Self {pos, color, tex_coord}
        }

        fn binding_description() -> vk::VertexInputBindingDescription {
            vk::VertexInputBindingDescription::builder()
                .binding(0)
                .stride(size_of::<Vertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)
                .build()
        }

        fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
            let pos = vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0)
                .build();

            let color = vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(size_of::<Vec3>() as u32)
                .build();

            let tex_coord = vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset((size_of::<Vec3>() + size_of::<Vec3>()) as u32)
                .build();

            [pos, color, tex_coord]
        }
    }

    impl PartialEq for Vertex {
        fn eq(&self, other: &Self) -> bool {
            self.pos == other.pos
                && self.color == other.color
                && self.tex_coord == other.tex_coord
        }
    }

    impl Eq for Vertex {}

    impl Hash for Vertex {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.pos[0].to_bits().hash(state);
            self.pos[1].to_bits().hash(state);
            self.pos[2].to_bits().hash(state);
            self.color[0].to_bits().hash(state);
            self.color[1].to_bits().hash(state);
            self.color[2].to_bits().hash(state);
            self.tex_coord[0].to_bits().hash(state);
            self.tex_coord[1].to_bits().hash(state);
        }
    }

    static VERTICES: [Vertex; 8] = [
        Vertex::new(
            vec3(-0.5, -0.5, 0.0), 
            vec3(1.0, 0.0, 0.0),
            vec2(1.0, 0.0),
        ),
        Vertex::new(
            vec3(0.5, -0.5, 0.0), 
            vec3(0.0, 1.0, 0.0),
            vec2(0.0, 0.0),
        ),
        Vertex::new(
            vec3(0.5, 0.5, 0.0), 
            vec3(0.0, 0.0, 1.0),
            vec2(0.0, 1.0),
        ),
        Vertex::new(
            vec3(-0.5, 0.5, 0.0), 
            vec3(1.0, 1.0, 1.0),
            vec2(1.0, 1.0),
        ),
        Vertex::new(
            vec3(-0.5, -0.5, -0.5), 
            vec3(1.0, 0.0, 0.0),
            vec2(1.0, 0.0),
        ),
        Vertex::new(
            vec3(0.5, -0.5, -0.5), 
            vec3(0.0, 1.0, 0.0),
            vec2(0.0, 0.0),
        ),
        Vertex::new(
            vec3(0.5, 0.5, -0.5), 
            vec3(0.0, 0.0, 1.0),
            vec2(0.0, 1.0),
        ),
        Vertex::new(
            vec3(-0.5, 0.5, -0.5), 
            vec3(1.0, 1.0, 1.0),
            vec2(1.0, 1.0),
        ),
    ];

    const INDICES: &[u32] = &[
        0,1,2,2,3,0,
        4,5,6,6,7,4,
    ];


    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    struct  UniformBufferObject {
        view: Mat4,
        proj: Mat4,
    }


    // Buffers

    use std::ptr::copy_nonoverlapping as memcpy;

    struct BufferWrapper {
        buffer_info: vk::BufferCreateInfoBuilder<'static>,
        buffer_memory: vk::DeviceMemory,
        buffer: vk::Buffer,
        device: Arc<DeviceWrapper>,
    }

    impl BufferWrapper {
        fn new(instance: &InstanceWrapper,
               device: Arc<DeviceWrapper>,
               size: vk::DeviceSize,
               properties: vk::MemoryPropertyFlags, 
               usage: vk::BufferUsageFlags,
            ) -> Result<Self> {
            
            let buffer_info = vk::BufferCreateInfo::builder()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            // Safety: memory concerns for sparse memory
            let buffer = unsafe {
                device.create_buffer(&buffer_info, None)?
            };

            let requirements = unsafe {
                device.get_buffer_memory_requirements(buffer)  
            };

            let memory_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(requirements.size)
                .memory_type_index(device.get_memory_type_index(instance, properties, requirements)?);

            let buffer_memory = unsafe {
                device.allocate_memory(&memory_info, None)?  
            };

            unsafe {
                device.bind_buffer_memory(buffer,buffer_memory, 0)?  
            };

            Ok(Self { buffer_memory, buffer, buffer_info, device })
        }

        // Copies given other buffer into this one
        fn copy_buffer_into(&mut self, src_buffer: BufferWrapper, command_pool: &CommandPoolWrapper) -> Result<()> {
            /*
            let info = vk::CommandBufferAllocateInfo::builder()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(**command_pool)
                .command_buffer_count(1);

            let command_buffer = unsafe {
                self.device.allocate_command_buffers(&info)?[0]
            };
            

            let info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            unsafe {
                self.device.begin_command_buffer(command_buffer, &info)?;

                let regions = vk::BufferCopy::builder().size(src_buffer.buffer_info.size);
                self.device.cmd_copy_buffer(command_buffer, src_buffer.buffer, self.buffer, &[regions]);

                self.device.end_command_buffer(command_buffer)?;
            };

            let command_buffers = &[command_buffer];
            let info = vk::SubmitInfo::builder()
                .command_buffers(command_buffers);

            unsafe {
                self.device.queue_submit(self.device.graphics_queue, &[info], vk::Fence::null())?;
                self.device.queue_wait_idle(self.device.graphics_queue)?;

                self.device.free_command_buffers(**command_pool, command_buffers);
            }
            */

            let mut command_buffer = unsafe {
                OneTimeCommand::new(self.device.clone(), command_pool)?
            };

            unsafe {
                let regions = vk::BufferCopy::builder().size(src_buffer.buffer_info.size);
                self.device.cmd_copy_buffer(command_buffer.command_buffer, src_buffer.buffer, self.buffer, &[regions]);
            }

            unsafe {
                command_buffer.end(command_pool)?;
            }

            //*/

            Ok(())
        }
    }

    impl Drop for BufferWrapper {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_buffer(self.buffer, None);
                self.device.free_memory(self.buffer_memory, None)
            } 
        }
    }

    struct VertexBuffer {
        buffer: BufferWrapper
    }

    impl VertexBuffer {
        fn new(
            instance: &InstanceWrapper,
            device: Arc<DeviceWrapper>,
            command_pool: &CommandPoolWrapper,
            vertices: &Vertices,
        ) -> Result<Self> {
            let size = (size_of::<Vertex>() * vertices.len()) as u64;

            let staging_buffer = BufferWrapper::new(
                instance, 
                device.clone(), 
                size, 
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?;

            let memory = unsafe {
                device.map_memory(
                    staging_buffer.buffer_memory, 
                    0, 
                    staging_buffer.buffer_info.size,
                    vk::MemoryMapFlags::empty()
                )
            }?;
            
            unsafe {
                memcpy(vertices.as_ptr(), memory.cast(), vertices.len());
                device.unmap_memory(staging_buffer.buffer_memory);
            };  

            let mut buffer = BufferWrapper::new(
                instance, 
                device.clone(),
                size,
                vk::MemoryPropertyFlags::DEVICE_LOCAL, 
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;

            buffer.copy_buffer_into(staging_buffer, command_pool)?;

            Ok(Self { buffer })
        }
    }

    struct IndexBuffer {
        buffer: BufferWrapper,
    }

    impl IndexBuffer {
        fn new(
            instance: &InstanceWrapper,
            device: Arc<DeviceWrapper>,
            command_pool: &CommandPoolWrapper,
            indices: &Indices,
        ) -> Result<Self> {
            let size = (size_of::<u32>() * indices.len()) as u64;

            let staging_buffer = BufferWrapper::new(
                instance, 
                device.clone(), 
                size, 
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE, 
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?;

            let memory = unsafe {
                device.map_memory(
                    staging_buffer.buffer_memory, 
                    0, 
                    size, 
                    vk::MemoryMapFlags::empty(),
                )
            }?;

            unsafe {
                memcpy(indices.as_ptr(), memory.cast(), indices.len());

                device.unmap_memory(staging_buffer.buffer_memory);
            }

            let mut buffer = BufferWrapper::new(
                instance, 
                device.clone(),
                size, 
                vk::MemoryPropertyFlags::DEVICE_LOCAL, 
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            )?;

            buffer.copy_buffer_into(staging_buffer, command_pool)?;

            Ok(Self { buffer })
        }
    }

    struct UniformBuffers {
        buffers: Vec<BufferWrapper>,
    }

    impl UniformBuffers {
        fn new(
            instance: &InstanceWrapper,
            device: Arc<DeviceWrapper>,
            swapchain: &SwapchainWrapper,
        ) -> Result<Self> {
            let mut buffers = Vec::new();

            for _ in 0..swapchain.swapchain_images.len() {
                buffers.push(BufferWrapper::new(
                    instance, 
                    device.clone(), 
                    size_of::<UniformBufferObject>() as u64, 
                    vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE, 
                    vk::BufferUsageFlags::UNIFORM_BUFFER
                )?);
            }

            Ok(Self { buffers })
        }
    }



    struct Model {
        image: ImageWrapper,
        sampler: SamplerWrapper,
        vertices: Vertices,
        indices: Indices,
        index_buffer: IndexBuffer,
        vertex_buffer: VertexBuffer,
        mip_levels: u32,
    }

    type Vertices = Vec<Vertex>;
    type Indices = Vec<u32>;

    impl Model {
        fn new(
            instance: &InstanceWrapper, 
            device: Arc<DeviceWrapper>, 
            command_pool: &CommandPoolWrapper,
        ) -> Result<Self>{

            let (vertices, indices) = Model::load_model()?;

            let vertex_buffer = VertexBuffer::new(&instance, device.clone(), &command_pool, &vertices)?;
            let index_buffer = IndexBuffer::new(&instance, device.clone(), &command_pool, &indices)?;

            let image = File::open("resources/viking_room.png").context("Texture file failure")?;

            let decoder = png::Decoder::new(image);
            let mut reader = decoder.read_info()?;

            let mut pixels = vec![0; reader.info().raw_bytes()];
            reader.next_frame(&mut pixels)?;

            let size = reader.info().raw_bytes() as u64;
            let (width, height) = reader.info().size();
            let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

            if width != 1024 || height != 1024 || reader.info().color_type != png::ColorType::Rgba {
                return Err(anyhow!("Invalid texture image"));
            }

            // Create (Staging)

            let staging_buffer = 
                BufferWrapper::new(
                    instance, 
                    device.clone(), 
                    size, 
                    vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE, 
                    vk::BufferUsageFlags::TRANSFER_SRC
                )?;

            // Copy (staging)

            let memory = unsafe {
                device.map_memory(
                    staging_buffer.buffer_memory, 
                    0, 
                    size, 
                    vk::MemoryMapFlags::empty()
                )?
            };

            unsafe {
                memcpy(pixels.as_ptr(), memory.cast(), pixels.len());

                device.unmap_memory(staging_buffer.buffer_memory);
            }

            // Create (Image)

            let mut image = ImageWrapper::new(
                instance, 
                device.clone(), 
                width, 
                height, 
                vk::Format::R8G8B8A8_SRGB, 
                vk::ImageTiling::OPTIMAL, 
                vk::ImageUsageFlags::SAMPLED 
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::ImageAspectFlags::COLOR,
                mip_levels,
                vk::SampleCountFlags::_1,
            )?;

            // Transition + Copy (Image)

            image.transition_layout(
                command_pool, 
                // vk::Format::R8G8B8A8_SRGB, 
                vk::ImageLayout::UNDEFINED, 
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                mip_levels,
            )?;

            image.copy_buffer_to_image(
                device.clone(), 
                command_pool, 
                staging_buffer, 
                width, 
                height,
            )?;
            
            Model::generate_mipmaps(
                instance, 
                &device, 
                command_pool, 
                &image,
                vk::Format::R8G8B8A8_SRGB, 
                width, 
                height, 
                mip_levels
            )?;

            
            /*
            image.transition_layout(
                command_pool, 
                vk::Format::R8G8B8A8_SRGB, 
                vk::ImageLayout::TRANSFER_DST_OPTIMAL, 
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?;
            */

            let sampler = SamplerWrapper::new(device.clone(), mip_levels)?;

            

            
            Ok(Self { image, sampler, vertex_buffer, index_buffer, vertices, indices, mip_levels })
        }

        fn load_model(

        ) -> Result<(Vertices, Indices)> {

            let mut reader = BufReader::new(File::open("resources/viking_room.obj")?);

            let (modles, _) = tobj::load_obj_buf(
                &mut reader, 
                &tobj::LoadOptions {triangulate: true, ..Default::default()}, 
                |_| Ok(Default::default()),
            )?;

            let mut vertices: Vertices = Vec::new();
            let mut indices: Indices = Vec::new();

            let mut unqiue_vertices = HashMap::new();

            for model in &modles {
                for index in &model.mesh.indices {

                    let pos_offset = (3 * index) as usize;
                    let tex_coord_offset = (2 * index) as usize;

                    let vertex = Vertex {
                        pos: vec3(
                            model.mesh.positions[pos_offset],
                            model.mesh.positions[pos_offset + 1],
                            model.mesh.positions[pos_offset + 2],
                        ),
                        color: vec3(1.0, 1.0, 1.0),
                        tex_coord: vec2(
                            model.mesh.texcoords[tex_coord_offset],
                            1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                        ),
                    };

                    if let Some(index) = unqiue_vertices.get(&vertex) {
                        indices.push(*index as u32);
                    } else {
                        let index = vertices.len();
                        unqiue_vertices.insert(vertex, index);
                        vertices.push(vertex);
                        indices.push(index as u32);
                    }
                }
            }

            Ok((vertices, indices))
        }
    
        fn generate_mipmaps(
            instance: &InstanceWrapper,
            device: &Arc<DeviceWrapper>,
            command_pool: &CommandPoolWrapper,
            image: &ImageWrapper,
            format: vk::Format,
            width: u32,
            height: u32,
            mip_levels: u32,
        ) -> Result<()> {

            if unsafe {
                !instance
                .get_physical_device_format_properties(device.physical_device, format)
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)}
            {
                return Err(anyhow!("Texture image format does not support linear blitting"));
            }

            let mut command_buffer = unsafe {
                OneTimeCommand::new(device.clone(), command_pool)?
            };

            let subresource = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .level_count(1);

            let mut barrier = vk::ImageMemoryBarrier::builder()
                .image(image.image)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(subresource);

            let mut mip_width = width;
            let mut mip_height = height;

            // Transition each mipmap level individually, from each step to another

            for i in 1..mip_levels {
                barrier.subresource_range.base_mip_level = i - 1;
                barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer.command_buffer, 
                        vk::PipelineStageFlags::TRANSFER, 
                        vk::PipelineStageFlags::TRANSFER, 
                        vk::DependencyFlags::empty(), 
                        &[] as &[vk::MemoryBarrier], 
                        &[] as &[vk::BufferMemoryBarrier], 
                        &[barrier],
                    );
                }

                let src_subresource = vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(i - 1)
                    .base_array_layer(0)
                    .layer_count(1);

                let dst_subresource = vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(i)
                    .base_array_layer(0)
                    .layer_count(1);

                let blit = vk::ImageBlit::builder()
                    .src_offsets([
                        vk::Offset3D {x: 0, y: 0, z: 0},
                        vk::Offset3D {
                            x: mip_width as i32,
                            y: mip_height as i32,
                            z: 1,
                        },
                    ])
                    .src_subresource(src_subresource)
                    .dst_offsets([
                        vk::Offset3D {x: 0, y: 0, z: 0},
                        vk::Offset3D {
                            x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                            y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                            z: 1,
                        },
                    ])
                    .dst_subresource(dst_subresource);

                    unsafe {
                        device.cmd_blit_image(
                            command_buffer.command_buffer, 
                            image.image, 
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL, 
                            image.image, 
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL, 
                            &[blit],
                            vk::Filter::LINEAR,
                        );
                    }

                    barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                    barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
                    barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

                    unsafe {
                        device.cmd_pipeline_barrier(
                            command_buffer.command_buffer, 
                            vk::PipelineStageFlags::TRANSFER, 
                            vk::PipelineStageFlags::FRAGMENT_SHADER, 
                            vk::DependencyFlags::empty(), 
                            &[] as &[vk::MemoryBarrier], 
                            &[] as &[vk::BufferMemoryBarrier], 
                            &[barrier],
                            
                        );
                    }

                    if mip_width > 1 {
                        mip_width /= 2;
                    }

                    if mip_height > 1 {
                        mip_height /= 2;
                    }

                    
            }

            barrier.subresource_range.base_mip_level = mip_levels - 1;
            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer.command_buffer, 
                    vk::PipelineStageFlags::TRANSFER, 
                    vk::PipelineStageFlags::FRAGMENT_SHADER, 
                    vk::DependencyFlags::empty(), 
                    &[] as &[vk::MemoryBarrier], 
                    &[] as &[vk::BufferMemoryBarrier], 
                    &[barrier],
                    
                );
            }

            unsafe {
                command_buffer.end(command_pool)?;
            }
                
            Ok(())

        }
    }

    struct TextureImageWrapper {
        image: ImageWrapper,
        sampler: SamplerWrapper,
        index_buffer: IndexBuffer,
        vertex_buffer: VertexBuffer,
    }

    impl TextureImageWrapper {
        fn new(instance: &InstanceWrapper, device: Arc<DeviceWrapper>, command_pool: &CommandPoolWrapper) -> Result<Self> {
            
            let mut vertices: Vertices = VERTICES.to_vec();
            let mut indices: Indices = INDICES.to_vec();

            let vertex_buffer = VertexBuffer::new(&instance, device.clone(), &command_pool, &vertices)?;
            let index_buffer = IndexBuffer::new(&instance, device.clone(), &command_pool, &indices)?;

            let image = File::open("resources/texture.png").context("Texture file failure")?;

            let decoder = png::Decoder::new(image);
            let mut reader = decoder.read_info()?;

            let mut pixels = vec![0; reader.info().raw_bytes()];
            reader.next_frame(&mut pixels)?;

            let size = reader.info().raw_bytes() as u64;
            let (width, height) = reader.info().size();

            // Create (Staging)

            let staging_buffer = 
                BufferWrapper::new(
                    instance, 
                    device.clone(), 
                    size, 
                    vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE, 
                    vk::BufferUsageFlags::TRANSFER_SRC
                )?;

            // Create (Image)

            let memory = unsafe {
                device.map_memory(
                    staging_buffer.buffer_memory, 
                    0, 
                    size, 
                    vk::MemoryMapFlags::empty()
                )?
            };

            unsafe {
                memcpy(pixels.as_ptr(), memory.cast(), pixels.len());

                device.unmap_memory(staging_buffer.buffer_memory);
            }

            let mut image = ImageWrapper::new(
                instance, 
                device.clone(), 
                width, 
                height, 
                vk::Format::R8G8B8A8_SRGB, 
                vk::ImageTiling::OPTIMAL, 
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST, 
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::ImageAspectFlags::COLOR,
                1,
                vk::SampleCountFlags::_1,
            )?;
            

            // Transition + Copy (Image)

            image.transition_layout(
                command_pool, 
                // vk::Format::R8G8B8A8_SRGB, 
                vk::ImageLayout::UNDEFINED, 
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                1,
            )?;

            image.copy_buffer_to_image(
                device.clone(), 
                command_pool, 
                staging_buffer, 
                width, 
                height,
            )?;

            image.transition_layout(
                command_pool, 
                // vk::Format::R8G8B8A8_SRGB, 
                vk::ImageLayout::TRANSFER_DST_OPTIMAL, 
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                1,
            )?;

            let sampler = SamplerWrapper::new(device.clone(), 1)?;
            
            Ok(Self { image, sampler, vertex_buffer, index_buffer })
        }

        
    }

    struct DepthImageWrapper {
        depth_image: ImageWrapper,
        depth_image_view: ImageViewWrapper,
    }

    impl DepthImageWrapper {
        fn new(
            instance: &InstanceWrapper,
            device: Arc<DeviceWrapper>,
            swapchain: &SwapchainWrapper,
        ) -> Result<Self> {
            let format = DepthImageWrapper::get_depth_format(&device, instance)?;

            let depth_image = ImageWrapper::new(
                instance, 
                device.clone(), 
                swapchain.swapchain_extent.width, 
                swapchain.swapchain_extent.height, 
                format, 
                vk::ImageTiling::OPTIMAL, 
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, 
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::ImageAspectFlags::DEPTH,
                1,
                device.msaa_samples,
            )?;

            let depth_image_view = ImageViewWrapper::new(
                device.clone(), 
                depth_image.image, 
                format,
                vk::ImageAspectFlags::DEPTH,
                1
            )?;



            Ok(Self { depth_image, depth_image_view })
        }

        fn get_supported_format(
            device: &Arc<DeviceWrapper>,
            instance: &InstanceWrapper,
            candidates: &[vk::Format],
            tiling: vk::ImageTiling,
            features: vk::FormatFeatureFlags,
        ) -> Result<vk::Format> {
            candidates
                .iter()
                .cloned()
                .find(|f| {
                    let properties = unsafe {
                        instance.get_physical_device_format_properties(
                            device.physical_device, 
                            *f,
                        )
                    };
                    match tiling {
                        vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                        vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                        _ => true,
                    }
                })
                .ok_or_else(|| anyhow!("Failed to find supported format for depth buffer"))
        }

        fn get_depth_format(
            device: &Arc<DeviceWrapper>,
            instance: &InstanceWrapper,
        ) -> Result<vk::Format> {
            let candidates = &[
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ];

            DepthImageWrapper::get_supported_format(
                device, 
                instance, 
                candidates, 
                vk::ImageTiling::OPTIMAL, 
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
            )
        }


    }
    
    struct ColorImageWrapper {
        color_image: ImageWrapper,
        color_image_view: ImageViewWrapper,
    }

    impl ColorImageWrapper {
        fn new(
            instance: &InstanceWrapper,
            device: Arc<DeviceWrapper>,
            swapchain: &SwapchainWrapper,
            // samples: vk::SampleCountFlags,
        ) -> Result<Self> {
            
            let color_image = ImageWrapper::new(
                instance, 
                device.clone(), 
                swapchain.swapchain_extent.width, 
                swapchain.swapchain_extent.height, 
                swapchain.swapchain_format, 
                vk::ImageTiling::OPTIMAL, 
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT, 
                vk::MemoryPropertyFlags::DEVICE_LOCAL, 
                vk::ImageAspectFlags::COLOR, 
                1, 
                device.msaa_samples,
            )?;

            let color_image_view = ImageViewWrapper::new(
                device, 
                color_image.image, 
                swapchain.swapchain_format, 
                vk::ImageAspectFlags::COLOR, 
                1,
            )?;

            Ok(Self { color_image, color_image_view })
        }
    }

    struct ImageWrapper {
        image: vk::Image,
        image_memory: vk::DeviceMemory,
        image_view: ImageViewWrapper,
        device: Arc<DeviceWrapper>,
    }

    impl ImageWrapper {
        fn new(
            instance: &InstanceWrapper, 
            device: Arc<DeviceWrapper>,
            width: u32,
            height: u32,
            format: vk::Format,
            tiling: vk::ImageTiling,
            usage: vk::ImageUsageFlags,
            properties: vk::MemoryPropertyFlags,
            aspects: vk::ImageAspectFlags,
            mip_levels: u32,
            samples: vk::SampleCountFlags,
        ) -> Result<Self> {
            let image = unsafe {
                let info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::_2D)
                .extent(vk::Extent3D {width, height, depth: 1})
                .mip_levels(mip_levels)
                .array_layers(1)
                .format(format)
                .tiling(tiling)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .samples(samples)
                .flags(vk::ImageCreateFlags::empty());

                device.create_image(&info, None)? 
            };

            let image_memory = unsafe {
                let requirements = 
                    device.get_image_memory_requirements(image);

                let info = vk::MemoryAllocateInfo::builder()
                .allocation_size(requirements.size)
                .memory_type_index(device.get_memory_type_index(
                    instance, 
                    properties, 
                    requirements
                )?);

                device.allocate_memory(&info, None)?
            };

            unsafe {
                device.bind_image_memory(image, image_memory, 0)?;
            }

            let image_view = ImageViewWrapper::new(
                device.clone(), 
                image, 
                format,
                aspects,
                mip_levels,
            )?;

            Ok(Self { image, image_memory, image_view, device })
        }

    
        fn transition_layout(
            &mut self,
            command_pool: &CommandPoolWrapper,
            // format: vk::Format,
            old_layout: vk::ImageLayout,
            new_layout: vk::ImageLayout,
            mip_levels: u32,
        ) -> Result<()> {

            let (
                src_access_mask,
                dst_access_mask,
                src_stage_mask,
                dst_stage_mask,
            ) = match (old_layout, new_layout) {
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                ),
                (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                ),
                _ => return Err(anyhow!("Unssupported image layout transitiopn!")),
            };

            let mut command_buffer = unsafe {
                 OneTimeCommand::new(self.device.clone(), command_pool)?
            };

            let subresource = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(mip_levels)
                .base_array_layer(0)
                .layer_count(1);

            let barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(old_layout)
                .new_layout(new_layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.image)
                .subresource_range(subresource)
                .src_access_mask(src_access_mask)
                .dst_access_mask(dst_access_mask);

            
            unsafe {
                self.device.cmd_pipeline_barrier(
                    command_buffer.command_buffer, 
                    src_stage_mask, 
                    dst_stage_mask, 
                    vk::DependencyFlags::empty(), 
                    &[] as &[vk::MemoryBarrier], 
                    &[] as &[vk::BufferMemoryBarrier],
                    &[barrier],
                );
    
    
                command_buffer.end(command_pool)?;
            }

            Ok(())
        }
    
        fn copy_buffer_to_image(
            &mut self,
            device: Arc<DeviceWrapper>,
            command_pool: &CommandPoolWrapper,
            buffer: BufferWrapper,
            width: u32,
            height: u32,
        ) -> Result<()> {
            let mut command_buffer = unsafe {
                OneTimeCommand::new(device.clone(), command_pool)?
            };

            let subresource = vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1);

            let region = vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(subresource)
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0})
                .image_extent(vk::Extent3D {width, height, depth: 1});


            unsafe {
                device.cmd_copy_buffer_to_image(
                    command_buffer.command_buffer, 
                    buffer.buffer, 
                    self.image, 
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL, 
                    &[region],
                );


                command_buffer.end(command_pool)?;
            }

            Ok(())
        }
    }

    impl Drop for ImageWrapper {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_image(self.image, None);

                self.device.free_memory(self.image_memory, None);
            }
        }
    }

    struct ImageViewWrapper {
        image_view: vk::ImageView,
        device: Arc<DeviceWrapper>,
    }

    impl ImageViewWrapper {
        fn new(
            device: Arc<DeviceWrapper>,
            image: vk::Image,
            format: vk::Format,
            aspects: vk::ImageAspectFlags,
            mip_levels: u32,
        ) -> Result<Self> {
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(aspects)
                .base_mip_level(0)
                .level_count(mip_levels)
                .base_array_layer(0)
                .layer_count(1);

            let info = vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::_2D)
                .format(format)
                .subresource_range(subresource_range);

            Ok(Self {
                image_view: unsafe {
                    device.create_image_view(&info, None)?
                },
                device,
            })
        }
    }

    impl Drop for ImageViewWrapper {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_image_view(self.image_view, None);
            }
        }
    }

    struct SamplerWrapper {
        sampler: vk::Sampler,
        device: Arc<DeviceWrapper>
    }

    impl SamplerWrapper {
        fn new(
            device: Arc<DeviceWrapper>, 
            mip_levels: u32,
        ) -> Result<Self> {
            let info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT) // Handles sampler beyond standard range (0.0 - 1.0)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                .unnormalized_coordinates(false)
                .compare_enable(false)
                .compare_op(vk::CompareOp::ALWAYS)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .mip_lod_bias(0.0)
                .min_lod(0.0)
                .max_lod(mip_levels as f32);

            let sampler = unsafe {
                device.create_sampler(&info, None)?
            };

            Ok(Self { sampler, device })
        }
    }

    impl Drop for SamplerWrapper {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_sampler(self.sampler, None);
            }
            
        }
    }

}

