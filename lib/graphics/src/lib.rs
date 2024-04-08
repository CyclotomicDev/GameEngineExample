use anyhow::{Result,anyhow};
use winit::window::{Window, WindowBuilder};
//use control::control::{InstructionBuffer,Instruction, Layer};
use layers::Layer;
use tokio::sync::Mutex;


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

}

impl Layer for GraphicsHandler {
    
}


/// For holding low-level (unsafe) Vulkan
mod vulkan {
    // General imports
    use winit::window::{Window, WindowBuilder};
    use anyhow::{anyhow, Context, Result};
    use thiserror::Error;
    use log::*;
    use std::sync::Arc;

    use vulkanalia::loader::{LibloadingLoader, LIBRARY};
    use vulkanalia::{bytecode, window as vk_window};
    use vulkanalia::prelude::v1_0::*;
    use vulkanalia::vk::{BufferCreateInfoBuilder, ExtDebugUtilsExtension, Image, ImageView, KhrSurfaceExtension, KhrSwapchainExtension, MemoryPropertyFlags, PipelineColorBlendAttachmentState};

    use std::collections::HashSet;
    use std::ffi::CStr;
    use std::os::raw::c_void;
    use std::ops::{Deref, DerefMut};

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
        recreate: RecreateWrapper, // Handles objects that may be recreated
        vertex_buffer: VertexBuffer,
        device: Arc<DeviceWrapper>, // Destroyed after all other objects
        instance: InstanceWrapper, // Destroyed after device
        
        frame: usize,
        resized: bool,
    }

    impl VulkanHandler {
        pub fn new(window: &Window) -> Result<Self> {
            let instance = InstanceWrapper::new(window)?;
            let device = Arc::new(DeviceWrapper::new(&instance, window)?);
            let vertex_buffer = VertexBuffer::new(&instance, device.clone())?;
            let recreate = RecreateWrapper::new(device.clone(), &instance, window, &vertex_buffer)?;

            /*
            let swapchain = SwapchainWrapper::new(&instance, device.clone() , window)?;
            let pipeline = PipelineWrapper::new(device.clone(), swapchain.swapchain_extent, swapchain.swapchain_format)?;
            let framebuffers = DeviceWrapper::create_framebuffers(&logical_device, &swapchain, &pipeline.render_pass)?;
            let command = CommandWrapper::new(&instance, physical_device, &logical_device, &framebuffers, &swapchain, &pipeline, &vertex_buffer)?;
            */
            Ok(Self {instance, device, frame: 0, resized: false,vertex_buffer, recreate })
        }

        pub fn render(&mut self, window: &Window) -> Result<()> {
            let logical_device = &self.device.logical_device;
            let image_available_semaphores = &self.recreate.command.image_available_semaphores;
            let render_finsished_semaphores = &self.recreate.command.render_finished_semaphores;
            let in_flight_fences = &self.recreate.command.in_flight_fences;
            let images_in_flight = &mut self.recreate.command.images_in_flight;
            let frame = self.frame;

            // Wait for other frames to be completed; safety: both device and in_flight_fences created
            unsafe {
                self.device.logical_device.wait_for_fences(
                    &[in_flight_fences[frame]], 
                    true, 
                    u64::MAX,
                )?
            };

            // 1. Aquire image from swapchain

            
            let result = unsafe {
                logical_device.acquire_next_image_khr(
                    self.recreate.swapchain.swapchain, 
                    u64::MAX, 
                    self.recreate.command.image_available_semaphores[frame], 
                    vk::Fence::null()
                )
            };

            // Check if swapchain needs to be recreated (e.g. window resize)
            let image_index = match result {
                Ok((image_index, _)) => image_index as usize,
                Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return  self.recreate_swapchain(window),
                Err(e) => return  Err(anyhow!(e)),
            };

            // Safety: Each fence must not be currently associated with any queue command that has not completed execution
            unsafe {
                logical_device.reset_fences(&[in_flight_fences[frame]])?
            };


            let swapchain = self.recreate.swapchain.swapchain;

            
            /*
            let image_index = unsafe {
                logical_device
                    .acquire_next_image_khr(
                        swapchain, 
                        u64::MAX, 
                        image_available_semaphores[self.frame], 
                        vk::Fence::null(),
                    )?
                    .0 as usize
            };
            */

            if !images_in_flight[image_index].is_null() {
                unsafe {
                    logical_device.wait_for_fences(
                        &[images_in_flight[image_index]],
                        true, 
                        u64::MAX,
                    )?;
                }
            }

            images_in_flight[image_index] = in_flight_fences[frame];

            // 2. Execute command buffer with image as attachment in the framebuffer

            let wait_semaphores = &[image_available_semaphores[self.frame]];
            let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = &[self.recreate.command.command_buffers[image_index as usize]];
            let signal_semaphores = &[render_finsished_semaphores[self.frame]];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(wait_stages)
                .command_buffers(command_buffers)
                .signal_semaphores(signal_semaphores);

            unsafe {
                self.device.logical_device.queue_submit(
                    self.device.graphics_queue, 
                    &[submit_info], 
                    in_flight_fences[self.frame])?  
            };

            let swapchains = &[swapchain];
            let image_indices = &[image_index as u32];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(signal_semaphores)
                .swapchains(swapchains)
                .image_indices(image_indices);

            let result = unsafe {
                self.device.logical_device.queue_present_khr(self.device.present_queue, &present_info)
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
            self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

            Ok(())
        }

        pub fn set_resized(&mut self) {
            self.resized = true;
        }
        

        fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
            
            unsafe {
                self.device.device_wait_idle()?
            };

            self.recreate = RecreateWrapper::new(self.device.clone(), &self.instance, window, &self.vertex_buffer)?;
            
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
        command: CommandWrapper,
        framebuffers: Vec<vk::Framebuffer>,
        pipeline: PipelineWrapper,
        swapchain: SwapchainWrapper,
        device: Arc<DeviceWrapper>,
    }

    impl RecreateWrapper {
        fn new(device: Arc<DeviceWrapper>, instance: &InstanceWrapper, window: &Window, vertex_buffer: &VertexBuffer) -> Result<Self> {
            let swapchain = SwapchainWrapper::new(instance, device.clone(), window)?;
            let pipeline = PipelineWrapper::new(device.clone(), swapchain.swapchain_extent, swapchain.swapchain_format)?;
            let framebuffers = RecreateWrapper::create_framebuffers(&**device, &swapchain, &pipeline.render_pass)?;
            let mut command = CommandWrapper::new(&instance, device.clone(), &framebuffers, &swapchain, &pipeline, vertex_buffer)?;
            command.images_in_flight.resize(swapchain.swapchain_images.len(), vk::Fence::null());

            Ok(Self { command, framebuffers, pipeline, swapchain, device })
        }

        fn create_framebuffers(logical_device: &Device, swapchain: &SwapchainWrapper, render_pass: &vk::RenderPass) -> Result<Vec<vk::Framebuffer>> {
            let framebuffers = 
                swapchain.swapchain_image_views
                .iter()
                .map(|i| {
                    let attachments = &[*i];
                    let create_info = vk::FramebufferCreateInfo::builder()
                        .render_pass(*render_pass)
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

    impl Drop for RecreateWrapper {
        fn drop(&mut self) {
            unsafe {
                self.framebuffers
                    .iter()
                    .for_each(|f| self.device.destroy_framebuffer(*f, None));
            }
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
                        
                        return Ok(Self { physical_device, logical_device, graphics_queue, present_queue});
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

        fn get_memory_type_index(physical_device: vk::PhysicalDevice, instance: &InstanceWrapper, properties: vk::MemoryPropertyFlags, requirements: vk::MemoryRequirements) -> Result<u32> {
            
            // Two arrays: memory_types and memory_heaps
            let memory = unsafe {
                instance.instance.get_physical_device_memory_properties(physical_device)
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
        device: Arc<DeviceWrapper>
    }

    impl SwapchainWrapper {
        fn new(instance: &InstanceWrapper, device: Arc<DeviceWrapper>, window: &Window) -> Result<Self> {

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

            let swapchain_image_views = SwapchainWrapper::create_swapchain_image_views(surface_format.format, &swapchain_images, &**device)?;

            Ok(Self { swapchain_format: surface_format.format, swapchain_extent, swapchain, swapchain_images, swapchain_image_views, device })
        }

        fn create_swapchain_image_views(swapchain_format: vk::Format, swapchain_images: &Vec<Image>,logical_device: &Device) -> Result<Vec<ImageView>> {
            Ok(swapchain_images
                    .iter()
                    .map(|i| {
                        SwapchainWrapper::create_image_view(swapchain_format, logical_device, *i, vk::ImageAspectFlags::COLOR)
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
                self.swapchain_image_views
                    .iter()
                    .for_each(|v| self.device.destroy_image_view(*v, None));

                self.device.destroy_swapchain_khr(self.swapchain, None);
            }
        }
    }

    struct PipelineWrapper {
        pipeline_layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        pipeline: vk::Pipeline,
        device: Arc<DeviceWrapper>
    }

    impl PipelineWrapper {
        fn new(device: Arc<DeviceWrapper>, swapchain_extent: vk::Extent2D, swapchain_format: vk::Format) -> Result<Self> {
            let vert = include_bytes!("../shaders/vert.spv");
            let frag = include_bytes!("../shaders/frag.spv");

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
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false)
            };
                
            let multisample_state = {vk::PipelineMultisampleStateCreateInfo::builder()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::_1)
            };
            
            let attachment = {vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::all())
                .blend_enable(false)
            };

            let attachments = &[attachment];
            let color_blend_state = {vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(attachments)
                .blend_constants([0.0, 0.0, 0.0, 0.0])
            };
            
            let pipeline_layout ={
                let layout_info = vk::PipelineLayoutCreateInfo::builder();
                unsafe {
                    logical_device.create_pipeline_layout(&layout_info, None)?
                }
            };

            // let render_pass = RenderPassWeapper::new(swapchain_format, logical_device)?;

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

            let stages = &[vert_stage, frag_stage];
            let info = {vk::GraphicsPipelineCreateInfo::builder()
                .stages(stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
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

    struct CommandWrapper {
        command_pool: vk::CommandPool,
        command_buffers: Vec<vk::CommandBuffer>,
        // Semaphores to control GPU syncronization
        image_available_semaphores: Vec<vk::Semaphore>,
        render_finished_semaphores: Vec<vk::Semaphore>,
        // Fences to control CPU syncronization
        in_flight_fences: Vec<vk::Fence>,
        images_in_flight: Vec<vk::Fence>,
        device: Arc<DeviceWrapper>,
    }

    impl CommandWrapper {
        fn new(instance: &InstanceWrapper, 
               device: Arc<DeviceWrapper>,
               framebuffers: &Vec<vk::Framebuffer>, 
               swapchain: &SwapchainWrapper, 
               pipeline: &PipelineWrapper, 
               vertex_buffer: &VertexBuffer,
        ) -> Result<Self> {
            let command_pool = CommandWrapper::create_command_pool(instance, &*device)?;
            let command_buffers  = CommandWrapper::create_command_buffers(&*device, command_pool, framebuffers, swapchain, pipeline, vertex_buffer)?;

            let (image_available_semaphores, render_finished_semaphores, in_flight_fences) = unsafe {
                let semaphore_info = vk::SemaphoreCreateInfo::builder();
                let fence_info = vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::SIGNALED); // Want to have fences appear as if just complete frame at start rather than causing wait for non-existant frame

                let mut image_available_semaphores = Vec::new();
                let mut render_finished_semaphores = Vec::new();

                let mut in_flight_fences = Vec::new();

                for _ in 0..MAX_FRAMES_IN_FLIGHT {
                    image_available_semaphores
                        .push(device.create_semaphore(&semaphore_info, None)?);
                    render_finished_semaphores
                        .push(device.create_semaphore(&semaphore_info, None)?);
                
                    in_flight_fences
                        .push(device.create_fence(&fence_info, None)?);
                }

                (image_available_semaphores, render_finished_semaphores, in_flight_fences)
            };

            let images_in_flight = swapchain.swapchain_images
                .iter()
                .map(|_| vk::Fence::null())
                .collect();

            Ok(Self { command_pool, command_buffers, image_available_semaphores, render_finished_semaphores, in_flight_fences, images_in_flight, device })
        }

        fn create_command_pool(instance: &InstanceWrapper, device: &DeviceWrapper) -> Result<vk::CommandPool> {
            let indices = QueueFamilyIndices::new(instance, device.physical_device)?;

            let info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::empty())
                .queue_family_index(indices.graphics);

            let command_pool = unsafe {
                device.create_command_pool(&info, None)?
            };

            Ok(command_pool)
        }
    
        fn create_command_buffers(device: &DeviceWrapper, command_pool: vk::CommandPool, framebuffers: &Vec<vk::Framebuffer>, swapchain: &SwapchainWrapper, pipeline: &PipelineWrapper, vertex_buffer: &VertexBuffer) -> Result<Vec<vk::CommandBuffer>> {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(framebuffers.len() as u32);

            let command_buffers = unsafe {
                device.allocate_command_buffers(&allocate_info) ?
            };

            for (i, command_buffer) in command_buffers.iter().enumerate() {
                let inheritance = vk::CommandBufferInheritanceInfo::builder();

                let info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::empty())
                    .inheritance_info(&inheritance);

                unsafe {
                    device.begin_command_buffer(*command_buffer, &info)?
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
                    .render_pass(pipeline.render_pass)
                    .framebuffer(framebuffers[i])
                    .render_area(render_area)
                    .clear_values(clear_values);

                unsafe {
                    device.cmd_begin_render_pass(*command_buffer, &info, vk::SubpassContents::INLINE);
                    device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
                    device.cmd_bind_vertex_buffers(*command_buffer, 0, &[vertex_buffer.vertex_buffer.buffer], &[0]);
                    device.cmd_draw(*command_buffer, VERTICES.len() as u32, 1, 0, 0);
                    device.cmd_end_render_pass(*command_buffer);
                    device.end_command_buffer(*command_buffer)?;
                }
                
            }

            Ok(command_buffers)
        }

    }

    impl Drop for CommandWrapper {
        fn drop(&mut self) {
            unsafe {
                // Semaphores must match ones created
                self.render_finished_semaphores
                    .drain(..)
                    .for_each(|s| self.device.destroy_semaphore(s, None));
                self.image_available_semaphores
                    .drain(..)
                    .for_each(|s| self.device.destroy_semaphore(s, None));
                
                self.in_flight_fences
                    .drain(..)
                    .for_each(|f| self.device.destroy_fence(f, None));

                self.device.destroy_command_pool(self.command_pool, None);
            }
        }
    }
    
    use std::mem::size_of;
    use cgmath::{vec2, vec3};

    type Vec2 = cgmath::Vector2<f32>;
    type Vec3 = cgmath::Vector3<f32>;

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    struct Vertex {
        pos: Vec2,
        color: Vec3,
    }

    impl Vertex {
        const  fn new(pos: Vec2, color: Vec3) -> Self {
            Self {pos, color}
        }

        fn binding_description() -> vk::VertexInputBindingDescription {
            vk::VertexInputBindingDescription::builder()
                .binding(0)
                .stride(size_of::<Vertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)
                .build()
        }

        fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
            let pos = vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0)
                .build();

            let color = vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(size_of::<Vec2>() as u32)
                .build();

            [pos, color]
        }
    }

    static VERTICES: [Vertex; 3] = [
        Vertex::new(vec2(0.0, -0.5), vec3(1.0, 0.0, 0.0)),
        Vertex::new(vec2(0.5, 0.5), vec3(0.0, 1.0, 0.0)),
        Vertex::new(vec2(-0.5, 0.5), vec3(0.0, 0.0, 1.0)),
    ];

    // Buffers

    use std::ptr::copy_nonoverlapping as memcpy;

    struct Buffer {
        buffer_info: BufferCreateInfoBuilder<'static>,
        buffer_memory: vk::DeviceMemory,
        buffer: vk::Buffer,
        device: Arc<DeviceWrapper>,
    }

    impl Buffer {
        fn new(instance: &InstanceWrapper,
               device: Arc<DeviceWrapper>, 
               properties: vk::MemoryPropertyFlags, 
               usage: vk::BufferUsageFlags,
            ) -> Result<Self> {
            
            let buffer_info = vk::BufferCreateInfo::builder()
                .size((size_of::<Vertex>() * VERTICES.len()) as u64)
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
                .memory_type_index(DeviceWrapper::get_memory_type_index(device.physical_device, instance, properties, requirements)?);

            let buffer_memory = unsafe {
                device.allocate_memory(&memory_info, None)?  
            };

            unsafe {
                device.bind_buffer_memory(buffer,buffer_memory, 0)?  
            };

            Ok(Self { buffer_memory, buffer, buffer_info, device })
        }
    }

    impl Drop for Buffer {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_buffer(self.buffer, None);
                self.device.free_memory(self.buffer_memory, None)
            } 
        }
    }

    struct VertexBuffer {
        vertex_buffer: Buffer
    }

    impl VertexBuffer {
        fn new(instance: &InstanceWrapper,
               device: Arc<DeviceWrapper>,
        ) -> Result<Self> {

            let vertex_buffer = Buffer::new(
                instance, 
                device.clone(),
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE, 
                vk::BufferUsageFlags::VERTEX_BUFFER
            )?;

            let memory = unsafe {
                device.map_memory(
                    vertex_buffer.buffer_memory, 
                    0, 
                    vertex_buffer.buffer_info.size,
                    vk::MemoryMapFlags::empty()
                )
            }?;

            unsafe {
                memcpy(VERTICES.as_ptr(), memory.cast(), VERTICES.len());
                device.unmap_memory(vertex_buffer.buffer_memory);
            };

            Ok(Self { vertex_buffer })
        }
    }

}

