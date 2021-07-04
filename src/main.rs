use std::sync::Arc;
use vulkano::{
    device::{Device, DeviceExtensions, Features},
    instance::{Instance, PhysicalDevice},
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

struct Renderer {
    device: Arc<Device>,
}

impl Renderer {
    fn new(device: Arc<Device>) -> Self {
        Self {
            device,
        }
    }
}

fn main() {
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, vulkano::Version::V1_2, &extensions, None)
            .expect("Failed to create Vulkan instance")
    };

    let physical_device = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("No Vulkan devices are available");

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name.as_ref().unwrap(),
        physical_device.properties().device_type.unwrap(),
    );

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let queue_family = physical_device
        .queue_families()
        .find(|&q| {
            // We take the first queue that supports drawing to our window.
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        })
        .unwrap();

    let (device, mut queues) = {
        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };

        Device::new(
            physical_device,
            &Features { shading_rate_image: false , ..*physical_device.supported_features() },
            &device_ext,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .unwrap()
    };

    let _renderer = Renderer::new(device.clone());

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
            *control_flow = ControlFlow::Exit;
        },
        _ => {},
    });
}