use vulkano::{
    instance::{Instance, PhysicalDevice},
    sync::{self, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use renderer::*;

mod renderer;
mod primitives;

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
        .with_title("Dive: A deep sea adventure")
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let mut renderer = Renderer::start(physical_device.clone(), surface.clone()).unwrap();
    let mut previous_frame_end = Some(sync::now(renderer.device()).boxed());
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
            *control_flow = ControlFlow::Exit;
        },
        Event::WindowEvent { event: WindowEvent::Resized(dimensions), .. } => {
            renderer.window_resized([dimensions.width, dimensions.height]);
        },
        Event::RedrawEventsCleared => {
            previous_frame_end.as_mut().unwrap().cleanup_finished();
            renderer.render(previous_frame_end.take().unwrap());
            // TODO: handle somewhere else, also, handle GpuFuture failure
            previous_frame_end = Some(sync::now(renderer.device()).boxed());
        },
        _ => {},
    });
}