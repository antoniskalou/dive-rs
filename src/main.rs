use nalgebra_glm as glm;
use std::sync::Arc;
use vulkano::{
    buffer::{cpu_pool::CpuBufferPool, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, DynamicState, SubpassContents
    },
    device::{Device, DeviceExtensions, Features, Queue},
    descriptor::descriptor_set::PersistentDescriptorSet,
    format::Format,
    instance::{Instance, PhysicalDevice},
    render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass},
    pipeline::{
        viewport::Viewport,
        vertex::TwoBuffersDefinition,
        GraphicsPipeline, GraphicsPipelineAbstract
    },
    swapchain::{self, AcquireError, Surface, Swapchain, SwapchainCreationError},
    image::{
        attachment::AttachmentImage,
        view::ImageView,
        ImageUsage, SwapchainImage
    },
    sync::{self, FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[derive(Copy, Clone, Default)]
pub struct Vertex {
    pub position: [f32; 3],
}

vulkano::impl_vertex!(Vertex, position);

#[derive(Copy, Clone, Default)]
pub struct Normal {
    pub normal: [f32; 3],
}

vulkano::impl_vertex!(Normal, normal);

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub normals: Vec<Normal>,
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    v_normal = transpose(inverse(mat3(worldview))) * normal;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
}
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 0) out vec4 f_color;

const vec3 LIGHT = vec3(0.0, 0.0, 1.0);

void main() {
    float brightness = dot(normalize(v_normal), normalize(LIGHT));
    vec3 dark_color = vec3(0.6, 0.0, 0.0);
    vec3 regular_color = vec3(1.0, 0.0, 0.0);

    f_color = vec4(mix(dark_color, regular_color, brightness), 1.0);
}
        "
    }
}

fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: &vs::Shader,
    fs: &fs::Shader,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>
) -> (
    Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
) {
    let dimensions = images[0].dimensions();

    let depth_buffer = ImageView::new(
        AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap()
    )
    .unwrap();

    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(view)
                    .unwrap()
                    .add(depth_buffer.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>();

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input(TwoBuffersDefinition::<Vertex, Normal>::new())
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .viewports(std::iter::once(Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0..1.0,
            }))
            .fragment_shader(fs.main_entry_point(), ())
            .depth_stencil_simple_depth()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap()
    );

    (pipeline, framebuffers)
}

pub struct Renderer {
    surface: Arc<Surface<Window>>,
    pub device: Arc<Device>,
    queue: Arc<Queue>,
    // FIXME: synchronisation issues between swapchain and images?
    swapchain: Arc<Swapchain<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    vs: vs::Shader,
    fs: fs::Shader,
    uniform_buffer: CpuBufferPool<vs::ty::Data>,
}

impl Renderer {
    pub fn start<'a>(physical_device: PhysicalDevice<'a>, surface: Arc<Surface<Window>>) -> Result<Self, String> {
        let queue_family = physical_device
            .queue_families()
            .find(|&q| {
                // We take the first queue that supports drawing to our window.
                q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
            })
            .ok_or("Failed to find a device with required features".to_owned())?;

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
            ).map_err(|e| format!("Failed to create virtual device: {:?}", e))
        }?;

        let queue = queues.next().ok_or("No device queues are available")?;

        let caps = surface.capabilities(physical_device).unwrap();
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        let (swapchain, images) = 
            Swapchain::start(device.clone(), surface.clone())
                .usage(ImageUsage::color_attachment())
                .num_images(caps.min_image_count)
                .format(format)
                .dimensions(dimensions)
                .usage(ImageUsage::color_attachment())
                .sharing_mode(&queue)
                .composite_alpha(alpha)
                .build()
                .map_err(|e| format!("Failed to create swapchain: {:?}", e))?;

        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: swapchain.format(),
                        samples: 1,
                    },
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: Format::D16Unorm,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {depth}
                }
            ).map_err(|e| format!("Failed to create render pass: {:?}", e))?,
        );

        let vs = vs::Shader::load(device.clone()).map_err(|e| format!("Failed to create vertex shader: {:?}", e))?;
        let fs = fs::Shader::load(device.clone()).map_err(|e| format!("Failed to create fragment shader: {:?}", e))?;

        let (pipeline, framebuffers) = window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone());

        let uniform_buffer = CpuBufferPool::new(device.clone(), BufferUsage::all());

        Ok(Self {
            surface: surface.clone(),
            device: device.clone(), 
            queue,
            swapchain,
            images,
            render_pass,
            pipeline,
            framebuffers,
            vs,
            fs,
            uniform_buffer,
        })
    }

    pub fn recreate_swapchain(&mut self) -> Result<(), SwapchainCreationError> {
        let dimensions: [u32; 2] = self.dimensions();
        let (swapchain, images) = self.swapchain.recreate().dimensions(dimensions).build()?;
        self.swapchain = swapchain;
        self.images = images;

        let (pipeline, framebuffers) = 
            window_size_dependent_setup(self.device.clone(), &self.vs, &self.fs, &self.images, self.render_pass.clone());
        self.pipeline = pipeline;
        self.framebuffers = framebuffers;

        Ok(())
    }

    pub fn render(&mut self, draw_future: impl GpuFuture) -> bool {
        let uniform_buffer_subbuffer = {
            let rotation = glm::rotation(0.0, &glm::Vec3::y_axis());

            // FIXME: remove after teapot, it has OpenGL coords
            let aspect_ratio = self.dimensions()[0] as f32 / self.dimensions()[1] as f32;
            let proj = glm::perspective(aspect_ratio, std::f32::consts::FRAC_PI_2, 0.01, 100.0);
            let view = glm::look_at_rh(
                &glm::vec3(0.3, 0.3, 1.0), 
                &glm::vec3(0.0, 0.0, 0.0), 
                &glm::vec3(0.0, -1.0, 0.0),
            );
            let scale = glm::identity::<f32, 4>() * 0.01;

            let uniform_data = vs::ty::Data {
                world: rotation.into(),
                view: (view * scale).into(),
                proj: proj.into(),
            };

            self.uniform_buffer.next(uniform_data).unwrap()
        };

        let layout = self.pipeline.layout().descriptor_set_layout(0).unwrap();
        let set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(uniform_buffer_subbuffer)
                .unwrap()
                .build()
                .unwrap(),
        );

        let (image_num, suboptimal, acquire_future) = 
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    return true;
                },
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };

        if suboptimal { 
            return true; 
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(), self.queue.family(), CommandBufferUsage::OneTimeSubmit
        ).unwrap();

        builder
            .begin_render_pass(
                self.framebuffers[image_num].clone(),
                SubpassContents::Inline,
                vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()]
            )
            .unwrap()
            .end_render_pass()
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = draw_future
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => (),
            Err(FlushError::OutOfDate) => {
                return true;
            },
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
            }
        }

        false
    }

    pub fn dimensions(&self) -> [u32; 2] {
        self.surface.window().inner_size().into()
    }

    pub fn add_mesh(&mut self, mesh: Mesh) -> Result<(), String> {
        Ok(())
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
        .with_title("Dive: A deep sea adventure")
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let mut renderer = Renderer::start(physical_device.clone(), surface.clone()).unwrap();
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(renderer.device.clone()).boxed());
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
            *control_flow = ControlFlow::Exit;
        },
        Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
            recreate_swapchain = true;
        },
        Event::RedrawEventsCleared => {
            previous_frame_end.as_mut().unwrap().cleanup_finished();

            if recreate_swapchain { 
                renderer.recreate_swapchain().unwrap();
             }

            recreate_swapchain = renderer.render(previous_frame_end.take().unwrap());
            // TODO: handle somewhere else, also, handle GpuFuture failure
            previous_frame_end = Some(sync::now(renderer.device.clone()).boxed());
        },
        _ => {},
    });
}