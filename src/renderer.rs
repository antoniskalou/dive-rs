use nalgebra_glm as glm;
use std::sync::Arc;
use vulkano::{
    buffer::{cpu_pool::CpuBufferPool, BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, DynamicState, SubpassContents,
        PrimaryAutoCommandBuffer,
    },
    device::{Device, DeviceExtensions, DeviceOwned, Features, Queue},
    descriptor::descriptor_set::{DescriptorSet, PersistentDescriptorSet},
    format::{ClearValue, Format},
    instance::PhysicalDevice,
    render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass},
    pipeline::{
        viewport::Viewport,
        vertex::TwoBuffersDefinition,
        GraphicsPipeline, GraphicsPipelineAbstract
    },
    swapchain::{self, AcquireError, Surface, Swapchain, SwapchainCreationError},
    image::{
        attachment::AttachmentImage,
        view::{ImageView, ImageViewAbstract},
        ImageUsage, SwapchainImage
    },
    sync::{FlushError, GpuFuture},
};
use winit::window::Window;
use crate::primitives::{Vertex, Normal};

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

pub struct RenderContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    surface: Arc<Surface<Window>>,
}

pub struct DrawContext {
    pub commands: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    pub descriptor_sets: Vec<Arc<dyn DescriptorSet + Send + Sync>>,
}

/// Draws images onto another image.
pub struct DrawTarget {
    color_attachment: Arc<ImageView<Arc<AttachmentImage>>>,
    depth_stencil_attachment: Arc<ImageView<Arc<AttachmentImage>>>,
    framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,
}

impl DrawTarget {
    pub fn new(render_context: &RenderContext, dimensions: [u32; 2]) -> Result<Self, String> {
        let color_format = [Format::R8G8B8A8Unorm]
            .iter()
            .cloned()
            .find(|format| {
                let physical_device = render_context.device.physical_device();
                let features = format.properties(physical_device).optimal_tiling_features;
                features.color_attachment && features.blit_src
            })
            .ok_or("Device does not support color format".to_owned())?;

        let depth_stencil_format = [
            Format::D32Sfloat_S8Uint,
            Format::D24Unorm_S8Uint,
            Format::D16Unorm_S8Uint,
        ]
            .iter()
            .cloned()
            .find(|format| {
                let physical_device = render_context.device.physical_device();
                let features = format.properties(physical_device).optimal_tiling_features;
                features.depth_stencil_attachment
            })
            .ok_or("Device does not support depth stencils".to_owned())?;

        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(render_context.device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: color_format,
                        samples: 1,
                    },
                    depth_stencil: {
                        load: Clear,
                        store: DontCare,
                        format: depth_stencil_format,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {depth_stencil}
                }
            )
            .map_err(|e| format!("Failed to create render pass: {:?}", e))?
        );

        let (color_attachment, depth_stencil_attachment) = Self::create_attachments(
            render_context.device.clone(), 
            dimensions, 
            color_format, 
            depth_stencil_format
        )?;

        let framebuffer = Arc::new(
            Framebuffer::start(render_pass)
                .add(color_attachment.clone()).unwrap()
                .add(depth_stencil_attachment.clone()).unwrap()
                .build()
                .map_err(|e| format!("Failed to create framebuffer: {:?}", e))?
        );

        Ok(Self {
            color_attachment,
            depth_stencil_attachment,
            framebuffer,
        })
    }

    pub fn resize(&mut self, render_context: &RenderContext, dimensions: [u32; 2]) -> Result<(), String> {
        let (color_attachment, depth_attachment) = Self::create_attachments(
            render_context.device.clone(),
            dimensions,
            self.color_attachment.format(),
            self.depth_stencil_attachment.format(),
        )?;
        self.color_attachment = color_attachment;
        self.depth_stencil_attachment = depth_attachment;

        self.framebuffer = Arc::new(
            Framebuffer::start(self.render_pass().clone())
                .add(self.color_attachment.clone()).unwrap()
                .add(self.depth_stencil_attachment.clone()).unwrap()
                .build()
                .map_err(|e| format!("Failed to build framebuffer: {:?}", e))?
        );

        Ok(())
    }

    pub fn render_pass(&self) -> &Arc<RenderPass> {
        &self.framebuffer.render_pass()
    }

    pub fn framebuffer(&self) -> &Arc<dyn FramebufferAbstract + Send + Sync> {
        &self.framebuffer
    }

    pub fn color_attachment(&self) -> &Arc<ImageView<Arc<AttachmentImage>>> {
        &self.color_attachment
    }

    fn create_attachments(
        device: Arc<Device>, 
        dimensions: [u32; 2], 
        color_format: Format, 
        depth_stencil_format: Format
    ) -> Result<(
        Arc<ImageView<Arc<AttachmentImage>>>,
        Arc<ImageView<Arc<AttachmentImage>>>
    ), String> {
        let color_attachment = ImageView::new(
            AttachmentImage::with_usage(
                device.clone(),
                dimensions,
                color_format,
                ImageUsage {
                    color_attachment: true,
                    transfer_source: true,
                    ..ImageUsage::none()
                }
            ).map_err(|e| format!("Couldn't create color attachment: {:?}", e))?
        ).unwrap();

        let depth_stencil_attachment = ImageView::new(
            AttachmentImage::with_usage(
                device.clone(),
                dimensions,
                depth_stencil_format,
                ImageUsage {
                    depth_stencil_attachment: true,
                    transient_attachment: true,
                    ..ImageUsage::none()
                }
            ).map_err(|e| format!("Couldn't create depth stencil attachment: {:?}", e))?
        ).unwrap();

        Ok((color_attachment, depth_stencil_attachment))
    }
}

/// Presents images on screen.
pub struct PresentTarget {
    pub images: Vec<Arc<SwapchainImage<Window>>>,
    pub swapchain: Arc<Swapchain<Window>>,
    needs_recreate: bool,
}

impl PresentTarget {
    pub fn new(device: Arc<Device>, surface: Arc<Surface<Window>>) -> Result<Self, String> {
        let caps = surface.capabilities(device.physical_device()).unwrap();
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
                .composite_alpha(alpha)
                .build()
                .map_err(|e| format!("Failed to create swapchain: {:?}", e))?;

        Ok(Self {
            images,
            swapchain,
            needs_recreate: false,
        })
    }

    fn recreate(&mut self) -> Result<(), SwapchainCreationError> {
        let surface = self.swapchain.surface();

        let caps = surface.capabilities(self.swapchain.device().physical_device()).unwrap();
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        let (swapchain, images) = self.swapchain.recreate()
            .format(format)
            .dimensions(dimensions)
            .composite_alpha(alpha)
            .build()?;

        self.swapchain = swapchain;
        self.images = images;

        Ok(())
    }

    pub fn dimensions(&self) -> [u32; 2] {
        self.swapchain.dimensions()
    }

    pub fn needs_recreate(&self) -> bool {
        self.needs_recreate
    }

    pub fn window_resized(&mut self) {
        self.needs_recreate = true;
    }

    pub fn present(
        &mut self,
        queue: Arc<Queue>,
        commands: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        draw_future: impl GpuFuture,
    ) -> Result<(), String> {
        if self.needs_recreate() {
            return Ok(());
        }

        let (image_num, suboptimal, swapchain_future) = 
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(ok) => ok,
                Err(AcquireError::OutOfDate) => {
                    self.needs_recreate = true;
                    return Ok(());
                }
                Err(e) => Err(format!("Couldn't acquire frame buffer: {:?}", e))?,
            };

        self.needs_recreate = suboptimal;

        let fence_future = draw_future
            .join(swapchain_future)
            .then_execute(queue.clone(), commands.build().unwrap())
            .unwrap()
            .then_swapchain_present(queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence();

        match fence_future.wait(None) {
            Ok(_) => (),
            Err(FlushError::OutOfDate) => self.needs_recreate = true,
            Err(e) => Err(format!("Failed to fence swapchain: {:?}", e))?,
        }

        Ok(())
    }
}

pub struct Renderer {
    pub render_context: RenderContext,
    pub present_target: PresentTarget,
    pub draw_target: DrawTarget,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
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

        let render_context = RenderContext {
            surface: surface.clone(),
            device: device.clone(), 
            queue,
        };

        let present_target = PresentTarget::new(device.clone(), surface.clone()).unwrap();
        let draw_target = DrawTarget::new(&render_context, present_target.dimensions()).unwrap();

        let vs = vs::Shader::load(device.clone()).map_err(|e| format!("Failed to create vertex shader: {:?}", e))?;
        let fs = fs::Shader::load(device.clone()).map_err(|e| format!("Failed to create fragment shader: {:?}", e))?;

        let pipeline = window_size_dependent_setup(
            device.clone(), 
            &vs, 
            &fs, 
            draw_target.render_pass().clone(), 
            present_target.dimensions()
        );

        let uniform_buffer = CpuBufferPool::new(device.clone(), BufferUsage::all());

        Ok(Self {
            render_context,
            present_target,
            draw_target,
            pipeline,
            uniform_buffer,
        })
    }

    pub fn window_resized(&mut self, dimensions: [u32; 2]) {
        if dimensions != self.dimensions() {
            self.present_target.window_resized();
            self.draw_target.resize(&self.render_context, dimensions).unwrap();
        }
    }

    pub fn render(&mut self, draw_future: impl GpuFuture)  {
        if self.present_target.needs_recreate() {
            self.present_target.recreate().unwrap();
        }

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

        let dynamic_state = DynamicState::none();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.render_context.device.clone(), 
            self.render_context.queue.family(), 
            CommandBufferUsage::OneTimeSubmit
        ).unwrap();

        builder
            .begin_render_pass(
                self.draw_target.framebuffer().clone(),
                SubpassContents::Inline,
                std::array::IntoIter::new([
                    ClearValue::Float([0.0, 0.0, 0.0, 1.0]).into(),
                    ClearValue::DepthStencil((1.0, 0)).into(),
                ]),
            )
            .unwrap()
            .end_render_pass()
            .unwrap();

        let draw_context = DrawContext {
            commands: AutoCommandBufferBuilder::primary(
                self.render_context.device.clone(), 
                self.render_context.queue.family(),
                CommandBufferUsage::OneTimeSubmit
            ).unwrap(),
            descriptor_sets: Vec::with_capacity(12),
        };

        self.present_target.present(
            self.render_context.queue.clone(), 
            builder,
            draw_future
        ).unwrap();
    }

    pub fn dimensions(&self) -> [u32; 2] {
        self.render_context.surface.window().inner_size().into()
    }

    pub fn device(&self) -> Arc<Device> {
        self.render_context.device.clone()
    }
}

fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: &vs::Shader,
    fs: &fs::Shader,
    render_pass: Arc<RenderPass>,
    dimensions: [u32; 2]
) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync>
 {
    Arc::new(
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
    )
}