use crate::primitives::{Normal, Vertex};
use nalgebra_glm as glm;
use std::sync::Arc;
use vulkano::{
    buffer::{cpu_pool::CpuBufferPool, BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, DynamicState, PrimaryAutoCommandBuffer,
        SubpassContents,
    },
    descriptor::descriptor_set::{DescriptorSet, PersistentDescriptorSet},
    device::{Device, DeviceExtensions, DeviceOwned, Features, Queue},
    format::{ClearValue, Format},
    image::{
        attachment::AttachmentImage,
        view::{ImageView, ImageViewAbstract},
        ImageUsage, SwapchainImage,
    },
    instance::PhysicalDevice,
    pipeline::{
        vertex::TwoBuffersDefinition, viewport::Viewport, GraphicsPipeline,
        GraphicsPipelineAbstract,
    },
    render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass},
    swapchain::{self, AcquireError, Surface, Swapchain, SwapchainCreationError},
    sync::{FlushError, GpuFuture},
};
use winit::window::Window;
use genmesh::{Vertices, MapVertex, Indexer, MapToVertices, Triangulate, generators::{SharedVertex, IndexedPolygon}};

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
    commands: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    descriptor_sets: Vec<Arc<dyn DescriptorSet + Send + Sync>>,
}

/// Draws images onto another image.
pub struct DrawTarget {
    render_pass: Arc<RenderPass>,
    depth_stencil_attachment: Arc<ImageView<Arc<AttachmentImage>>>,
    framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
}

impl DrawTarget {
    pub fn new(
        render_context: &RenderContext,
        images: &Vec<Arc<SwapchainImage<Window>>>,
        dimensions: [u32; 2],
    ) -> Result<Self, String> {
        let color_format = ImageView::new(images[0].clone()).unwrap().format();

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
            .map_err(|e| format!("Failed to create render pass: {:?}", e))?,
        );

        let depth_stencil_attachment = Self::create_attachments(
            render_context.device.clone(),
            dimensions,
            depth_stencil_format,
        )?;

        let framebuffers = images
            .iter()
            .map(|image| {
                let framebuffer: Arc<dyn FramebufferAbstract + Send + Sync> = Arc::new(
                    Framebuffer::start(render_pass.clone())
                        .add(ImageView::new(image.clone()).unwrap())
                        .unwrap()
                        .add(depth_stencil_attachment.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                );

                framebuffer
            })
            .collect::<Vec<_>>();

        Ok(Self {
            render_pass,
            depth_stencil_attachment,
            framebuffers,
        })
    }

    pub fn resize(
        &mut self,
        render_context: &RenderContext,
        images: &Vec<Arc<SwapchainImage<Window>>>,
        dimensions: [u32; 2],
    ) -> Result<(), String> {
        self.depth_stencil_attachment = Self::create_attachments(
            render_context.device.clone(),
            dimensions,
            self.depth_stencil_attachment.format(),
        )?;

        self.framebuffers = images
            .iter()
            .map(|image| {
                let framebuffer: Arc<dyn FramebufferAbstract + Send + Sync> = Arc::new(
                    Framebuffer::start(self.render_pass.clone())
                        .add(ImageView::new(image.clone()).unwrap())
                        .unwrap()
                        .add(self.depth_stencil_attachment.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                );

                framebuffer
            })
            .collect::<Vec<_>>();

        Ok(())
    }

    pub fn render_pass(&self) -> &Arc<RenderPass> {
        &self.render_pass
    }

    pub fn framebuffer(
        &self,
        image_num: usize,
    ) -> Option<&Arc<dyn FramebufferAbstract + Send + Sync>> {
        self.framebuffers.get(image_num)
    }

    fn create_attachments(
        device: Arc<Device>,
        dimensions: [u32; 2],
        depth_stencil_format: Format,
    ) -> Result<Arc<ImageView<Arc<AttachmentImage>>>, String> {
        let depth_stencil_attachment = ImageView::new(
            AttachmentImage::with_usage(
                device.clone(),
                dimensions,
                depth_stencil_format,
                ImageUsage {
                    depth_stencil_attachment: true,
                    transient_attachment: true,
                    ..ImageUsage::none()
                },
            )
            .map_err(|e| format!("Couldn't create depth stencil attachment: {:?}", e))?,
        )
        .unwrap();

        Ok(depth_stencil_attachment)
    }
}

/// Presents images on screen.
pub struct PresentTarget {
    images: Vec<Arc<SwapchainImage<Window>>>,
    swapchain: Arc<Swapchain<Window>>,
    needs_recreate: bool,
}

impl PresentTarget {
    pub fn new(device: Arc<Device>, surface: Arc<Surface<Window>>) -> Result<Self, String> {
        let caps = surface.capabilities(device.physical_device()).unwrap();
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        let (swapchain, images) = Swapchain::start(device.clone(), surface.clone())
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

        let caps = surface
            .capabilities(self.swapchain.device().physical_device())
            .unwrap();
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        let (swapchain, images) = self
            .swapchain
            .recreate()
            .format(format)
            .dimensions(dimensions)
            .composite_alpha(alpha)
            .build()?;

        self.swapchain = swapchain;
        self.images = images;

        self.needs_recreate = false;

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

    pub fn acquire(&mut self) -> Result<(impl GpuFuture, usize), AcquireError> {
        let (image_num, suboptimal, swapchain_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(ok) => ok,
                Err(AcquireError::OutOfDate) => {
                    self.needs_recreate = true;
                    return Err(AcquireError::OutOfDate);
                }
                Err(e) => Err(e)?,
            };

        self.needs_recreate = suboptimal;

        Ok((swapchain_future, image_num))
    }

    pub fn present(
        &mut self,
        queue: Arc<Queue>,
        commands: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        swapchain_future: impl GpuFuture,
        draw_future: impl GpuFuture,
        image_num: usize,
    ) -> Result<(), String> {
        if self.needs_recreate() {
            return Ok(());
        }

        let fence_result = draw_future
            .join(swapchain_future)
            .then_execute(queue.clone(), commands.build().unwrap())
            .unwrap()
            .then_swapchain_present(queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match fence_result {
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
    vs: vs::Shader,
    fs: fs::Shader,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    uniform_buffer: CpuBufferPool<vs::ty::Data>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    normal_buffer: Arc<CpuAccessibleBuffer<[Normal]>>,
    index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
}

impl Renderer {
    pub fn start<'a>(
        physical_device: PhysicalDevice<'a>,
        surface: Arc<Surface<Window>>,
    ) -> Result<Self, String> {
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
                &Features {
                    shading_rate_image: false,
                    ..*physical_device.supported_features()
                },
                &device_ext,
                [(queue_family, 0.5)].iter().cloned(),
            )
            .map_err(|e| format!("Failed to create virtual device: {:?}", e))
        }?;

        let queue = queues.next().ok_or("No device queues are available")?;

        let render_context = RenderContext {
            surface: surface.clone(),
            device: device.clone(),
            queue,
        };

        let present_target = PresentTarget::new(device.clone(), surface.clone()).unwrap();
        let draw_target = DrawTarget::new(
            &render_context,
            &present_target.images,
            present_target.dimensions(),
        )
        .unwrap();

        let vs = vs::Shader::load(device.clone())
            .map_err(|e| format!("Failed to create vertex shader: {:?}", e))?;
        let fs = fs::Shader::load(device.clone())
            .map_err(|e| format!("Failed to create fragment shader: {:?}", e))?;

        let pipeline = start_graphics_pipeline(
            device.clone(),
            &vs,
            &fs,
            draw_target.render_pass().clone(),
            present_target.dimensions(),
        );

        let mut vertices: Vec<(Vertex, Normal)> = Vec::new();
        let indices: Vec<u16> = {
            let mut indexer = genmesh::LruIndexer::new(8, |_, v: genmesh::Vertex| {
                vertices.push((
                    Vertex{ position: [v.pos.x, v.pos.y, v.pos.z] },
                    Normal { normal: [v.normal.x, v.normal.y, v.normal.z] }
                ))
            });

            genmesh::generators::Cube::new()
                .triangulate()
                .vertex(|v| indexer.index(v))
                .vertices()
                .map(|i| i as u16)
                .collect()
        };

        let uniform_buffer = CpuBufferPool::new(device.clone(), BufferUsage::all());
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            vertices.iter().map(|(vertex, _normal)| vertex).cloned(),
        )
        .unwrap();
        let normal_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            vertices.iter().map(|(_vertex, normal)| normal).cloned(),
        )
        .unwrap();
        let index_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            indices.iter().cloned(),
        )
        .unwrap();

        Ok(Self {
            render_context,
            present_target,
            draw_target,
            pipeline,
            uniform_buffer,
            vertex_buffer,
            normal_buffer,
            index_buffer,
            fs,
            vs,
        })
    }

    pub fn window_resized(&mut self, dimensions: [u32; 2]) {
        if dimensions != self.dimensions() {
            self.present_target.window_resized();
        }
    }

    pub fn render(&mut self, draw_future: impl GpuFuture) {
        if self.present_target.needs_recreate() {
            self.present_target.recreate().unwrap();
            self.draw_target
                .resize(&self.render_context, &self.present_target.images, self.present_target.dimensions())
                .unwrap();
            self.pipeline = start_graphics_pipeline(
                self.render_context.device.clone(),
                &self.vs,
                &self.fs,
                self.draw_target.render_pass().clone(),
                self.dimensions(),
            );
            return;
        }

        let uniform_buffer_subbuffer = {
            let world: glm::Mat4 = glm::identity();

            // FIXME: remove after teapot, it has OpenGL coords
            let aspect_ratio = self.dimensions()[0] as f32 / self.dimensions()[1] as f32;
            let proj = glm::perspective(aspect_ratio, std::f32::consts::FRAC_PI_2, 0.01, 100.0);
            let view = glm::look_at_rh(
                &glm::vec3(-3.0, 1.0, 1.0),
                &glm::vec3(0.0, 0.0, 0.0),
                &glm::vec3(0.0, -1.0, 0.0),
            );
            // let scale = glm::identity::<f32, 4>() * 0.01;

            let uniform_data = vs::ty::Data {
                world: world.into(),
                view: view.into(),
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

        let mut builder = AutoCommandBufferBuilder::primary(
            self.render_context.device.clone(),
            self.render_context.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let (swapchain_future, image_num) = self.present_target.acquire().unwrap();

        builder
            .begin_render_pass(
                self.draw_target.framebuffer(image_num).unwrap().clone(), // issue is (probably) here, framebuffer isn't attached to image
                SubpassContents::Inline,
                std::array::IntoIter::new([
                    ClearValue::Float([0.0, 0.0, 0.0, 1.0]).into(),
                    ClearValue::DepthStencil((1.0, 0)).into(),
                ]),
            )
            .unwrap()
            .draw_indexed(
                self.pipeline.clone(),
                &DynamicState::none(),
                vec![self.vertex_buffer.clone(), self.normal_buffer.clone()],
                self.index_buffer.clone(),
                set.clone(),
                (),
                None,
            )
            .unwrap()
            .end_render_pass()
            .unwrap();

        self.present_target
            .present(
                self.render_context.queue.clone(),
                builder,
                swapchain_future,
                draw_future,
                image_num,
            )
            .unwrap();
    }

    pub fn dimensions(&self) -> [u32; 2] {
        self.render_context.surface.window().inner_size().into()
    }

    // FIXME: remove me
    pub fn device(&self) -> Arc<Device> {
        self.render_context.device.clone()
    }
}

fn start_graphics_pipeline(
    device: Arc<Device>,
    vs: &vs::Shader,
    fs: &fs::Shader,
    render_pass: Arc<RenderPass>,
    dimensions: [u32; 2],
) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
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
            .unwrap(),
    )
}
