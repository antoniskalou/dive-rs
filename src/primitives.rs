#[derive(Copy, Clone, Debug, Default)]
pub struct Vertex {
    pub position: [f32; 3],
}

vulkano::impl_vertex!(Vertex, position);

#[derive(Copy, Clone, Debug, Default)]
pub struct Normal {
    pub normal: [f32; 3],
}

vulkano::impl_vertex!(Normal, normal);

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub normals: Vec<Normal>,
}
