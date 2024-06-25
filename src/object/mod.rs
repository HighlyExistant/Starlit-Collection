/// Object Header contains the indices to various pieces of Object Information
/// stored in various pool allocators. As these objects are deallocated these
/// headers must update their information. Object Headers are stored in their
/// own unique pool so that we can keep track of whether the object is in use or not
/// by taking into account whether or not an object is in use we can see whether or not
/// to take their information into account when doing operations on them.
/// [in_use]: is a value used in the specialized ObjectHeader pool allocator for seeing
/// whether the object is in use inside the gpu to perform prefix sums.
/// [mesh_idx]: is an offset into the specialized MeshHeader pool allocator.
/// [transform_idx]: is an offset to a mat4x4 to transform local coordinates to world coordinates.
pub struct ObjectHeader {
    in_use: u32,
    mesh_idx: u32,
    transform_idx: u32,
}
/// MeshHeaders contain the location of the vertices and indices inside their respective
/// allocators. Vertices and Indices will have pool allocators dedicated to them.
/// [vertex_size]: denotes the size of the vertex allocation, which will be in terms of
/// FVec4 so vertex_size = 1 means size of 32 bytes.
/// [index_size]: denotes the size of the index allocation, which will be in terms of u32
/// so index_size = 1 means size of 4 bytes.
/// [vertex]: offset in terms of FVec4 inside the vertex allocation.
/// [index]: offset in terms of u32 inside the index allocation.
pub struct MeshHeader {
    vertex_size: u32,
    index_size: u32,
    vertex: u32,
    index: u32,
}