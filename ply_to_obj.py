import numpy as np
import open3d as o3d

# Load point cloud
point_cloud = o3d.io.read_point_cloud("airplane.ply")

# Stuff from tutorial
# point_cloud = o3d.data.BunnyMesh()
# mesh = o3d.io.read_triangle_mesh(point_cloud.path)
# point_cloud = mesh.sample_points_poisson_disk(100000)

point_cloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals


# Estimate normals for the point cloud
point_cloud.estimate_normals()

# Ensure that normals are oriented consistently
point_cloud.orient_normals_consistent_tangent_plane(100)

# Convert point cloud to mesh using Poisson reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)


# Remove low density vertices
vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)

# # Simplify the mesh
# mesh = mesh.simplify_quadric_decimation(100000)

o3d.visualization.draw_geometries([mesh])

# Save the mesh
o3d.io.write_triangle_mesh("airplane.obj", mesh)
o3d.io.write_triangle_mesh("airplane.obj", mesh)
