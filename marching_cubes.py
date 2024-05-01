import numpy as np
import plotly.graph_objects as go

from constants import edgeTable, triTable
from utils import vertex_interp


class MarchingSquaresSolver:
    def __init__(self, grid_dim, grid_size):
        """
        grid_dim: number of cells in each dimension
        grid_size: size of the grid in one dimension
        """
        self.grid_dim = grid_dim
        self.grid_size = grid_size

        self.cell_size = grid_size / grid_dim
        self.axis = np.linspace(0, grid_size, grid_dim + 1)
        self.isolevels = np.zeros([grid_dim + 1, grid_dim + 1, grid_dim + 1]) * np.inf

        xv, yv, zv = np.meshgrid(self.axis, self.axis, self.axis, indexing="ij")
        self.grid = np.stack([xv, yv, zv], axis=3)

    def solve(self, particles):
        """
        particles: [N, 3] array of particles
        """
        # Radius and h are hyperparams
        h = 0.1

        total_nonzero = 0

        for i in range(self.grid_dim + 1):
            for j in range(self.grid_dim + 1):
                for k in range(self.grid_dim + 1):
                    # Look within a radius r of the cell.

                    # Calculate distance from each particle to the voxel
                    dist = np.sqrt(np.sum((particles - self.grid[i, j, k]) ** 2, axis=1))

                    # Get particles within the radius
                    dist_in_radius = dist[dist <= h]
                    total_nonzero += len(dist_in_radius) > 0

                    smoothing_constant = 315 / (64 * np.pi * h**9)

                    self.isolevels[i, j, k] = np.sum(smoothing_constant * (h**2 - dist_in_radius**2) ** 3)

        print(total_nonzero)
        print(self.isolevels)

        print(np.mean(self.isolevels), np.std(self.isolevels))

        print(np.min(self.isolevels), np.max(self.isolevels))

    def polygonize(self):
        isolevel = 5  # Ca tune
        self.triangles = []
        for i in range(self.grid_dim):
            for j in range(self.grid_dim):
                for k in range(self.grid_dim):
                    cube_index = 0
                    cube_coords = [np.zeros(3) for _ in range(8)]
                    cube_coords[0] = self.grid[i, j, k]
                    cube_coords[1] = self.grid[i + 1, j, k]
                    cube_coords[2] = self.grid[i + 1, j + 1, k]
                    cube_coords[3] = self.grid[i, j + 1, k]
                    cube_coords[4] = self.grid[i, j, k + 1]
                    cube_coords[5] = self.grid[i + 1, j, k + 1]
                    cube_coords[6] = self.grid[i + 1, j + 1, k + 1]
                    cube_coords[7] = self.grid[i, j + 1, k + 1]

                    cube_isolevel = [
                        self.isolevels[i, j, k],
                        self.isolevels[i + 1, j, k],
                        self.isolevels[i + 1, j + 1, k],
                        self.isolevels[i, j + 1, k],
                        self.isolevels[i, j, k + 1],
                        self.isolevels[i + 1, j, k + 1],
                        self.isolevels[i + 1, j + 1, k + 1],
                        self.isolevels[i, j + 1, k + 1],
                    ]

                    if self.isolevels[i, j, k] < isolevel:
                        cube_index |= 1
                    if self.isolevels[i + 1, j, k] < isolevel:
                        cube_index |= 2
                    if self.isolevels[i + 1, j + 1, k] < isolevel:
                        cube_index |= 4
                    if self.isolevels[i, j + 1, k] < isolevel:
                        cube_index |= 8
                    if self.isolevels[i, j, k + 1] < isolevel:
                        cube_index |= 16
                    if self.isolevels[i + 1, j, k + 1] < isolevel:
                        cube_index |= 32
                    if self.isolevels[i + 1, j + 1, k + 1] < isolevel:
                        cube_index |= 64
                    if self.isolevels[i, j + 1, k + 1] < isolevel:
                        cube_index |= 128

                    # print(cube_index)

                    if edgeTable[cube_index] == 0:
                        continue

                    vert_list = np.zeros([12, 3])
                    if edgeTable[cube_index] & 1:
                        vert_list[0] = vertex_interp(
                            isolevel,
                            cube_coords[0],
                            cube_coords[1],
                            cube_isolevel[0],
                            cube_isolevel[1],
                        )
                    if edgeTable[cube_index] & 2:
                        vert_list[1] = vertex_interp(
                            isolevel,
                            cube_coords[1],
                            cube_coords[2],
                            cube_isolevel[1],
                            cube_isolevel[2],
                        )
                    if edgeTable[cube_index] & 4:
                        vert_list[2] = vertex_interp(
                            isolevel,
                            cube_coords[2],
                            cube_coords[3],
                            cube_isolevel[2],
                            cube_isolevel[3],
                        )
                    if edgeTable[cube_index] & 8:
                        vert_list[3] = vertex_interp(
                            isolevel,
                            cube_coords[3],
                            cube_coords[0],
                            cube_isolevel[3],
                            cube_isolevel[0],
                        )
                    if edgeTable[cube_index] & 16:
                        vert_list[4] = vertex_interp(
                            isolevel,
                            cube_coords[4],
                            cube_coords[5],
                            cube_isolevel[4],
                            cube_isolevel[5],
                        )
                    if edgeTable[cube_index] & 32:
                        vert_list[5] = vertex_interp(
                            isolevel,
                            cube_coords[5],
                            cube_coords[6],
                            cube_isolevel[5],
                            cube_isolevel[6],
                        )
                    if edgeTable[cube_index] & 64:
                        vert_list[6] = vertex_interp(
                            isolevel,
                            cube_coords[6],
                            cube_coords[7],
                            cube_isolevel[6],
                            cube_isolevel[7],
                        )
                    if edgeTable[cube_index] & 128:
                        vert_list[7] = vertex_interp(
                            isolevel,
                            cube_coords[7],
                            cube_coords[4],
                            cube_isolevel[7],
                            cube_isolevel[4],
                        )
                    if edgeTable[cube_index] & 256:
                        vert_list[8] = vertex_interp(
                            isolevel,
                            cube_coords[0],
                            cube_coords[4],
                            cube_isolevel[0],
                            cube_isolevel[4],
                        )
                    if edgeTable[cube_index] & 512:
                        vert_list[9] = vertex_interp(
                            isolevel,
                            cube_coords[1],
                            cube_coords[5],
                            cube_isolevel[1],
                            cube_isolevel[5],
                        )
                    if edgeTable[cube_index] & 1024:
                        vert_list[10] = vertex_interp(
                            isolevel,
                            cube_coords[2],
                            cube_coords[6],
                            cube_isolevel[2],
                            cube_isolevel[6],
                        )
                    if edgeTable[cube_index] & 2048:
                        vert_list[11] = vertex_interp(
                            isolevel,
                            cube_coords[3],
                            cube_coords[7],
                            cube_isolevel[3],
                            cube_isolevel[7],
                        )

                    idx = 0
                    while triTable[cube_index][idx] != -1:

                        self.triangles.append(
                            [
                                vert_list[triTable[cube_index][idx]],
                                vert_list[triTable[cube_index][idx + 1]],
                                vert_list[triTable[cube_index][idx + 2]],
                            ]
                        )
                        idx += 3


if __name__ == "__main__":
    solver = MarchingSquaresSolver(20, 1)

    # particles = np.random.rand(10000, 3) * 10
    # Generate particles that are at z = 0 but vary across x and y
    particles = np.random.rand(10000, 3) / 10

    # Vary z from
    # particles[:, 2] = np.random.rand(100000) / 2

    # Graph particles
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=particles[:, 0],
            y=particles[:, 1],
            z=particles[:, 2],
            mode="markers",
            marker=dict(size=2),
        )
    )
    fig.show()

    solver.solve(particles)
    solver.polygonize()

    fig = go.Figure()
    for triangle in solver.triangles:
        fig.add_trace(
            go.Mesh3d(
                x=triangle[0],
                y=triangle[1],
                z=triangle[2],
                color="lightpink",
            )
        )
    fig.show()
