import numpy as np
import uuid
import os
import cupy as cp
import open3d as o3d


class Magnet:
    def __init__(self, vertices: np.array, magnetisation: np.array, rel_perm: float = 1.0) -> None:
        # Initialisation
        self._vertices = vertices
        self._magnetisation = magnetisation
        self._rel_perm = rel_perm
        self._force = 0.0
        self._torque = 0.0
        self._id = uuid.uuid4()

        # Set up pointcloud
        self._pc = o3d.t.geometry.PointCloud(vertices)
        self._mesh = self._pc.compute_convex_hull().to_legacy()

        # Compute the mesh properties
        self.compute_mesh_properties()


    def subdivide(self, quantisation: int = 1) -> None:
        # Subdivide mesh then recompute mesh properties
        self._mesh = self._mesh.subdivide_midpoint(quantisation)
        self.compute_mesh_properties()

    
    def compute_mesh_properties(self) -> None:
        self._mesh.compute_triangle_normals()
        self._connections = np.asarray(self._mesh.triangles)
        self._nodes = np.asarray(self._mesh.vertices)
        self._sigma = np.asarray(self._mesh.triangle_normals) @ self._magnetisation
        self._centres = np.mean([self._nodes[self._connections[:,i],:] for i in range(3)], axis=0)
        self._areas = np.array(0.5 * np.linalg.norm(np.cross(self._nodes[self._connections[:,0],:] - self._nodes[self._connections[:,1],:], self._nodes[self._connections[:,0],:] - self._nodes[self._connections[:,2],:]), axis=1))


    def move(self, displacement: np.array = np.array([0,0,0])) -> None:
        raise NotImplementedError


    def rotate(self, rotation_matrix) -> None:
        raise NotImplementedError


    def transform(self, transform_matrix) -> None:
        raise NotImplementedError



class CudaMag:
    def __init__(self) -> None:
        self._magnets: list[Magnet] = []
        self._dir_path = os.path.dirname(os.path.realpath(__file__))
        print("Initialised class.")


    def add_magnet(self, magnet: Magnet) -> None:
        self._magnets.append(magnet)
        print("Added magnet.")


    def remove_magnet(self, magnet: Magnet) -> None:
        raise NotImplementedError
        #try:
        #    self.magnets.remove(magnet)
        #    print("Removed magnet.")
        #except:
        #    print("Magnet not found.")


    def solve_system(self) -> None:
        # Compute the areas and surface charge densities matrix
        h_sigma = np.zeros((len(self._magnets), sum([len(magnet._areas) for magnet in self._magnets])))
        h_area = np.zeros((len(self._magnets), sum([len(magnet._areas) for magnet in self._magnets])))
        h_num_pts = 0
        h_centres = np.concatenate(([magnet._centres.transpose() for magnet in self._magnets]), axis=1)
        for ii, magnet in enumerate(self._magnets):
            h_sigma[ii, h_num_pts:h_num_pts+len(magnet._areas)] = magnet._sigma
            h_area[ii, h_num_pts:h_num_pts+len(magnet._areas)] = magnet._areas
            h_num_pts = h_num_pts + len(magnet._areas)

        # Assign GPU memory
        d_centres = cp.array(h_centres, dtype=np.float32)
        d_sigma = cp.array(h_sigma, dtype=np.float32)
        d_area = cp.array(h_area, dtype=np.float32)
        d_B = cp.zeros((3, h_num_pts, h_num_pts), dtype=np.float32)

        # Set up CUDA code and construct B matrix
        threads_per_block = 32
        blocks_per_grid = (int)(np.ceil(h_num_pts / threads_per_block))
        with open(self._dir_path + "/cuda/calcB.cu") as f:
            calc_B_kernel = cp.RawKernel(f.read(), 'calcB')

        calc_B_kernel((threads_per_block,), (blocks_per_grid,), (d_centres, h_num_pts, d_B))

        # Compute forces
        d_F = d_area * d_sigma @ d_B @ (d_sigma * d_area).transpose() * 1e-7

        h_F = np.sum(cp.asnumpy(d_F), axis=1)
        for ii in range(len(self._magnets)):
            self._magnets[ii]._force = h_F[:, ii]
