from scipy.spatial import ConvexHull
import numpy as np
import uuid
import os
import cupy as cp


class Magnet:
    def __init__(self, vertices: np.array, magnetisation: np.array, rel_perm: float = 1.0) -> None:
        self._vertices = vertices
        self._magnetisation = magnetisation
        self._rel_perm = rel_perm
        self._force = 0.0
        self._torque = 0.0
        self._id = uuid.uuid4()

        # Proabably come up with a convex hull here
        self._hull = ConvexHull(self._vertices)
        self._connections = self._hull.simplices
        self._nodes = np.array(self._vertices)
        self._sigma = np.matmul(self._hull.equations[:,:3], magnetisation)
        self._centres = np.mean([self._nodes[self._connections[:,i],:] for i in range(3)], axis=2)
        self._areas = np.array(0.5 * np.linalg.norm(np.cross(self._nodes[self._connections[:,0],:] - self._nodes[self._connections[:,1],:], self._nodes[self._connections[:,0],:] - self._nodes[self._connections[:,2],:]), axis=1))
        # TODO: Add subdivision

    # def quadruple_mesh(self) -> None:
    #     con_len = len(self._connections)
    #     for ii in range(con_len):
    #         for jj in range(3):
    #             newpt = 0.5 * np.add(self._nodes[self._connections[ii][jj]], self._nodes[self._connections[ii][(jj + 1) % 3]])
    #             # Append new point to set of nodes
    #             self._nodes = np.concatenate((self._nodes, np.array([newpt])))
    #         # Adapt set of connections and sigma
    #         num_nodes = len(self._nodes)
    #         self._connections = np.concatenate((self._connections, [[self._connections[ii][1], num_nodes - 3, num_nodes - 2], [num_nodes - 3, num_nodes - 2, num_nodes - 1], [num_nodes - 2, num_nodes - 1, self._connections[ii][2]]]))
    #         self._connections[ii][1] = num_nodes - 3
    #         self._connections[ii][2] = num_nodes - 1
    #         self._sigma = np.concatenate((self._sigma,np.array([self._sigma[ii]]*3)))
    #     self._areas = np.concatenate((self._areas / 4, self._areas / 4, self._areas / 4, self._areas / 4))
    #     self._centres = np.mean([self._nodes[self._connections[:,i],:] for i in range(3)], axis=2)


    def subdivide(self, quantisation: int) -> None:
        raise NotImplementedError


    def move(self, displacement) -> None:
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


    def initialise(self) -> None:
        # Is this even necessary?
        pass


    def solve_system(self) -> None:
        # Compute the areas and surface charge densities matrix
        h_sigma_area = np.zeros((len(self._magnets), sum([len(magnet._areas) for magnet in self._magnets])))
        h_num_pts = 0
        for ii, magnet in enumerate(self._magnets):
            h_sigma_area[ii, h_num_pts:h_num_pts+len(magnet._areas)] = magnet._areas * magnet._sigma
            h_num_pts = h_num_pts + len(magnet._areas)
        h_centres = np.concatenate(([magnet._centres.transpose() for magnet in self._magnets]))
        d_centres = cp.array(h_centres, dtype=np.float32)
        d_sigma_area = cp.array(h_sigma_area, dtype=np.float32)
        d_B = cp.zeros((3, h_num_pts, h_num_pts), dtype=np.float32)

        # Construct B matrix
        with open(self._dir_path + "/cuda/calcB.cu") as f:
            calc_B_kernel = cp.RawKernel(f.read(), 'calcB')

            threads_per_block = 32
            blocks_per_grid = 32
            calc_B_kernel((threads_per_block,), (blocks_per_grid,), (d_centres, 24, d_B))

        # Compute forces
        d_F = d_sigma_area @ d_B @ d_sigma_area.transpose()
        print(d_F)
        pass