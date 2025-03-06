import numpy as np
import uuid
import os
import cupy as cp
from scipy.spatial import ConvexHull


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
        self._hull = ConvexHull(self._vertices)
        self._mesh = self._hull.simplices
        self._normals = self._hull.equations[:, :3]
        
        # Compute the area of each mesh element
        vecs_a = self._vertices[self._mesh[:, 1]] - self._vertices[self._mesh[:, 0]]
        vecs_b = self._vertices[self._mesh[:, 2]] - self._vertices[self._mesh[:, 0]]
        self._areas = 0.5 * np.linalg.norm(np.cross(vecs_a, vecs_b), axis=1)

        self._sigma = np.asarray(self._normals) @ self._magnetisation


    def subdivide(self, quantisation: int = 1) -> None:
        # Subdivide mesh then recompute mesh properties
        if quantisation > 1:
            num_triangles = len(self._mesh)
            # For each triangle in the mesh, subdivide it:
            for i in range(num_triangles):
                # Do an n^2 subdivision routine
                vec_a = self._vertices[self._mesh[i, 0], :] - self._vertices[self._mesh[i, 1], :]
                vec_b = self._vertices[self._mesh[i, 2], :] - self._vertices[self._mesh[i, 1], :]
                pt_ctr = len(self._vertices)-1
                for j in range(quantisation):
                    # Add new points
                    for k in range(quantisation-j-1):
                        self._vertices = np.append(self._vertices, [self._vertices[self._mesh[i, 1], :] + (j+0.0)/quantisation * vec_a + (k+1.0)/quantisation * vec_b], axis=0)
                    if j != 0:
                        self._vertices = np.append(self._vertices, [self._vertices[self._mesh[i, 1], :] + (j+0.0)/quantisation * vec_a + (quantisation-j)/quantisation * vec_b], axis=0)
                    if j != quantisation-1:
                        self._vertices = np.append(self._vertices, [self._vertices[self._mesh[i, 1], :] + (j+1.0)/quantisation * vec_a], axis=0)

                # Compute the triangles
                for j in range(quantisation-1):
                    bl_tri = True
                    if j == 0:
                        offset = quantisation
                    else:
                        offset = quantisation - j + 1
                    for k in range(2*(quantisation-j)-1):
                        if bl_tri:
                            self._mesh = np.append(self._mesh, [[pt_ctr, pt_ctr+1, pt_ctr+offset]], axis=0)
                            pt_ctr += 1
                            bl_tri = False
                        else:
                            self._mesh = np.append(self._mesh, [[pt_ctr, pt_ctr+offset, pt_ctr+offset-1]], axis=0)
                            bl_tri = True

                    if j == 0:
                        # Fix ends
                        self._mesh[-2*quantisation+1,0] = self._mesh[i, 1]
                        self._mesh[-1,1] = self._mesh[i, 2]
                    else:
                        offset += 1
                        pt_ctr += 1

                
                self._mesh[i, :] = [len(self._vertices)-2, len(self._vertices)-1, self._mesh[i, 0]]
                self._areas[i] = self._areas[i] / quantisation**2
                self._areas = np.append(self._areas, np.repeat(self._areas[i], quantisation**2 - 1))
                self._normals = np.append(self._normals, np.repeat([self._normals[i, :]], quantisation**2 - 1, axis=0), axis=0)
                self._sigma = np.append(self._sigma, np.repeat(self._sigma[i], quantisation**2 - 1))


    def move(self, displacement: np.array = np.array([0,0,0])) -> None:
        raise NotImplementedError


    def rotate(self, rotation_matrix) -> None:
        raise NotImplementedError


    def transform(self, transform_matrix) -> None:
        raise NotImplementedError


    @property
    def force(self) -> np.array:
        return self._force



class CudaMag:
    def __init__(self) -> None:
        self._magnets: list[Magnet] = []
        self._dir_path = os.path.dirname(os.path.realpath(__file__))


    def add_magnet(self, magnet: Magnet) -> None:
        self._magnets.append(magnet)


    def remove_magnet(self, magnet: Magnet) -> None:
        raise NotImplementedError
        #try:
        #    self.magnets.remove(magnet)
        #    print("Removed magnet.")
        #except:
        #    print("Magnet not found.")


    def solve_system(self, data_type: type = np.float32) -> None:
        # Compute the areas and surface charge densities matrix
        h_sigma = np.zeros((len(self._magnets), sum([len(magnet._areas) for magnet in self._magnets])))
        h_area = np.zeros((len(self._magnets), sum([len(magnet._areas) for magnet in self._magnets])))
        h_num_pts = 0
        for ii, magnet in enumerate(self._magnets):
            h_sigma[ii, h_num_pts:h_num_pts+len(magnet._areas)] = magnet._sigma
            h_area[ii, h_num_pts:h_num_pts+len(magnet._areas)] = magnet._areas
            h_num_pts = h_num_pts + len(magnet._areas)

        # Assign GPU memory
        d_sigma = cp.array(h_sigma, dtype=data_type, order='C')
        d_area = cp.array(h_area, dtype=data_type, order='C')
        d_B = cp.zeros((3, h_num_pts, h_num_pts), dtype=data_type, order='C')
        d_normals = cp.array(np.concatenate(([magnet._normals for magnet in self._magnets]), axis=0), dtype=data_type, order='C')

        d_nodes = cp.array(np.concatenate([magnet._vertices for magnet in self._magnets]), dtype=data_type, order='C')
        h_connections = np.zeros((sum([len(magnet._mesh) for magnet in self._magnets]), 3))
        ctr = 0
        # This is a bit janky, to fix:
        for ii, magnet in enumerate(self._magnets):
            h_connections[ctr:ctr+len(magnet._mesh), :] = np.max(h_connections) + magnet._mesh + ii
            ctr = ctr + len(magnet._mesh)
        d_connections = cp.array(h_connections, dtype=np.uint32, order='C')
        
        threads_per_block = 32
        blocks_per_grid = (int)(np.ceil(h_num_pts / threads_per_block))
        with open(self._dir_path + "/cuda/calcB.cu") as f:
            if data_type == np.float32:
                name_exp = ['calcB<float>']
            elif data_type == np.float64:
                name_exp = ['calcB<double>']
            else:
                raise TypeError("Needs 32-bit or 64-bit floating point type")
            cuda_module = cp.RawModule(code=f.read(), name_expressions=name_exp)
            calc_B_kernel = cuda_module.get_function(name_exp[0])

        num_triangles = sum([np.shape(magnet._mesh)[0] for magnet in self._magnets])
        num_nodes = sum([np.shape(magnet._vertices)[0] for magnet in self._magnets])
        calc_B_kernel((threads_per_block,), (blocks_per_grid,), (d_nodes, d_connections, d_normals, num_nodes, num_triangles, d_B))

        # Compute forces
        d_F = d_sigma @ d_B @ (d_sigma * d_area).transpose() * 1e-7

        h_F = np.sum(cp.asnumpy(d_F), axis=1)
        for ii in range(len(self._magnets)):
            self._magnets[ii]._force = h_F[:, ii]
