from scipy.spatial import ConvexHull
import numpy as np
import uuid


class Magnet:
    def __init__(self, vertices: list[list[float]], magnetisation: list[float], rel_perm: float = 1.0) -> None:
        self._vertices = vertices
        self._magnetisation = magnetisation
        self._rel_perm = rel_perm
        self.force = 0.0
        self.torque = 0.0
        self.id = uuid.uuid4()

        # Proabably come up with a convex hull here
        self.hull = ConvexHull(self._vertices)
        self.connections = self.hull.simplices
        self.nodes = self._vertices
        self.sigma = np.matmul(self.hull.equations[:,:3], magnetisation)
        # TODO: Add subdivision

    # def quadruple_mesh(self) -> None:
    #     con_len = len(self.connections)
    #     for ii in range(con_len):
    #         for jj in range(3):
    #             newpt = 0.5 * np.add(self.nodes[self.connections[ii][jj]], self.nodes[self.connections[ii][(jj + 1) % 3]])
    #             # Append new point to set of nodes
    #             self.nodes.append(list(newpt))
    #         # Adapt set of connections and sigma
    #         num_nodes = len(self.nodes)
    #         self.connections = np.concatenate((self.connections, [[self.connections[ii][1], num_nodes - 3, num_nodes - 2], [num_nodes - 3, num_nodes - 2, num_nodes - 1], [num_nodes - 2, num_nodes - 1, self.connections[ii][2]]]))
    #         self.connections[ii][1] = num_nodes - 3
    #         self.connections[ii][2] = num_nodes - 1
    #         self.sigma = np.concatenate((self.sigma,np.array([self.sigma[ii]]*3)))

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
        print("Initialised class.")


    def __del__(self) -> None: # Called automatically by garbage collector
        pass


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
        pass


    def solve_system(self) -> None:
        pass