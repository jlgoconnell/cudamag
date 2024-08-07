from ctypes import *
import ctypes
from scipy.spatial import ConvexHull
import numpy as np
import uuid


class Magnet:
    def __init__(self, vertices: list[list[float]], magnetisation: list[float], rel_perm: float = 1.0) -> None:
        self.vertices = vertices
        self.magnetisation = magnetisation
        self.rel_perm = rel_perm
        self.force = 0.0
        self.torque = 0.0
        self.id = uuid.uuid4()

        # Proabably come up with a convex hull here
        self.hull = ConvexHull(self.vertices)
        self.connections = self.hull.simplices
        self.nodes = self.vertices
        self.sigma = np.matmul(self.hull.equations[:,:3], magnetisation)
        # TODO: Add subdivision

    def quadruple_mesh(self) -> None:
        con_len = len(self.connections)
        for ii in range(con_len):
            for jj in range(3):
                newpt = 0.5 * np.add(self.nodes[self.connections[ii][jj]], self.nodes[self.connections[ii][(jj + 1) % 3]])
                # Append new point to set of nodes
                self.nodes.append(list(newpt))
            # Adapt set of connections and sigma
            num_nodes = len(self.nodes)
            self.connections = np.concatenate((self.connections, [[self.connections[ii][1], num_nodes - 3, num_nodes - 2], [num_nodes - 3, num_nodes - 2, num_nodes - 1], [num_nodes - 2, num_nodes - 1, self.connections[ii][2]]]))
            self.connections[ii][1] = num_nodes - 3
            self.connections[ii][2] = num_nodes - 1
            self.sigma = np.concatenate((self.sigma,np.array([self.sigma[ii]]*3)))

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
        self.magnets: list[Magnet] = []
        self.magnet_idx: list[int] = []
        self.nodes: list[list[float]] = []
        self.connections: list[list[int]] = []
        self.sigma: list[float] = []

        # Setup C/C++ functions
        dll = ctypes.CDLL('./cudamag_cu.so')
        self.get_magnet_system = dll.getMagnetSystem
        self.get_magnet_system.argtypes = None
        self.get_magnet_system.restype = None
        self.destroy_magnet_system = dll.destroyMagnetSystem
        self.destroy_magnet_system.argtypes = None
        self.destroy_magnet_system.restype = None
        self.init = dll.init
        self.init.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.init.restype = None
        self.solve = dll.solve
        self.solve.argtypes = None
        self.solve.restype = None

        # Create a magnet system
        self.mag_sys = self.get_magnet_system()

        print("Initialised Python class.")

    def __del__(self) -> None: # Called automatically by garbage collector
        self.destroy_magnet_system(self.mag_sys)


    def add_magnet(self, magnet: Magnet) -> None:
        self.magnets.append(magnet)
        print("Added magnet.")


    def remove_magnet(self, magnet: Magnet) -> None:
        raise NotImplementedError
        #try:
        #    self.magnets.remove(magnet)
        #    print("Removed magnet.")
        #except:
        #    print("Magnet not found.")


    def initialise(self) -> None:
        # Combine all magnets into a single set of data structures
        for magnet in self.magnets:
            self.magnet_idx.append(len(self.connections))
            self.connections.extend((magnet.connections + len(self.nodes)).tolist())
            self.nodes.extend(magnet.nodes)
            self.sigma.extend(magnet.sigma)

        print(self.magnet_idx)

        # Set up data structures for use in C
        p_nodes = [item for sublist in self.nodes for item in sublist]
        c_nodes = (ctypes.c_float * len(p_nodes))(*p_nodes)
        p_connections = [item for sublist in self.connections for item in sublist]
        c_connections = (ctypes.c_int * len(p_connections))(*p_connections)
        p_sigma = self.sigma
        c_sigma = (ctypes.c_float * len(p_sigma))(*p_sigma)
        p_mag_idx = self.magnet_idx
        c_mag_idx = (ctypes.c_int * len(self.magnet_idx))(*self.magnet_idx)
        c_num_mags = len(self.magnets)

        
        # Call the init() function
        self.init(ctypes.cast(c_nodes, ctypes.POINTER(ctypes.c_float)), int(len(p_nodes)/3), ctypes.cast(c_connections, ctypes.POINTER(ctypes.c_int)), int(len(p_connections)/3), ctypes.cast(c_sigma, ctypes.POINTER(ctypes.c_float)), ctypes.cast(c_mag_idx, ctypes.POINTER(ctypes.c_int)), c_num_mags)


    def solve_system(self) -> None:
        print("Solving system.")
        self.solve()
