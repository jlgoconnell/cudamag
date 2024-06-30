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
        self.connectivity = self.hull.simplices
        self.nodes = self.vertices # TODO: Add subdivision

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
        self.nodes: list[list[float]] = []
        self.connectivity: list[list[int]] = []

        # Setup C/C++ functions
        dll = ctypes.CDLL('./cudainterface.so')
        self.get_magnet_system = dll.getMagnetSystem
        self.get_magnet_system.argtypes = None
        self.get_magnet_system.restype = None
        self.destroy_magnet_system = dll.destroyMagnetSystem
        self.destroy_magnet_system.argtypes = None
        self.destroy_magnet_system.restype = None
        self.init = dll.init
        self.init.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.init.restype = None

        # Create a magnet system
        self.mag_sys = self.get_magnet_system()

        print("Initialised Python class.")

    def __del__(self) -> None: # Called automatically by garbage collector
        self.destroy_magnet_system(self.mag_sys)


    def add_magnet(self, magnet: Magnet) -> None:
        self.magnets.append(magnet)
        print("Added magnet.")


    def remove_magnet(self, magnet: Magnet) -> None:
        try:
            self.magnets.remove(magnet)
            print("Removed magnet.")
        except:
            print("Magnet not found.")


    def initialise(self) -> None:
        # Combine all magnets into a single set of data structures
        for magnet in self.magnets:
            self.connectivity.extend((magnet.connectivity + len(self.nodes)).tolist())
            self.nodes.extend(magnet.nodes)

        # Set up data structures for use in C
        p_nodes = [item for sublist in self.nodes for item in sublist]
        c_nodes = (ctypes.c_float * len(p_nodes))(*p_nodes)
        p_connectivity = [item for sublist in self.connectivity for item in sublist]
        c_connectivity = (ctypes.c_int * len(p_connectivity))(*p_connectivity)
        
        # Call the init() function
        self.init(ctypes.cast(c_nodes, ctypes.POINTER(ctypes.c_float)), len(p_nodes), ctypes.cast(c_connectivity, ctypes.POINTER(ctypes.c_int)), len(p_connectivity))


    def solve_system(self) -> None:
        print("Solving system.")
