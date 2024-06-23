from ctypes import *
import ctypes
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

class CudaMag:
    def __init__(self) -> None:
        self.magnets: list[Magnet] = []
        self.num_magnets = 0
        print("Initialised magnet system.\n")

    def add_magnet(self, magnet: Magnet) -> None:
        self.magnets.append(magnet)
        print("Added magnet.\n")

    def solve_system(self) -> None:
        print("Solved system.\n")

#magnet = Magnet([[1.0]], [1.0], 1.0)
#cudamag = CudaMag()

def get_createMagnetSystem():
    dll = ctypes.CDLL('./cudainterface.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.createMagnetSystem
    func.argtypes = []
    return func

__createMagnetSystem = get_createMagnetSystem()

if __name__ == "__main__":
    __createMagnetSystem()