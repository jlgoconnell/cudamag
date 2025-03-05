import cudamag
import numpy as np
import pytest as pt
import matplotlib.pyplot as plt

def test_akoun_yonnet():

    # Set up geometry
    d_vec = np.arange(0, 0.03, 0.002)
    Fx = [0.6, 0.25, 0, -0.25, -0.6, -0.9, -1.05, -1.1, -1.05, -0.88, -0.55, -0.25, -0.08, 0, 0.05, 0.05]
    Fy = [0.6, 0.63, 0.64, 0.63, 0.6, 0.51, 0.425, 0.31, 0.23, 0.14, 0.05, 0.02, 0, -0.01, -0.02, -0.02]
    Fz = [-1.76, -1.85, -1.85, -1.85, -1.76, -1.47, -1.08, -0.64, -0.25, 0.15, 0.4, 0.43, 0.33, 0.25, 0.18, 0.15]

    for ii in range(len(d_vec)):
        d = d_vec[ii]

        system = cudamag.CudaMag()

        bottom_magnet = cudamag.Magnet(
        vertices=np.array([[-0.01, -0.006, -0.003],
                [0.01, -0.006, -0.003],
                [-0.01, 0.006, -0.003],
                [0.01, 0.006, -0.003],
                [-0.01, -0.006, 0.003],
                [0.01, -0.006, 0.003],
                [-0.01, 0.006, 0.003],
                [0.01, 0.006, 0.003]]),
                magnetisation=[0, 0, 0.38/(4*np.pi*10**-7)])
        top_magnet = cudamag.Magnet(
        vertices=np.array([[-0.01+d, -0.014, 0.005],
                [0.002+d, -0.014, 0.005],
                [-0.01+d, 0.006, 0.005],
                [0.002+d, 0.006, 0.005],
                [-0.01+d, -0.014, 0.011],
                [0.002+d, -0.014, 0.011],
                [-0.01+d, 0.006, 0.011],
                [0.002+d, 0.006, 0.011]]),
                magnetisation=[0, 0, 0.38/(4*np.pi*10**-7)])

        # Increase mesh density for accuracy
        quantisation = 24
        bottom_magnet.subdivide(quantisation)
        top_magnet.subdivide(quantisation)

        # Set up and solve the system
        system.add_magnet(bottom_magnet)
        system.add_magnet(top_magnet)
        system.solve_system(data_type=np.float32)

        print(top_magnet.force)
        print(bottom_magnet.force)



if __name__ == "__main__":
    test_akoun_yonnet()