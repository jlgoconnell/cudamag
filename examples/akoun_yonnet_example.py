import cudamag
import numpy as np

system = cudamag.CudaMag()

# Set up geometry
d = 0.0
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
quantisation = 4
bottom_magnet.subdivide(quantisation)
top_magnet.subdivide(quantisation)

# Set up and solve the system
system.add_magnet(bottom_magnet)
system.add_magnet(top_magnet)
system.solve_system()

print("The top magnet has a force of [", top_magnet._force[0], ", ", top_magnet._force[1], ", ", top_magnet._force[2], "].")
print("The bottom magnet has a force of [", bottom_magnet._force[0], ", ", bottom_magnet._force[1], ", ", bottom_magnet._force[2], "].")