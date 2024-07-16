from cudamag import CudaMag, Magnet
import numpy as np

test = CudaMag()

# Akoun and Yonnet
d = 0.016
mag1 = Magnet([[-0.01, -0.006, -0.003],
              [0.01, -0.006, -0.003],
              [-0.01, 0.006, -0.003],
              [0.01, 0.006, -0.003],
              [-0.01, -0.006, 0.003],
              [0.01, -0.006, 0.003],
              [-0.01, 0.006, 0.003],
              [0.01, 0.006, 0.003]],
              [0, 0, 0.38/(4*np.pi*10**-7)],
              1.0)
mag2 = Magnet([[-0.01+d, -0.014, 0.005],
               [0.002+d, -0.014, 0.005],
               [-0.01+d, 0.006, 0.005],
               [0.002+d, 0.006, 0.005],
               [-0.01+d, -0.014, 0.011],
               [0.002+d, -0.014, 0.011],
               [-0.01+d, 0.006, 0.011],
               [0.002+d, 0.006, 0.011]],
               [0, 0, 0.38/(4*np.pi*10**-7)],
               1.0)

# Tetrahedron test case
# mag1 = Magnet([[0,      0,      0.03],
#                [0,      0.03,   0.03],
#                [0.03,   0,      0.03],
#                [0,      0,      0.06]],
#                [0, 0, 1], 1.0)
# mag2 = Magnet([[0,      0,     -0.03],
#                [0,      0.03,  -0.03],
#                [0.03,   0,     -0.03],
#                [0,      0,     -0.06]],
#                [0, 0, -1], 1.0)

# Cuboid case
# l = 1
# d = 0.2*l
# mag1 = Magnet([[0,  0,  d/2],
#                [l,  0,  d/2],
#                [0,  l,  d/2],
#                [l,  l,  d/2],
#                [0,  0,  d/2+l],
#                [l,  0,  d/2+l],
#                [0,  l,  d/2+l],
#                [l,  l,  d/2+l]],
#                [0, 0, 1/(4*np.pi*10**-7)], 1.0)
# mag2 = Magnet([[0,  0,  -d/2],
#                [l,  0,  -d/2],
#                [0,  l,  -d/2],
#                [l,  l,  -d/2],
#                [0,  0,  -d/2-l],
#                [l,  0,  -d/2-l],
#                [0,  l,  -d/2-l],
#                [l,  l,  -d/2-l]],
#                [0, 0, 1], 1)

for _ in range(2):
    mag1.quadruple_mesh()
    mag2.quadruple_mesh()

test.add_magnet(mag1)
test.add_magnet(mag2)
test.initialise()
test.solve_system()