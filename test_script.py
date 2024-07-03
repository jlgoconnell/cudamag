from cudamag import CudaMag, Magnet

test = CudaMag()

mag = Magnet([[0, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [1, 1, 0],
              [0, 0, 1],
              [1, 0, 1],
              [0, 1, 1],
              [1, 1, 1]],
              [0, 0, 1],
              1.0)
mag2 = Magnet([[0, 0, 2],
              [1, 0, 2],
              [0, 1, 2],
              [1, 1, 2],
              [0, 0, 3],
              [1, 0, 3],
              [0, 1, 3],
              [1, 1, 3]],
              [0, 0, 1],
              1.0)
test.add_magnet(mag)
test.add_magnet(mag2)
test.initialise()
test.solve_system()