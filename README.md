# CudaMag: Fast and accurate computation of forces between magnets using the NVIDIA CUDA framework

This project is an extension of my PhD, where I derived expressions and methodologies for computing magnetic fields, forces, and torques due to permanent magnets. This repository contains code implementing those algorithms using the NVIDIA CUDA framework to compute data in parallel. As such it requires an NVIDIA CUDA-capable GPU to rune. This is a work in progress and will be updated as I find more time.

## Why CUDA?

The algorithms I developed rely on surface meshes, with more surface elements leading to more accuracy. The methodologies are based on many simultaneous calculations on each element, and uses a considerable amount of matrix algebra. Thus, a GPU framework should in theory lead to extremely fast computations.

## Assumptions and limitations

While this framework will allow fast and accurate computation of magnetic variables, it relies on several assumptions, limiting its use in more generalised cases. These assumptions include (but are not limited to):
* The relative permeability may be non-unity, but must be spatially and temporally constant. This limits the use of highly non-linear materials such as iron; however, for such soft materials, it is possible that an exceptionally large permeability may suffice.
* Upon applying a surface mesh, it is assumed that the "magnetic charge density" across each element is constant. As mesh density increases, this assumption becomes more accurate, but for coarse meshes this must be considered.
* The volumetric "magnetic charge density" is assumed to be zero. For most cases using permanent magnets, the volumetric charge density is negligible, but should be considered for magnetically soft materials.

## Roadmap

The development of this project is a continual work in progress and will take time to realise. The current plan is outlined below but is not necessarily in order.

- [ ] Theory documentation
    - [ ] Matrix-based physical description of permanent magnet systems
    - [ ] Point-monopole fields and forces
    - [ ] Geometry-based fields and forces
    - [ ] Non-unity relative permeability derivations
- [ ] Basic point-monopole force computation with unity relative permeability
    - [X] Basic force computation with minimum meshing
    - [ ] Mesh manipulation function
    - [ ] More accurate force computation with better meshing
- [ ] Geometry-based force computation with unity relative permeability
- [ ] Geometry-based force computation with non-unity relative permeability
- [ ] Python interfacing with the CUDA C++ code allowing simple scripting and data postprocessing
    - [X] C interface between Python and CUDA C++
    - [X] Python driver script
    - [ ] Python driver script additional features
    - [ ] Python postprocessing features
- [ ] Visualisations of magnetic interactions