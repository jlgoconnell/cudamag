#include "cudamag.h"
#include <iostream>


__global__ void calcB(float* Bout, float* d_pts, int numPts)
{
    // threadIdx.x is the index of the point we're considering
    // blockIdx.x is the dimension (0 for x, 1 for y, 2 for z)
    for (int ii = 0; ii < numPts; ii++)
    {
        float distCubed = pow(pow(d_pts[3*(ii+threadIdx.x)]-d_pts[3*ii],2) + pow(d_pts[3*(ii+threadIdx.x)+1]-d_pts[3*ii+1],2) + pow(d_pts[3*(ii+threadIdx.x)+2]-d_pts[3*ii+2],2), -1.5);
        Bout[ii*numPts+threadIdx.x+numPts*numPts*blockIdx.x] = (d_pts[3*(ii+threadIdx.x)+blockIdx.x]-d_pts[3*ii+blockIdx.x]) * distCubed;
    }
}


CudaMag::CudaMag()
{
    numPts = 0;
    //numMagnets = 0;

    std::cout << "Magnet system created.\n";
}

CudaMag::~CudaMag()
{
    // Delete any memory allocated
}

// Set up memory, etc
void CudaMag::init()
{
    std::cout << "Initialising CudaMag.\n";
}
