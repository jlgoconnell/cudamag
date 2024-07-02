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
    //numMagnets = 0;

    std::cout << "Magnet system created.\n";
}

CudaMag::~CudaMag()
{
    // Delete any memory allocated
    std::cout << "Freeing memory.\n";
    cudaFree(d_connectivity);
    cudaFree(d_nodes);
    cudaFree(d_sigma);
    std::cout << "Memory freed.\n";
}

// Set up memory, etc
void CudaMag::init(float* nodes, int numNodes, int* connectivity, int numConnections, float* sigma)
{
    std::cout << "Initialising CudaMag.\n";
    cudaMemcpy(d_connectivity, connectivity, numConnections*3*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes, nodes, numNodes*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma, sigma, numNodes*sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "Memory allocated.\n";
}


// Solve for the B matrices
void CudaMag::calcBmat()
{

}


// Solve the system
void CudaMag::solve()
{

}