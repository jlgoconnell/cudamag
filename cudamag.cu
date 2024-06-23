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
    //cudaFree(d_pts);
    //cudaFree(d_areas);
    //cudaFree(d_B);
}
/*
void CudaMag::addMagnet(Magnet* magnet)
{
    magnets.push_back(magnet);
}

void CudaMag::init()
{
    // Calculate total points
    for (int ii = 0; ii < magnets.size(); ii++) numPts += magnets[ii]->getNumPts();

    // Allocate memory
    cudaMalloc(&d_pts, numPts*3*sizeof(float));
    cudaMalloc(&d_areas, numPts*magnets.size()*sizeof(float));
    cudaMemset(d_areas, 0, numPts*magnets.size()*sizeof(float));
    cudaMalloc(&d_B, numPts*numPts*3*sizeof(float));
    cudaMalloc(&d_sigma, numPts*sizeof(float));

    // Transfer data to GPU
    int ctr = 0;
    for (int ii = 0; ii < magnets.size(); ii++)
    {
        cudaMemcpyAsync(d_pts+ctr, magnets[ii]->getPts(), magnets[ii]->getNumPts()*3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_areas+ctr+numPts, magnets[ii]->getAreas(), magnets[ii]->getNumPts()*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_sigma+ctr, magnets[ii]->getSigma(), magnets[ii]->getNumPts()*sizeof(float), cudaMemcpyHostToDevice);
        ctr += magnets[ii]->getNumPts();
    }
}


void CudaMag::calcBmat()
{
    calcB<<<3, numPts>>>(d_B, d_pts, numPts);
}

void CudaMag::solve()
{

}

*/