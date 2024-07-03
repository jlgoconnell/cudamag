#include "cudamag.h"
#include <iostream>


__global__ void calcB(float* Bout, float* nodes, int* connectivity)
{
    // Test code to print the centroids of all surface elements
    int theseConnections[3];
    for (int ii = 0; ii < 3; ii++) theseConnections[ii] = connectivity[3*threadIdx.x + ii];

    float pts[3];
    for (int ii = 0; ii < 3; ii++)
    {
        pts[ii] = 1.0/3.0 * (nodes[3*theseConnections[0]+ii] + nodes[3*theseConnections[1]+ii] + nodes[3*theseConnections[2]+ii]);
    }

    printf("Thread %i: Averaging nodes %i, %i, and %i gives a centroid of [%f, %f, %f].\n", threadIdx.x, theseConnections[0], theseConnections[1], theseConnections[2], pts[0], pts[1], pts[2]);

    // threadIdx.x is the index of the point we're considering
    // blockIdx.x is the dimension (0 for x, 1 for y, 2 for z)
    /*for (int ii = 0; ii < numPts; ii++)
    {
        float distCubed = pow(pow(d_pts[3*(ii+threadIdx.x)]-d_pts[3*ii],2) + pow(d_pts[3*(ii+threadIdx.x)+1]-d_pts[3*ii+1],2) + pow(d_pts[3*(ii+threadIdx.x)+2]-d_pts[3*ii+2],2), -1.5);
        Bout[ii*numPts+threadIdx.x+numPts*numPts*blockIdx.x] = (d_pts[3*(ii+threadIdx.x)+blockIdx.x]-d_pts[3*ii+blockIdx.x]) * distCubed;
    }*/
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
    this->numNodes = numNodes;
    this->numConnections = numConnections;
    cudaMalloc(&d_connectivity, this->numConnections*sizeof(int));
    cudaMalloc(&d_nodes, this->numNodes*sizeof(float));
    cudaMalloc(&d_sigma, this->numNodes*sizeof(float));
    cudaMemcpy(d_connectivity, connectivity, numConnections*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes, nodes, this->numNodes*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma, sigma, this->numNodes/3.0*sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "Memory allocated.\n";
}


// Solve for the B matrices
void CudaMag::calcBmat()
{
    std::cout << "About to run calcB.\n";
    calcB<<<1, this->numConnections/3>>>(nullptr, d_nodes, d_connectivity);
    cudaDeviceSynchronize();
}


// Solve the system
void CudaMag::solve()
{
    //this->calcBmat();
}