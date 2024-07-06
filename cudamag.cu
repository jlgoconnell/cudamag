#include "cudamag.h"
#include <iostream>


__global__ void calcB(float* Bout, float* nodes, int* connectivity, int numConnections)
{
    int idx = threadIdx.x; // Fix later
    float oneThird = 1.0/3.0;
    if (idx < numConnections)
    {
        // Compute the centre of this element
        float thisX = oneThird * (nodes[3*connectivity[3*idx]] + nodes[3*connectivity[3*idx+1]] + nodes[3*connectivity[3*idx+2]]);
        float thisY = oneThird * (nodes[3*connectivity[3*idx]+1] + nodes[3*connectivity[3*idx+1]+1] + nodes[3*connectivity[3*idx+2]+1]);
        float thisZ = oneThird * (nodes[3*connectivity[3*idx]+2] + nodes[3*connectivity[3*idx+1]+2] + nodes[3*connectivity[3*idx+2]+2]);

        // Compute the area of this element
        float ABx = nodes[3*connectivity[3*idx+1]] - nodes[3*connectivity[3*idx]];
        float ABy = nodes[3*connectivity[3*idx+1]+1] - nodes[3*connectivity[3*idx]+1];
        float ABz = nodes[3*connectivity[3*idx+1]+2] - nodes[3*connectivity[3*idx]+2];
        float ACx = nodes[3*connectivity[3*idx+2]] - nodes[3*connectivity[3*idx]];
        float ACy = nodes[3*connectivity[3*idx+2]+1] - nodes[3*connectivity[3*idx]+1];
        float ACz = nodes[3*connectivity[3*idx+2]+2] - nodes[3*connectivity[3*idx]+2];
        // Cross product
        float crossProd[3] = {ABy*ACz-ABz*ACy, ABz*ACx-ABx*ACz, ABx*ACy-ABy*ACx};
        // Area is half the length of the cross product
        float area = 0.5 * pow(pow(crossProd[0],2) + pow(crossProd[1],2) + pow(crossProd[2],2), 0.5);

        // Iterate through all other elements
        for (int ii = 0; ii < numConnections; ii++)
        {
            if (ii != idx)
            {
                float thatX = oneThird * (nodes[3*connectivity[3*ii]] + nodes[3*connectivity[3*ii+1]] + nodes[3*connectivity[3*ii+2]]);
                float thatY = oneThird * (nodes[3*connectivity[3*ii]+1] + nodes[3*connectivity[3*ii+1]+1] + nodes[3*connectivity[3*ii+2]+1]);
                float thatZ = oneThird * (nodes[3*connectivity[3*ii]+2] + nodes[3*connectivity[3*ii+1]+2] + nodes[3*connectivity[3*ii+2]+2]);

                float dx = thisX - thatX;
                float dy = thisY - thatY;
                float dz = thisZ - thatZ;
                float invDistCubed = pow(pow(dx,2) + pow(dy,2) + pow(dz,2), -1.5);

                // Insert value into matrix
                Bout[idx*numConnections + ii] = area * dx * invDistCubed;
                Bout[idx*numConnections + ii + numConnections*numConnections] = area * dy * invDistCubed;
                Bout[idx*numConnections + ii + 2*numConnections*numConnections] = area * dz * invDistCubed;
            } else {
                Bout[idx*numConnections + ii] = 0;
                Bout[idx*numConnections + ii + numConnections*numConnections] = 0;
                Bout[idx*numConnections + ii + 2*numConnections*numConnections] = 0;
            }

        }
    }
}


CudaMag::CudaMag()
{
    std::cout << "Magnet system created.\n";
}

CudaMag::~CudaMag()
{
    // Delete any memory allocated
    std::cout << "Freeing memory.\n";
    cudaFree(d_connectivity);
    cudaFree(d_nodes);
    cudaFree(d_sigma);
    cudaFree(d_B);
    std::cout << "Memory freed.\n";
}

// Set up memory, etc
void CudaMag::init(float* nodes, int numNodes, int* connectivity, int numConnections, float* sigma)
{
    std::cout << "Initialising CudaMag.\n";
    this->numNodes = numNodes;
    this->numConnections = numConnections;
    cudaMalloc(&d_connectivity, 3*this->numConnections*sizeof(int));
    cudaMalloc(&d_nodes, 3*this->numNodes*sizeof(float));
    cudaMalloc(&d_sigma, this->numNodes*sizeof(float));
    cudaMalloc(&d_B, this->numConnections*this->numConnections*3*sizeof(float));
    cudaMemcpy(d_connectivity, connectivity, 3*numConnections*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes, nodes, 3*this->numNodes*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma, sigma, this->numNodes*sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "Memory allocated.\n";
}


// Solve for the B matrices
void CudaMag::calcBmat()
{
    std::cout << "About to run calcB.\n";
    calcB<<<1, this->numConnections>>>(d_B, this->d_nodes, this->d_connectivity, this->numConnections);
    cudaDeviceSynchronize();
    float* h_B = (float*)malloc(this->numConnections*this->numConnections*3*sizeof(float));
    cudaMemcpy(h_B, d_B, this->numConnections*this->numConnections*3*sizeof(float), cudaMemcpyDeviceToHost);
    for (int kk = 0; kk < 3; kk++)
    {
        for (int jj = 0; jj < this->numConnections; jj++)
        {
            for (int ii = 0; ii < this->numConnections; ii++)
            {

            }
        }
    }
    free(h_B);
    printf("Num elements: %i.\n", this->numConnections);
}


// Solve the system
void CudaMag::solve()
{
    //this->calcBmat();
}

