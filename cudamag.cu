#include "cudamag.h"
#include <cublas_v2.h>
#include <iostream>
#include <chrono>

#define THREADS_PER_BLOCK 256  


__global__ void subdivideMesh(float* nodes, int* connectivity, int numConnections, int meshIncrease)
{
    int idx = threadIdx.x + THREADS_PER_BLOCK*blockIdx.x;

    if (idx < numConnections)
    {
        
    }
}


__global__ void calcAreas(float* areas, float* nodes, int* connectivity, int numConnections)
{
    int idx = threadIdx.x + THREADS_PER_BLOCK*blockIdx.x;

    if (idx < numConnections)
    {
        float ABx = nodes[3*connectivity[3*idx+1]] - nodes[3*connectivity[3*idx]];
        float ABy = nodes[3*connectivity[3*idx+1]+1] - nodes[3*connectivity[3*idx]+1];
        float ABz = nodes[3*connectivity[3*idx+1]+2] - nodes[3*connectivity[3*idx]+2];
        float ACx = nodes[3*connectivity[3*idx+2]] - nodes[3*connectivity[3*idx]];
        float ACy = nodes[3*connectivity[3*idx+2]+1] - nodes[3*connectivity[3*idx]+1];
        float ACz = nodes[3*connectivity[3*idx+2]+2] - nodes[3*connectivity[3*idx]+2];
        // Cross product
        float crossProd[3] = {ABy*ACz-ABz*ACy, ABz*ACx-ABx*ACz, ABx*ACy-ABy*ACx};
        // Area is half the length of the cross product
        areas[idx] = 0.5 * pow(pow(crossProd[0],2) + pow(crossProd[1],2) + pow(crossProd[2],2), 0.5);
    }
}


__global__ void calcB(float* Bout, float* nodes, int* connectivity, int numConnections)
{
    int idx = threadIdx.x + THREADS_PER_BLOCK*blockIdx.x;
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
                Bout[idx*numConnections + ii] = area * dx * invDistCubed * 1e-7;
                Bout[idx*numConnections + ii + numConnections*numConnections] = area * dy * invDistCubed * 1e-7;
                Bout[idx*numConnections + ii + 2*numConnections*numConnections] = area * dz * invDistCubed * 1e-7;
            } else {
                Bout[idx*numConnections + ii] = 0;
                Bout[idx*numConnections + ii + numConnections*numConnections] = 0;
                Bout[idx*numConnections + ii + 2*numConnections*numConnections] = 0;
            }

        }
    }
}



void printMat(float* d_mat, int m, int n, int o)
{
    float* h_mat = (float*)malloc(m*n*o*sizeof(float));
    cudaMemcpy(h_mat, d_mat, m*n*o*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int kk = 0; kk < o; kk++)
    {
        for (int ii = 0; ii < m; ii++)
        {
            for (int jj = 0; jj < n; jj++)
            {
                printf("%1.5f,\t", h_mat[ii + jj*m + kk*m*n]);
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    free(h_mat);
}


CudaMag::CudaMag()
{
    cudaStreamCreate(&stream);
    cublasCreate(&cublasHandle);
    cublasSetStream(cublasHandle, stream);
    std::cout << "Magnet system created.\n";
}

CudaMag::~CudaMag()
{
    // Delete any memory allocated
    std::cout << "Freeing memory.\n";
    free(this->magIdx);
    free(this->h_connectivity);
    free(this->h_nodes);
    free(this->h_sigma);
    cudaFree(d_connectivity);
    cudaFree(d_nodes);
    cudaFree(d_sigma);
    cudaFree(d_sigmaSegmented);
    cudaFree(d_areas);
    cudaFree(d_B);
    cudaFree(d_tempMat);
    cudaFree(d_tempMat2);
    cudaFree(d_forces);
    cublasDestroy(cublasHandle);
    cudaStreamDestroy(stream);
    std::cout << "Memory freed.\n";
}

// Set up memory, etc
void CudaMag::init(float* nodes, int numNodes, int* connectivity, int numConnections, float* sigma, int* magnetIdx, int numMags)
{
    std::cout << "Initialising CudaMag.\n";
    this->numNodes = numNodes;
    this->numConnections = numConnections;
    this->numMags = numMags;
    this->magIdx = (int*)malloc(numMags*sizeof(int));
    memcpy(this->magIdx, magnetIdx, numMags*sizeof(int));

    this->h_connectivity = (int*)malloc(3*numConnections*sizeof(int));
    memcpy(h_connectivity, connectivity, 3*numConnections*sizeof(int));
    this->h_nodes = (float*)malloc(3*numNodes*sizeof(float));
    memcpy(h_nodes, nodes, 3*numNodes*sizeof(float));
    this->h_sigma = (float*)malloc(numNodes*sizeof(float));
    memcpy(h_sigma, sigma, numNodes*sizeof(float));

    std::cout << "Memory allocated.\n";
}


void CudaMag::transferToGpu()
{
    // Allocate memory
    cudaMalloc(&d_connectivity, 3*this->numConnections*sizeof(int));
    cudaMalloc(&d_nodes, 3*this->numNodes*sizeof(float));
    cudaMalloc(&d_sigma, this->numNodes*sizeof(float));
    cudaMalloc(&d_areas, this->numConnections*sizeof(float));
    cudaMalloc(&d_B, this->numConnections*this->numConnections*3*sizeof(float));
    cudaMalloc(&d_tempMat, this->numConnections*this->numConnections*3*sizeof(float));
    cudaMalloc(&d_tempMat2, this->numConnections*numMags*3*sizeof(float));
    cudaMalloc(&d_forces, this->numMags*this->numMags*3*sizeof(float));

    // Copy necessary data
    cudaMemcpy(d_connectivity, this->h_connectivity, 3*numConnections*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes, this->h_nodes, 3*this->numNodes*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Calculate the area of each triangular element
    calcAreas<<<floor(this->numConnections / THREADS_PER_BLOCK)+1, THREADS_PER_BLOCK>>>(d_areas, d_nodes, d_connectivity, this->numConnections);

    // Copy the surface charges
    cudaMemcpy(d_sigma, this->h_sigma, this->numConnections*sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    // printf("d_sigmaSegmented is:\n");
    // printMat(d_sigmaSegmented, this->numConnections, this->numMags, 1);
}


// Solve for the B matrices
void CudaMag::calcBmat()
{
    std::cout << "About to run calcB.\n";

    // Calculate big ol' matrix
    std::cout << "Calling calcB with " << this->numConnections << " threads.\n";
    calcB<<<floor(this->numConnections / THREADS_PER_BLOCK)+1, THREADS_PER_BLOCK>>>(d_B, this->d_nodes, this->d_connectivity, this->numConnections);
    //cudaDeviceSynchronize();

    printf("Num elements: %i.\n", this->numConnections);
}


// Solve the system
void CudaMag::solve()
{
    auto start = std::chrono::high_resolution_clock::now();

    cudaMalloc(&d_sigmaSegmented, this->numConnections*numMags*sizeof(float));
    cudaMemset(d_sigmaSegmented, 0, this->numConnections*numMags*sizeof(float));
    for (int ii = 0; ii < numMags-1; ii++) cudaMemcpyAsync(d_sigmaSegmented + this->magIdx[ii] + ii*this->numConnections, this->h_sigma + this->magIdx[ii], (this->magIdx[ii+1]-this->magIdx[ii]) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_sigmaSegmented + this->magIdx[numMags-1] + (numMags-1)*this->numConnections, this->h_sigma + this->magIdx[numMags-1], (this->numConnections-this->magIdx[numMags-1]) * sizeof(float), cudaMemcpyHostToDevice);

    // To compute the forces, do a few matrix multiplications
    for (int ii = 0; ii < 3; ii++)
    {
        cublasSdgmm(cublasHandle,
        CUBLAS_SIDE_LEFT,
        this->numConnections,
        this->numConnections,
        d_B + ii*this->numConnections*this->numConnections,
        this->numConnections,
        d_areas,
        1,
        d_tempMat + ii*this->numConnections*this->numConnections,
        this->numConnections);
        //cudaDeviceSynchronize();
    }
    //cudaDeviceSynchronize();
    //printf("d_tempMat is:\n");
    //printMat(d_tempMat, this->numConnections, this->numConnections, 3);

    // Premultiply sigmaSegmented
    const float alpha = 1.0;
    const float beta = 0.0;
    cublasSgemmStridedBatched(cublasHandle,
    CUBLAS_OP_T,
    CUBLAS_OP_T,
    this->numMags,
    this->numConnections,
    this->numConnections,
    &alpha,
    this->d_sigmaSegmented,
    this->numConnections,
    0,
    this->d_tempMat,
    this->numConnections,
    this->numConnections*this->numConnections,
    &beta,
    this->d_tempMat2,
    this->numMags,
    this->numConnections*this->numMags,
    3);
    //cudaDeviceSynchronize();
    // printf("d_tempMat2 is:\n");
    // printMat(d_tempMat2, this->numMags, this->numConnections, 3);

    // Postmultiply sigmaSegmented
    cublasSgemmStridedBatched(cublasHandle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    this->numMags,
    this->numMags,
    this->numConnections,
    &alpha,
    this->d_tempMat2,
    this->numMags,
    this->numMags*this->numConnections,
    this->d_sigmaSegmented,
    this->numConnections,
    0,
    &beta,
    this->d_forces,
    this->numMags,
    this->numMags*this->numMags,
    3);
    //cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    printf("Forces computed in %f microseconds:\n", 1e6*elapsed.count());
    printMat(d_forces, this->numMags, this->numMags, 3);

}

