#ifndef CUDAMAG_H
#define CUDAMAG_H

#include <vector>
#include <cublas_v2.h>
#include <iostream>



class CudaMag
{
    public:
        CudaMag();
        ~CudaMag();

        struct Magnet
        {
            int numVertices;
        };
        void addMagnet(Magnet* magnet);
        void init(float* nodes, int numNodes, int* connectivity, int numConnections, float* sigma, int* magnetIdx, int numMags);
        void calcBmat();
        void solve();

        cublasHandle_t cublasHandle;
        cudaStream_t stream;

    private:
        std::vector<Magnet*> magnets;

        int* d_connectivity;
        float* d_nodes;
        float* d_areas;

        int numNodes;
        int numConnections;
        int numMags;

        float* d_B;
        float* d_sigma;
        float* d_sigmaSegmented;
        float* d_forces;
        float* d_tempMat;
        float* d_tempMat2;

};

#endif

// Helper functions to allow Python to interface with the CudaMag class
extern "C"
{

// Create the CudaMag object for use in this interface
CudaMag* magSys = new CudaMag;


// Return a handle to the CudaMag object
void* getMagnetSystem()
{
    return reinterpret_cast<void*>(magSys);
}

// Destroy the CudaMag object
void destroyMagnetSystem()
{
    delete magSys;
}

// Initialise data structures and memory
void init(float* nodes, int numNodes, int* connectivity, int numConnections, float* sigma, int* magnetIdx, int numMags)
{
    magSys->init(nodes, numNodes, connectivity, numConnections, sigma, magnetIdx, numMags);
}


// Solve the system
void solve()
{
    std::cout << "Solving magnet system.\n";
    magSys->calcBmat();
    magSys->solve();
}

}