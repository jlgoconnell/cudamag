#include "cudamag.h"
#include <iostream>

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
void init(float* nodes, int numNodes, int* connectivity, int numConnections, float* sigma)
{
    magSys->init(nodes, numNodes, connectivity, numConnections, sigma);
}


// Solve the system
void solve()
{
    std::cout << "Solving magnet system.\n";
    magSys->calcBmat();
    magSys->solve();
}

}