#include "cudamag.h"

// Helper functions to allow Python to interface with the CudaMag class
extern "C"
{

void createMagnetSystem()
{
    CudaMag magSys;
}

}