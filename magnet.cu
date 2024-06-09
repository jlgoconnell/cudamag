#include "magnet.h"

Magnet::~Magnet()
{
    free(this->pts);
    free(this->areas);
}

int Magnet::getNumPts()
{
    return this->numPts;
}

float* Magnet::getPts()
{
    return this->pts;
}

float* Magnet::getAreas()
{
    return this->areas;
}

float* Magnet::getSigma()
{
    return this->sigma;
}

Magnet::Magnet(int numVertices, float* vertices, float* areas, float* sigma)
{
    this->numPts = numVertices;
    this->pts = (float*)malloc(numVertices*3*sizeof(float));
    this->areas = (float*)malloc(numVertices*sizeof(float));
    this->sigma = (float*)malloc(numVertices*sizeof(float));
    
    // Write data to members
    for (int ii = 0; ii < 3*numVertices; ii++) this->pts[ii] = vertices[ii];
    for (int ii = 0; ii < numVertices; ii++) this->areas[ii] = areas[ii];
    for (int ii = 0; ii < numVertices; ii++) this->sigma[ii] = sigma[ii];
}
