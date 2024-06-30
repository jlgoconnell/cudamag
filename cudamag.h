#ifndef CUDAMAG_H
#define CUDAMAG_H

#include<vector>
#include<iostream>
#include "magnet.h"



class CudaMag
{
    public:
        CudaMag();
        ~CudaMag();

        void addMagnet(Magnet* magnet);
        void init();
        void calcBmat();
        void solve();

    private:
        struct Magnet
        {
            int numVertices;
        };
        std::vector<Magnet*> magnets;
        int* testInt;
        float* d_pts;
        float* d_areas;
        int numPts;
        //int numMagnets;
        float* d_B;
        float* d_sigma;

};

#endif