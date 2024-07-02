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
        void init(float* nodes, int numNodes, int* connectivity, int numConnections, float* sigma);
        void calcBmat();
        void solve();

    private:
        struct Magnet
        {
            int numVertices;
        };
        std::vector<Magnet*> magnets;

        int* d_connectivity;
        float* d_nodes;

        float* d_B;
        float* d_sigma;

};

#endif