#ifndef CUDAMAG_H
#define CUDAMAG_H

#include<vector>
#include<iostream>



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
        void init(float* nodes, int numNodes, int* connectivity, int numConnections, float* sigma);
        void calcBmat();
        void solve();

    private:
        std::vector<Magnet*> magnets;

        int* d_connectivity;
        float* d_nodes;

        int numNodes;
        int numConnections;

        float* d_B;
        float* d_sigma;

};

#endif