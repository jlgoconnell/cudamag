#ifndef MAGNET_H
#define MAGNET_H

class Magnet
{
    public:
        Magnet(int numVertices, float* vertices, float* areas, float* sigma);
        ~Magnet();

        int getNumPts();
        float* getPts();
        float* getAreas();
        float* getSigma();

    private:
        int numPts;
        float* pts;
        float* areas;
        float* sigma;
};
#endif