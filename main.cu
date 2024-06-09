#include "cudamag.h"
#include "magnet.h"
#include <iostream>

int main()
{
    std::cout << "Hello world!\n";

    const int N = 10; // Square root of number of subdivisions of each face
    const int numElemsPerMagnet = N*N*6;
    const int numMagnets = 2;

    // Set up magnet parameter variables
    float h_pts1[numElemsPerMagnet*3];
    float h_pts2[numElemsPerMagnet*3];
    float h_areas1[numElemsPerMagnet];
    float h_areas2[numElemsPerMagnet];
    float h_mag1[3] = {0, 0, 1};
    float h_mag2[3] = {0, 0, 1};
    float h_sigma1[numElemsPerMagnet];
    float h_sigma2[numElemsPerMagnet];

    float divPts[N]; // Centroids of equal squares on 2-unit cube
    for (int ii = 0; ii < N; ii++) divPts[ii] = -1 + (2*ii+1)/float(N);

    // Populate points array
    for (int ii = 0; ii < N*N; ii++)
    {
        for (int jj = 0; jj < N*N; jj++)
        {
            // Top side
            h_pts1[ii+6*jj+0*N*N+0] = divPts[ii]; // x
            h_pts1[ii+6*jj+0*N*N+1] = divPts[jj]; // y
            h_pts1[ii+6*jj+0*N*N+2] = 1; // z
            h_areas1[ii+6*jj+0*N*N] = 4/(float)(N*N);
            h_sigma1[ii+6*jj+0*N*N] = 1;
            // Bottom side
            h_pts1[ii+6*jj+3*N*N+0] = divPts[ii];
            h_pts1[ii+6*jj+3*N*N+1] = divPts[jj];
            h_pts1[ii+6*jj+3*N*N+2] = -1;
            h_areas1[ii+6*jj+1*N*N] = 4/(float)(N*N);
            h_sigma1[ii+6*jj+1*N*N] = 0;
            // Left side
            h_pts1[ii+6*jj+6*N*N+0] = -1;
            h_pts1[ii+6*jj+6*N*N+1] = divPts[ii];
            h_pts1[ii+6*jj+6*N*N+2] = divPts[jj];
            h_areas1[ii+6*jj+2*N*N] = 4/(float)(N*N);
            h_sigma1[ii+6*jj+2*N*N] = 0;
            // Right side
            h_pts1[ii+6*jj+9*N*N+0] = 1;
            h_pts1[ii+6*jj+9*N*N+1] = divPts[ii];
            h_pts1[ii+6*jj+9*N*N+2] = divPts[jj];
            h_areas1[ii+6*jj+3*N*N] = 4/(float)(N*N);
            h_sigma1[ii+6*jj+3*N*N] = 0;
            // Back side
            h_pts1[ii+6*jj+12*N*N+0] = divPts[ii];
            h_pts1[ii+6*jj+12*N*N+1] = -1;
            h_pts1[ii+6*jj+12*N*N+2] = divPts[jj];
            h_areas1[ii+6*jj+4*N*N] = 4/(float)(N*N);
            h_sigma1[ii+6*jj+4*N*N] = 0;
            // Front side
            h_pts1[ii+6*jj+15*N*N+0] = divPts[ii];
            h_pts1[ii+6*jj+15*N*N+1] = 1;
            h_pts1[ii+6*jj+15*N*N+2] = divPts[jj];
            h_areas1[ii+6*jj+5*N*N] = 4/(float)(N*N);
            h_sigma1[ii+6*jj+5*N*N] = -1;
        }
    }

    // Add a second magnet
    for (int ii = 0; ii < 3*numElemsPerMagnet; ii++)
    {
        if (ii < numElemsPerMagnet) h_areas2[ii] = h_areas1[ii];
        if (ii < numElemsPerMagnet) h_sigma2[ii] = h_sigma1[ii];
        h_pts2[ii] = h_pts1[ii];
        if (ii > 2*numElemsPerMagnet) h_pts2[ii] += 2.0; // Move magnet up
    }


    Magnet botMag = Magnet(numElemsPerMagnet, h_pts1, h_areas1, h_sigma1);
    Magnet topMag = Magnet(numElemsPerMagnet, h_pts2, h_areas2, h_sigma2);


    CudaMag magSys;
    magSys.addMagnet(&botMag);
    magSys.addMagnet(&topMag);
    magSys.init();
    magSys.calcBmat();



    return 0;
}