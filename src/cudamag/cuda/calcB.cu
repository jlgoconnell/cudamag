extern "C" __global__
void calcB(float* centres, int numPts, float* B)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numPts)
    {
        float distCubed;
        float srcPt[3] = {centres[3 * index], centres[3 * index + 1], centres[3 * index + 2]};
        float relPos[3];

        for (int ii = 0; ii < numPts; ii++)
        {
            for (int jj = 0; jj < 3; jj++) relPos[jj] = centres[3 * ii + jj] - srcPt[jj];
            distCubed = pow(relPos[0] * relPos[0] + relPos[1] * relPos[1] + relPos[2] * relPos[2], 1.5f);
            if (distCubed > 0) for (int jj = 0; jj < 3; jj++) B[jj * numPts * numPts + index * numPts + ii] = relPos[jj] / distCubed;
        }
    }
}
