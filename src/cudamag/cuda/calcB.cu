extern "C" __global__
void calcB(float* centres, int numPts, float* B)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numPts)
    {
        const float coeff = 1e-7;
        float distCubed;
        float srcPt[3] = {centres[index], centres[index + numPts], centres[index + 2 * numPts]};
        float relPos[3];

        for (int ii = 0; ii < numPts; ii++)
        {
            for (int jj = 0; jj < 3; jj++) relPos[jj] = centres[ii + jj * numPts] - srcPt[jj];
            distCubed = pow(relPos[0] * relPos[0] + relPos[1] * relPos[1] + relPos[2] * relPos[2], 1.5f);
            if (distCubed > 0) for (int jj = 0; jj < 3; jj++) B[jj * numPts * numPts + index * numPts + ii] = coeff * relPos[jj] / distCubed;
        }
    }
}

int main()
{
    return 0;
}