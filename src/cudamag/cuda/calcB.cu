extern "C" __global__
void calcB(float* nodes, unsigned int* connections, int numNodes, int numTriangles, float* B)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numTriangles)
    {
        float distCubed;
        float oneThird = 1.0/3.0;
        float srcPt[3];
        for (int ii = 0; ii < 3; ii++) srcPt[ii] = oneThird * (nodes[3 * connections[3 * index] + ii] + nodes[3 * connections[3 * index + 1] + ii] + nodes[3 * connections[3 * index + 2] + ii]);
        float relPos[3]; // The relative position of each point with respect to the source point

        for (int ii = 0; ii < numTriangles; ii++)
        {
            for (int jj = 0; jj < 3; jj++) relPos[jj] = oneThird * (nodes[3 * connections[3 * ii] + jj] + nodes[3 * connections[3 * ii + 1] + jj] + nodes[3 * connections[3 * ii + 2] + jj]) - srcPt[jj];
            distCubed = pow(relPos[0] * relPos[0] + relPos[1] * relPos[1] + relPos[2] * relPos[2], 1.5f);
            if (ii != index) for (int jj = 0; jj < 3; jj++) B[jj * numTriangles * numTriangles + index * numTriangles + ii] = relPos[jj] / distCubed;
        }
    }
}
