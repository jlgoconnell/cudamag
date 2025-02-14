extern "C" __global__
void calcB(float* nodes, unsigned int* connections, int numNodes, int numTriangles, float* B)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numTriangles)
    {
        float distCubed;
        float oneThird = 1.0/3.0;
        float srcPt[3] = {0, 0, 0};
        float pts[3][3];
        /*   
                [x1  x2  x3]
        pts  =  [y1  y2  y3]
                [z1  z2  z3]
        */

        // Find the points making up this triangle (pts), as well as its centre (srcPt)
        for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++) pts[ii][jj] = nodes[3 * connections[3 * index + ii] + jj];
        for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++) srcPt[jj] += oneThird * pts[ii][jj];
        
        float relPos[3]; // The relative position of each point with respect to the source point

        float otherCentre[3];
        for (int ii = 0; ii < numTriangles; ii++)
        {
            // Compute relative position of each triangle centre from the query triangle
            for (int jj = 0; jj < 3; jj++)
            {
                otherCentre[jj] = oneThird * (nodes[3 * connections[3 * ii] + jj] + nodes[3 * connections[3 * ii + 1] + jj] + nodes[3 * connections[3 * ii + 2] + jj]);
                relPos[jj] = otherCentre[jj] - srcPt[jj];
            }
            // Find the cubed distance
            distCubed = pow(relPos[0] * relPos[0] + relPos[1] * relPos[1] + relPos[2] * relPos[2], 1.5f);
            // Populate the B matrix
            if (ii != index) for (int jj = 0; jj < 3; jj++) B[jj * numTriangles * numTriangles + index * numTriangles + ii] = relPos[jj] / distCubed;
        }
    }
}
