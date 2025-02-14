extern "C" __global__
void calcB(float* nodes, unsigned int* connections, float* normals, int numNodes, int numTriangles, float* B)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numTriangles)
    {
        float distCubed;
        float oneThird = 1.0/3.0;
        float srcPt[3] = {0, 0, 0};
        float pts[3][3];
        /*  
                [x1  y1  z1]
        pts  =  [x2  y2  z2]
                [x3  y3  z3]
        */
        // Find the points making up this triangle (pts), as well as its centre (srcPt)
        for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++) pts[ii][jj] = nodes[3 * connections[3 * index + ii] + jj];
        for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++) srcPt[jj] += oneThird * pts[ii][jj];

        // Import normals, becoming the z-axis after transformation
        float R[3][3];
        float z[3];
        for (int ii = 0; ii < 3; ii++) R[2][ii] = normals[3 * index + ii]; // Should already be unit vector

        // Calculate the y-axis
        float y[3];
        for (int ii = 0; ii < 3; ii++) R[1][ii] = pts[1][ii] - pts[0][ii];
        float yLength = sqrt(R[1][0] * R[1][0] + R[1][1] * R[1][1] + R[1][2] * R[1][2]);
        for (int ii = 0; ii < 3; ii++) R[1][ii] = R[1][ii] / yLength; // Make unit vector

        // Calculate the x-axis
        float x[3];
        R[0][0] = R[1][1] * R[2][2] - R[1][2] * R[2][1];
        R[0][1] = R[1][2] * R[2][0] - R[1][0] * R[2][2];
        R[0][2] = R[1][0] * R[2][1] - R[1][1] * R[2][0];

        // Transform the points of this triangle
        float transformedPts[3][3];
        for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++) transformedPts[ii][jj] = pts[ii][0] * R[0][jj] + pts[ii][1] * R[1][jj] + pts[ii][2] * R[2][jj];
        /*  
                         [x1  y1  z1]
        transformedPts = [x2  y2  z2]
                         [x3  y3  z3]
        */
        
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
