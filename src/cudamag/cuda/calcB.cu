extern "C" __global__
void calcB(float* nodes, unsigned int* connections, float* normals, int numNodes, int numTriangles, float* B)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numTriangles)
    {

//----------------------------------------------------------------------------
//      Import data
//----------------------------------------------------------------------------
        float distCubed;
        float oneThird = 1.0/3.0;
        float pts[3][3];
        /*  
                [x1  y1  z1]
        pts  =  [x2  y2  z2]
                [x3  y3  z3]
        */
        // Find the points making up this triangle (pts)
        for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++) pts[ii][jj] = nodes[3 * connections[3 * index + ii] + jj];


//----------------------------------------------------------------------------
//      Compute rotation matrix
//----------------------------------------------------------------------------
        float R[3][3];

        // Calculate the z-axis
        for (int ii = 0; ii < 3; ii++) R[2][ii] = normals[3 * index + ii]; // Should already be unit vector

        // Calculate the y-axis
        for (int ii = 0; ii < 3; ii++) R[1][ii] = pts[1][ii] - pts[0][ii];
        float yLength = sqrt(R[1][0] * R[1][0] + R[1][1] * R[1][1] + R[1][2] * R[1][2]);
        for (int ii = 0; ii < 3; ii++) R[1][ii] = R[1][ii] / yLength; // Make unit vector

        // Calculate the x-axis
        R[0][0] = R[1][1] * R[2][2] - R[1][2] * R[2][1];
        R[0][1] = R[1][2] * R[2][0] - R[1][0] * R[2][2];
        R[0][2] = R[1][0] * R[2][1] - R[1][1] * R[2][0];


//----------------------------------------------------------------------------
//      Find points in the transformed local coordinate system
//----------------------------------------------------------------------------
        float transformedPts[3][3]; // The three vertices
        for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++) transformedPts[ii][jj] = pts[ii][0] * R[0][jj] + pts[ii][1] * R[1][jj] + pts[ii][2] * R[2][jj];
        float transformedSrcPt[3] = {0, 0, 0}; // The centre of the triangle
        for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++) transformedSrcPt[jj] += oneThird * transformedPts[ii][jj];
        /*  
                         [x1  y1  z1]
        transformedPts = [x2  y2  z2]
                         [x3  y3  z3]
        */
        
        float relPos[3]; // The relative position of each point with respect to the source point


//----------------------------------------------------------------------------
//      Iterate through all other triangles and compute B field
//----------------------------------------------------------------------------
        float otherCentre[3];
        float transformedOtherCentre[3];
        float localB[3];
        for (int ii = 0; ii < numTriangles; ii++)
        {
            // Compute the centre of the triangle
            for (int jj = 0; jj < 3; jj++) otherCentre[jj] = oneThird * (nodes[3 * connections[3 * ii] + jj] + nodes[3 * connections[3 * ii + 1] + jj] + nodes[3 * connections[3 * ii + 2] + jj]);


            // Transform centre coordinates into local coordinate system
            for (int jj = 0; jj < 3; jj++) transformedOtherCentre[jj] = otherCentre[0] * R[0][jj] + otherCentre[1] * R[1][jj] + otherCentre[2] * R[2][jj];

            
            // Calculate the relative position in the local coordinate system
            for (int jj = 0; jj < 3; jj++) relPos[jj] = transformedOtherCentre[jj] - transformedSrcPt[jj];
            // Find the cubed distance
            distCubed = pow(relPos[0] * relPos[0] + relPos[1] * relPos[1] + relPos[2] * relPos[2], 1.5f);
            // Compute the local field
            if (ii != index) for (int jj = 0; jj < 3; jj++) localB[jj] = relPos[jj] / distCubed;


            // Transform back into global coordinates and populate B matrix
            if (ii != index)
            {
                for (int jj = 0; jj < 3; jj++) B[jj * numTriangles * numTriangles + index * numTriangles + ii] = localB[0] * R[jj][0] + localB[1] * R[jj][1] + localB[2] * R[jj][2];
            } else {
                for (int jj = 0; jj < 3; jj++) B[jj * numTriangles * numTriangles + index * numTriangles + ii] = 0.0;
            }
        }
    }
}
