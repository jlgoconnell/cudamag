// Main kernel, templated for single or double precision
template<typename T>
__global__ void calcB(T* nodes, unsigned int* connections, T* normals, unsigned int numNodes, unsigned int numTriangles, T* B)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numTriangles)
    {
//----------------------------------------------------------------------------
//      Import data
//----------------------------------------------------------------------------
        T oneThird = 1.0/3.0;
        T pts[3][3];
        /*  
                [x1  y1  z1]
        pts  =  [x2  y2  z2]
                [x3  y3  z3]
        */
        // Find the vertices (pts) making up the "base" triangle
        for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++) pts[ii][jj] = nodes[3 * connections[3 * index + ii] + jj];

        // Check ordering of points
        T vecA[3], vecB[3];
        for (int ii = 0; ii < 3; ii++)
        {
            vecA[ii] = pts[1][ii] - pts[0][ii];
            vecB[ii] = pts[2][ii] - pts[0][ii];
        }
        // Compute scalar triple product
        T tripleProduct = normals[3 * index] * (vecA[1]*vecB[2]-vecB[1]*vecA[2]) + normals[3 * index + 1] * (vecA[2]*vecB[0]-vecB[2]*vecA[0]) + normals[3 * index + 2] * (vecA[0]*vecB[1]-vecB[0]*vecA[1]);
        if (tripleProduct > 0)
        {
            // Points 2 and 3 should be swapped
            for (int ii = 0; ii < 3; ii++) vecA[ii] = pts[1][ii];
            for (int ii = 0; ii < 3; ii++) pts[1][ii] = pts[2][ii];
            for (int ii = 0; ii < 3; ii++) pts[2][ii] = vecA[ii];
        }
        

//----------------------------------------------------------------------------
//      Compute rotation matrix
//----------------------------------------------------------------------------
        T R[3][3];

        // Calculate the z-axis
        for (int ii = 0; ii < 3; ii++) R[2][ii] = normals[3 * index + ii]; // Should already be unit vector


        // Calculate the y-axis
        for (int ii = 0; ii < 3; ii++) R[1][ii] = pts[1][ii] - pts[0][ii];
        T yLength = sqrt(R[1][0] * R[1][0] + R[1][1] * R[1][1] + R[1][2] * R[1][2]);
        for (int ii = 0; ii < 3; ii++) R[1][ii] = R[1][ii] / yLength; // Make unit vector

        // Calculate the x-axis
        R[0][0] = R[1][1] * R[2][2] - R[1][2] * R[2][1];
        R[0][1] = R[1][2] * R[2][0] - R[1][0] * R[2][2];
        R[0][2] = R[1][0] * R[2][1] - R[1][1] * R[2][0];



//----------------------------------------------------------------------------
//      Find points in the transformed local coordinate system
//----------------------------------------------------------------------------
        T transformedPts[3][3]; // The three vertices
        for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++) transformedPts[ii][jj] = pts[ii][0] * R[jj][0] + pts[ii][1] * R[jj][1] + pts[ii][2] * R[jj][2];
        /*  
                         [x1  y1  z1]
        transformedPts = [x2  y2  z2]
                         [x3  y3  z3]
        */



//----------------------------------------------------------------------------
//      Iterate through all other triangles and compute B field
//----------------------------------------------------------------------------
        // Local coordinate variables of the base triangle
        T xq[2] = {transformedPts[0][0], transformedPts[2][0]};
        T yp[2][2] = {{transformedPts[0][1], transformedPts[2][1]}, {transformedPts[1][1], transformedPts[2][1]}};
        T mp[2] = {(transformedPts[2][1] - transformedPts[0][1]) / (transformedPts[2][0] - transformedPts[0][0]), (transformedPts[2][1] - transformedPts[1][1]) / (transformedPts[2][0] - transformedPts[1][0])};

        // Declare relative variables
        T otherCentre[3];
        T transformedOtherCentre[3];
        T X[2];
        T Y[2][2];
        T Z;
        T Rpq[2][2];
        T Spq[2][2];
        T Tpq[2][2];
        T Upq[2][2];
        T localB[3];

        // Iterate over all "query" triangles
        for (int ii = 0; ii < numTriangles; ii++)
        {
            // Compute the centre of the query triangle
            for (int jj = 0; jj < 3; jj++) otherCentre[jj] = oneThird * (nodes[3 * connections[3 * ii] + jj] + nodes[3 * connections[3 * ii + 1] + jj] + nodes[3 * connections[3 * ii + 2] + jj]);


            // Transform centre coordinates into local coordinate system
            for (int jj = 0; jj < 3; jj++) transformedOtherCentre[jj] = otherCentre[0] * R[jj][0] + otherCentre[1] * R[jj][1] + otherCentre[2] * R[jj][2];


            // Apply PhD methodology
            for (int jj = 0; jj < 3; jj++) localB[jj] = 0.0;
            Z = transformedPts[0][2] - transformedOtherCentre[2];
            for (int kk = 0; kk < 2; kk++) X[kk] = transformedOtherCentre[0] - xq[kk];
            
            for (int jj = 0; jj < 2; jj++)
            {
                for (int kk = 0; kk < 2; kk++)
                {
                    Y[jj][kk] = transformedOtherCentre[1] - yp[jj][kk];

                    // Compute parameters
                    Rpq[jj][kk] = sqrt(X[kk] * X[kk] + Y[jj][kk] * Y[jj][kk] + Z * Z);
                    Spq[jj][kk] = sqrt(1 + mp[jj] * mp[jj]) * Rpq[jj][kk] - X[kk] - mp[jj] * Y[jj][kk];
                    Tpq[jj][kk] = Rpq[jj][kk] - Y[jj][kk];
                    Upq[jj][kk] = (mp[jj] * (X[kk] * X[kk] + Z * Z) - X[kk] * Y[jj][kk]) / (Z * Rpq[jj][kk]);

                    // Check for and correct singularities
                    T eps = 1e-8;
                    if (abs(Spq[jj][kk]) < eps) Spq[jj][kk] = 1.0 / Rpq[jj][kk];
                    if (abs(Tpq[jj][kk]) < eps) Tpq[jj][kk] = 1.0 / Rpq[jj][kk];
                    if (abs(Z) < eps) Upq[jj][kk] = 0.0;

                    // Add to the local pseudo-B field
                    localB[0] += pow((T)(-1.0), (T)(jj+kk)) * (log(Tpq[jj][kk]) - mp[jj] / sqrt(1 + mp[jj] * mp[jj]) * log(Spq[jj][kk]));
                    localB[1] += pow((T)(-1.0), (T)(jj+kk)) / sqrt(1 + mp[jj] * mp[jj]) * log(Spq[jj][kk]);
                    localB[2] += pow((T)(-1.0), (T)(jj+kk)) * atan(Upq[jj][kk]);
                }
            }


            // Transform back into global coordinates and populate B matrix
            if (ii != index)
            {
                for (int jj = 0; jj < 3; jj++) B[jj * numTriangles * numTriangles + index * numTriangles + ii] = localB[0] * R[0][jj] + localB[1] * R[1][jj] + localB[2] * R[2][jj];
            } else {
                for (int jj = 0; jj < 3; jj++) B[jj * numTriangles * numTriangles + index * numTriangles + ii] = 0.0;
            }
        }
    }
}