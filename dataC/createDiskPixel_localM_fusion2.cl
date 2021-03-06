float myDot(float p1x, float p1y, float p1z, 
          float p2x, float p2y, float p2z)
{
  return (p1x*p2x + p1y*p2y + p1z*p2z);
}

unsigned int leftOrRight(float px, float py, float pz, // Point
						 float dx, float dy, float dz, // Plane origin
						 float nx, float ny, float nz) // Plane normal
{
	float vx = px - dx;
	float vy = py - dy;
	float vz = pz - dz;
	if (myDot(vx, vy, vz, nx, ny, nz) >= 0) return 0;
	else return 1;
}

float distance2(float px, float py, float pz, 
			           float dx, float dy, float dz)
{
	float vx = px - dx;
	float vy = py - dy;
	float vz = pz - dz;
	return myDot(vx, vy, vz, vx, vy, vz);
}

// IntersectSeg between segment P1P2 and disk(D, N, R^2)
unsigned int intersectSeg(float p1x, float p1y, float p1z, 
                          float p2x, float p2y, float p2z, 
                          float dx, float dy, float dz, 
                          float nx, float ny, float nz,
                          float R)
{
  const float EPSILON = 0.000000000001; // Used to test D => PROBLEM HERE ???

  // Check if both points are outside the bounding box of the sphere
  if ((p1x < dx - R) && (p2x < dx - R))      return 0;
  else if ((p1x > dx + R) && (p2x > dx + R)) return 0;
  else if ((p1y < dy - R) && (p2y < dy - R)) return 0;
  else if ((p1y > dy + R) && (p2y > dy + R)) return 0;
  else if ((p1z < dz - R) && (p2z < dz - R)) return 0;
  else if ((p1z > dz + R) && (p2z > dz + R)) return 0;

  // Check if both points are on the same side of the plane
  else if (leftOrRight(p1x, p1y, p1z, dx, dy, dz, nx, ny, nz) ==
      leftOrRight(p2x, p2y, p2z, dx, dy, dz, nx, ny, nz)) 
    return 0;
  else {
    float R2 = R*R;
    float ux = p2x - p1x;
    float uy = p2y - p1y;
    float uz = p2z - p1z;
    float wx = p1x - dx;
    float wy = p1y - dy;
    float wz = p1z - dz;
    float D = myDot(nx, ny, nz, ux, uy, uz);
    //if (D < EPSILON) return 0;                // PROBLEM HERE ???
    float N = -myDot(nx, ny, nz, wx, wy, wz);
    float sI = N / D;
    float Ix = p1x + sI * ux;                  // compute segment intersect point
    float Iy = p1y + sI * uy;
    float Iz = p1z + sI * uz;
    if (sI < 0 || sI > 1) return 0; 
    else if (distance2(Ix, Iy, Iz, dx, dy, dz) < R2) return 1;
    else return 0;
  }
}

void fillConnectivity(int x, int y, int z,
                      float dx, float dy, float dz, 
                      float nx, float ny, float nz,
                      float R,
                      __global unsigned int* connectivity,
                      int W, int H)
{
    // Check all 26 neighbours for possible intersectSegs
    unsigned int bitPosition = 1;
    for (int k = -1; k <= 1; k++) {
      for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
          if ((i == 0) && (j == 0) && (k == 0)) 
            bitPosition = (bitPosition << 1);
          else if (intersectSeg(x + 0.5, y + 0.5, z + 0.5,
                                (x + i) + 0.5, (y + j) + 0.5, (z + k) + 0.5,
                                dx, dy, dz, nx, ny, nz, R) == 1)
                connectivity[(z*H + y)*W + x] |= bitPosition;
          bitPosition = (bitPosition << 1);
        }
      }
    }  
}

// Check if one segment intersects the disc
int checkConnectivity(int x, int y, int z,
                      float dx, float dy, float dz, 
                      float nx, float ny, float nz,
                      float R)
{
    // Check all 26 neighbours for possible intersectSegs
    for (int k = -1; k <= 1; k++) {
      for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
          if ((i == 0) && (j == 0) && (k == 0)) continue;
          if (intersectSeg(x + 0.5, y + 0.5, z + 0.5,
                                (x + i) + 0.5, (y + j) + 0.5, (z + k) + 0.5,
                                dx, dy, dz, nx, ny, nz, R) == 1)
            return 1;
        }
      }
    }
    return 0;  
}

// Check if there is a chance of intersection between a box and a sphere
unsigned int intersectBB(float p1x, float p1y, float p1z, 
                         float p2x, float p2y, float p2z, 
                         float dx, float dy, float dz, 
                         float radius)
{
  if (dx + radius < p1x)        return 0;
  else if (dx - radius > p2x)   return 0;
  else if (dy + radius < p1y)   return 0;
  else if (dy - radius > p2y)   return 0;
  else if (dz + radius < p1z)   return 0;
  else if (dz - radius > p2z)   return 0;
  else return 1;
}

__kernel void create_disk_pixel_localM_fusion2(
	  __global unsigned int* connectivity,       //0
    __global float* src_disks,                 //1
    int  W,                                    //2
    int  H,                                    //3
    int  nbDisks,                              //4
    __local  unsigned int* src_disksIndices,   //5
    int maxDisksPerBlock                       //6
) 
{ 
   //Voxel IDs
   const int x = get_global_id(0);
   const int y = get_global_id(1);
   const int z = get_global_id(2);
   const int xloc = get_local_id(0);
   const int yloc = get_local_id(1);
   const int zloc = get_local_id(2);
   const int startingx = x - xloc;
   const int startingy = y - yloc;
   const int startingz = z - zloc;
   const float bbx = startingx - 0.5;
   const float bby = startingy - 0.5;
   const float bbz = startingz - 0.5;
   const int xsize = get_local_size(0);
   const int ysize = get_local_size(1);
   const int zsize = get_local_size(2);
   const float bbex = startingx + xsize - 1 + 0.5;
   const float bbey = startingy + ysize - 1 + 0.5;
   const float bbez = startingz + zsize - 1 + 0.5;

   __local int cptDisksPerBlock;          // Shared 

   // Set atomic counter to 0
   if (((zloc*ysize + yloc)*xsize + xloc) == 0)
    cptDisksPerBlock = 0;
   barrier (CLK_LOCAL_MEM_FENCE);

   // Store the discs that may intersect the local 
   // block of size (xsize, ysize, zsize)    
   int incr = xsize*ysize*zsize;
   int ix;
   for (ix = (zloc*ysize + yloc)*xsize + xloc; 
        ix < nbDisks; 
        ix += incr) {
    // Each disk has 7 values: pos(3), normal(3), radius(1)
    float dx = src_disks[ix*7];
    float dy = src_disks[ix*7 + 1];
    float dz = src_disks[ix*7 + 2];
    float radius = src_disks[ix*7 + 6];
    if (intersectBB(bbx, bby, bbz, bbex, bbey, bbez,
                                       dx, dy, dz, radius) == 1) {
      if (cptDisksPerBlock < maxDisksPerBlock) {
        int old_cpt = atomic_inc(&cptDisksPerBlock);        // Atomics
        src_disksIndices[old_cpt] = ix;  
      }
    }
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   // Effectively intersecting discs
   unsigned int intersectingDiscs[1000];
   int nbInterDiscs = 0;
   const float EPSILON = 0.00001; // Used to test if discs are parallel

   //for (int ix = 0; ix < nbDisks; ix++) { 
   for (int cpt = 0; cpt < cptDisksPerBlock; cpt++) {
    int ix = src_disksIndices[cpt];
    // Each disk has 7 values: pos(3), normal(3), radius(1)
    float dx = src_disks[ix*7];
    float dy = src_disks[ix*7 + 1];
    float dz = src_disks[ix*7 + 2];
    float nx = src_disks[ix*7 + 3];
    float ny = src_disks[ix*7 + 4];
    float nz = src_disks[ix*7 + 5];
    float radius = src_disks[ix*7 + 6];
    if (checkConnectivity(x, y, z, dx, dy, dz, nx, ny, nz, radius) == 1) {
      int shouldBeAdded = 1;
      for (int ii = 0; ii < nbInterDiscs; ii++) {
        float nxII = src_disks[intersectingDiscs[ii]*7 + 3];
        float nyII = src_disks[intersectingDiscs[ii]*7 + 4];
        float nzII = src_disks[intersectingDiscs[ii]*7 + 5];
        //if (distance2(nx, ny, nz, nxII, nyII, nzII) < EPSILON)
        if ((nx == nxII) && (ny == nyII) && (nz == nzII)) {
          shouldBeAdded = 0;
          break;
        }
      }
      if ((shouldBeAdded == 1) && (nbInterDiscs < 1000))
        intersectingDiscs[nbInterDiscs++] = ix;
    }
   }

   connectivity[(z*H + y)*W + x] = 0;
   for (int ii = 0; ii < nbInterDiscs; ii++) {
      int ix = intersectingDiscs[ii];
      float dx = src_disks[ix*7];
      float dy = src_disks[ix*7 + 1];
      float dz = src_disks[ix*7 + 2];
      float nx = src_disks[ix*7 + 3];
      float ny = src_disks[ix*7 + 4];
      float nz = src_disks[ix*7 + 5];
      float radius = src_disks[ix*7 + 6];
      fillConnectivity(x, y, z, dx, dy, dz, nx, ny, nz, radius, 
                       connectivity, W, H);
   }
}

__kernel void show_disks(
    __global unsigned int* connectivity,       //0
    __global float* src_disks,                 //1
    int  W,                                    //2
    int  H,                                    //3
    int  nbDisks,                              //4
    __local  unsigned int* src_disksIndices,   //5
    int maxDisksPerBlock,                      //6
    int xMinTunnel,                            //7
    int xMaxTunnel,                            //8
    int yMinTunnel,                            //9
    int yMaxTunnel,                            //10
    int zMinTunnel,                            //11
    int zMaxTunnel                             //12
) 
{ 
   //Voxel IDs
   const int x = get_global_id(0);
   const int y = get_global_id(1);
   const int z = get_global_id(2);

   connectivity[(z*H + y)*W + x] = 0;

   const int xloc = get_local_id(0);
   const int yloc = get_local_id(1);
   const int zloc = get_local_id(2);
   const int startingx = x - xloc;
   const int startingy = y - yloc;
   const int startingz = z - zloc;
   const float bbx = startingx - 0.5;
   const float bby = startingy - 0.5;
   const float bbz = startingz - 0.5;
   const int xsize = get_local_size(0);
   const int ysize = get_local_size(1);
   const int zsize = get_local_size(2);
   const float bbex = startingx + xsize - 1 + 0.5;
   const float bbey = startingy + ysize - 1 + 0.5;
   const float bbez = startingz + zsize - 1 + 0.5;

   __local int cptDisksPerBlock;          // Shared 

   // Set atomic counter to 0
   if (((zloc*ysize + yloc)*xsize + xloc) == 0)
    cptDisksPerBlock = 0;
   barrier (CLK_LOCAL_MEM_FENCE);

   // Store the discs that may intersect the local 
   // block of size (xsize, ysize, zsize)    
   int incr = xsize*ysize*zsize;
   int ix;
   for (ix = (zloc*ysize + yloc)*xsize + xloc; 
        ix < nbDisks; 
        ix += incr) {
    // Each disk has 7 values: pos(3), normal(3), radius(1)
    float dx = src_disks[ix*7];
    float dy = src_disks[ix*7 + 1];
    float dz = src_disks[ix*7 + 2];
    float radius = src_disks[ix*7 + 6];
    if (intersectBB(bbx, bby, bbz, bbex, bbey, bbez,
                                       dx, dy, dz, radius) == 1) {
      if (cptDisksPerBlock < maxDisksPerBlock) {
        int old_cpt = atomic_inc(&cptDisksPerBlock);        // Atomics
        src_disksIndices[old_cpt] = ix;  
      }
    }
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   // Effectively intersecting discs
   for (int cpt = 0; cpt < cptDisksPerBlock; cpt++) {
    int ix = src_disksIndices[cpt];
    // Each disk has 7 values: pos(3), normal(3), radius(1)
    float dx = src_disks[ix*7];
    float dy = src_disks[ix*7 + 1];
    float dz = src_disks[ix*7 + 2];
    float nx = src_disks[ix*7 + 3];
    float ny = src_disks[ix*7 + 4];
    float nz = src_disks[ix*7 + 5];
    float radius = src_disks[ix*7 + 6];
    if ((checkConnectivity(x, y, z, dx, dy, dz, nx, ny, nz, radius) == 1) &&
      (connectivity[(z*H + y)*W + x] < ix) )
        connectivity[(z*H + y)*W + x]= ix;
   }

//  if ( ((x >= xMinTunnel) && (x <= xMaxTunnel)) &&
//        ((y >= yMinTunnel) && (y <= yMaxTunnel)) &&
//        ((z >= zMinTunnel) && (z <= zMinTunnel)) )
//    connectivity[(z*H + y)*W + x] = 1;
}