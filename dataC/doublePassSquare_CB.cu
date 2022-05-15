__device__
int access(int a, int b, int c, int W, int H) {
  return (c*H + b)*W + a;
}

__device__
int minAccessiblePixels (int x, int y, int z,
                         unsigned int *workImage, 
                         unsigned int* connectImage,
                           int  W, int  H, int D)
{
  int m = workImage[access(x, y, z, W, H)];
  int bitPosition = 1;
  for (int k = -1; k <= 1; k++) {
    for (int j = -1; j <= 1; j++) {
      for (int i = -1; i <= 1; i++) {
        if ( (i == 0) && (j == 0) && (k == 0)) bitPosition = bitPosition << 1;
        else {
          if (((connectImage[access(x, y, z, W, H)] & bitPosition) == 0) &&
              (x+i >= 0) && (x+i < W) && (y+j >= 0) && (y+j < H) &&
              (z+k >= 0) && (z+k < D) &&
              (workImage[access(x+i, y+j, z+k, W, H)] > 0) &&
              (workImage[access(x+i, y+j, z+k, W, H)] < m))
            m = workImage[access(x+i, y+j, z+k, W, H)];
        }
        bitPosition = bitPosition << 1;
      }
    }
  }
  return m;
}

__global__ 
void double_pass_square_CB(unsigned int* workImage,      //0
                        unsigned int* nbModifiedBlock,   //1
                        unsigned int* connectImage,      //2
                           int  W,                                //3
                           int  H,                                //4
                           int  D,                                //5
                           int  BLOCKSIZEX,                       //6
                           int  BLOCKSIZEY,                       //7
                           int  BLOCKSIZEZ                        //8
) {                     

   //Voxel ID
   int sx = threadIdx.x + blockDim.x*blockIdx.x;
   int sy = threadIdx.y + blockDim.y*blockIdx.y;
   int sz = threadIdx.z + blockDim.z*blockIdx.z;

   int x, y, z;
   int i, j, k;
   unsigned int nbModified = 0;

   /* Right down forward pass */
   for (z = sz * BLOCKSIZEZ; z < (sz + 1) * BLOCKSIZEZ; z++)
    for (y = sy * BLOCKSIZEY; y < (sy + 1) * BLOCKSIZEY; y++)
      for (x = sx * BLOCKSIZEX; x < (sx + 1) * BLOCKSIZEX; x++)
        {
            int m = minAccessiblePixels(x, y, z, workImage, connectImage, W, H, D);
            if (m < workImage[access(x, y, z, W, H)]) {
              workImage[access(x, y, z, W, H)] = m;
              nbModified++;
            }
        }

   /* Left up backward pass */
   for (z = (sz +1 ) * BLOCKSIZEZ - 1; z >= sz * BLOCKSIZEZ; z--)
    for (y = (sy  + 1) * BLOCKSIZEY - 1; y >= sy * BLOCKSIZEY; y--)
      for (x = (sx + 1) * BLOCKSIZEX - 1; x >= sx * BLOCKSIZEX; x--)
        {
          int m = minAccessiblePixels(x, y, z, workImage, connectImage, W, H, D);
          if (m < workImage[access(x, y, z, W, H)]) {
            workImage[access(x, y, z, W, H)] = m;
            nbModified++;
          }
        }

   nbModifiedBlock[(sz*(H/BLOCKSIZEY) + sy)*(W/BLOCKSIZEX) + sx ] = nbModified;
}