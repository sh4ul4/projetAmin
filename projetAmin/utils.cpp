#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>

#include "utils.hpp"

void storeXYZnonZerovaluesWithTunnel(unsigned int *imageOut, const char *filename, 
            int cols, int rows, int depth,
            int minx, int maxx, int miny, int maxy, int minz, int maxz) {

  FILE* xyzfile; 
  xyzfile = fopen(filename, "wb"); 
  
  // Corners
  imageOut[(0*rows + 0)*cols + 0] = 1;
  imageOut[(0*rows + 0)*cols + (cols - 1)] = 1;
  imageOut[(0*rows + (rows - 1))*cols + 0] = 1;
  imageOut[(0*rows + (rows - 1))*cols + (cols - 1)] = 1;
  imageOut[((depth-1)*rows + 0)*cols + 0] = 1;
  imageOut[((depth-1)*rows + 0)*cols + (cols - 1)] = 1;
  imageOut[((depth-1)*rows + (rows - 1))*cols + 0] = 1;
  imageOut[((depth-1)*rows + (rows - 1))*cols + (cols - 1)] = 1;

  int nbNonZero = 0;

    for (int k = 0; k < depth; ++k)
     for (int j = 0; j < rows; j++) 
      for (int i = 0; i < cols; i++) {
        if ((k >= minz) && (j >= miny) && (i >= minx) &&
            (k <= maxz) && (j <= maxy) && (i <= maxx))
          continue;
        if (imageOut[(k*rows + j)*cols + i] != 0) nbNonZero++;
      }

    // Total number of "atoms" 
    fprintf(xyzfile, "%d\n\n", nbNonZero);  
  
    int count = 0; 
    for (int k = 0; k < depth; ++k)
     for (int j = 0; j < rows; j++) 
      for (int i = 0; i < cols; i++) {
        if ((k >= minz) && (j >= miny) && (i >= minx) &&
            (k <= maxz) && (j <= maxy) && (i <= maxx))
          continue;
        if (imageOut[(k*rows + j)*cols + i] != 0) 
            fprintf(xyzfile, "C\t %d\t %d \t %d \t %d\n", 
              imageOut[(k*rows + j)*cols + i], i, j, k); 
      }
    fclose(xyzfile); 
}

void storeXYZnonZerovalues(unsigned int *imageOut, const char *filename, 
            int cols, int rows, int depth) {

  FILE* xyzfile; 
  xyzfile = fopen(filename, "wb"); 
  
  // Corners
  imageOut[(0*rows + 0)*cols + 0] = 1;
  imageOut[(0*rows + 0)*cols + (cols - 1)] = 1;
  imageOut[(0*rows + (rows - 1))*cols + 0] = 1;
  imageOut[(0*rows + (rows - 1))*cols + (cols - 1)] = 1;
  imageOut[((depth-1)*rows + 0)*cols + 0] = 1;
  imageOut[((depth-1)*rows + 0)*cols + (cols - 1)] = 1;
  imageOut[((depth-1)*rows + (rows - 1))*cols + 0] = 1;
  imageOut[((depth-1)*rows + (rows - 1))*cols + (cols - 1)] = 1;

  int nbNonZero = 0;

    for (int k = 0; k < depth; ++k)
     for (int j = 0; j < rows; j++) 
      for (int i = 0; i < cols; i++) {
        if (imageOut[(k*rows + j)*cols + i] != 0) nbNonZero++;
      }

    // Total number of "atoms" 
    fprintf(xyzfile, "%d\n\n", nbNonZero);  
  
    int count = 0; 
    for (int k = 0; k < depth; ++k)
     for (int j = 0; j < rows; j++) 
      for (int i = 0; i < cols; i++) {
        if (imageOut[(k*rows + j)*cols + i] != 0) 
            fprintf(xyzfile, "C\t %d\t %d \t %d \t %d\n", 
              imageOut[(k*rows + j)*cols + i], i, j, k); 
      }
    fclose(xyzfile); 
}

void storeXYZnonOnevalues(unsigned int *imageOut, const char *filename, 
        int cols, int rows, int depth) {

  FILE* xyzfile; 
  xyzfile = fopen(filename, "wb"); 
  
  // Corners
  imageOut[(0*rows + 0)*cols + 0] = 0;
  imageOut[(0*rows + 0)*cols + (cols - 1)] = 0;
  imageOut[(0*rows + (rows - 1))*cols + 0] = 0;
  imageOut[(0*rows + (rows - 1))*cols + (cols - 1)] = 0;
  imageOut[((depth-1)*rows + 0)*cols + 0] = 0;
  imageOut[((depth-1)*rows + 0)*cols + (cols - 1)] = 0;
  imageOut[((depth-1)*rows + (rows - 1))*cols + 0] = 0;
  imageOut[((depth-1)*rows + (rows - 1))*cols + (cols - 1)] = 0;

  int nbNonZero = 0;

    for (int k = 0; k < depth; ++k)
     for (int j = 0; j < rows; j++) 
      for (int i = 0; i < cols; i++) 
//        if (imageOut[(k*rows + j)*cols + i] != 0) nbNonZero++;
        if (imageOut[(k*rows + j)*cols + i] != 1) nbNonZero++;

    // Total number of "atoms" 
    fprintf(xyzfile, "%d\n\n", nbNonZero);  
  
    int count = 0; 
    for (int k = 0; k < depth; ++k)
     for (int j = 0; j < rows; j++) 
      for (int i = 0; i < cols; i++) 
//        if (imageOut[(k*rows + j)*cols + i] != 0) 
        if (imageOut[(k*rows + j)*cols + i] != 1) {
            fprintf(xyzfile, "C\t %d\t %d \t %d \t %d\n", 
              imageOut[(k*rows + j)*cols + i], i, j, k); 
        }
    fclose(xyzfile); 
}

void storeXYZregions(unsigned int *imageOut, const char *filename, 
        int cols, int rows, int depth, int nbNonZero)  
{

    FILE* xyzfile; 
    xyzfile = fopen(filename, "wb"); 
  
    // Total number of "atoms" 
    fprintf(xyzfile, "%d\n\n", nbNonZero);
    //fprintf(xyzfile, "%d\n\n", rows*cols*depth);  
  
    int count = 0; 
    for (int k = 0; k < depth; ++k)
     for (int j = 0; j < rows; j++) 
      for (int i = 0; i < cols; i++) 
        //if (imageOut[(k*rows + j)*cols + i] != 0) 
            fprintf(xyzfile, "C %d\t %d \t %d \t %d\n", 
              imageOut[(k*rows + j)*cols + i]%256, i, j, k); 
    fclose(xyzfile); 
}

// This function reads in a text file and stores it as a char pointer
char* readSource(char* kernelPath) {

   cl_int status;
   FILE *fp;
   char *source;
   long int size;

   printf("Program file is: %s\n", kernelPath);

   fp = fopen(kernelPath, "rb");
   if(!fp) {
      printf("Could not open kernel file\n");
      exit(-1);
   }
   status = fseek(fp, 0, SEEK_END);
   if(status != 0) {
      printf("Error seeking to end of file\n");
      exit(-1);
   }
   size = ftell(fp);
   if(size < 0) {
      printf("Error getting file position\n");
      exit(-1);
   }

   rewind(fp);

   source = (char *)malloc(size + 1);

   int i;
   for (i = 0; i < size+1; i++) {
      source[i]='\0';
   }

   if(source == NULL) {
      printf("Error allocating space for the kernel source\n");
      exit(-1);
   }

   fread(source, 1, size, fp);
   source[size] = '\0';

   return source;
}

void chk(cl_int status, const char* cmd) {

   if(status != CL_SUCCESS) {
      printf("%s failed (%d)\n", cmd, status);
      exit(-1);
   }
   else printf("%s succeeded\n", cmd);
}

int search(unsigned int* array, int length, int valueToFind)
{
    int l = 0;
    int r = length-1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (array[m] == valueToFind) return m;
        if (array[m] <  valueToFind) l = m + 1;
        else r = m - 1;
    }
    return -1;
}

float myDot(float p1x, float p1y, float p1z, 
          float p2x, float p2y, float p2z)
{
  return (p1x*p2x + p1y*p2y + p1z*p2z);
}

int leftOrRight(float px, float py, float pz, // Point
             float dx, float dy, float dz, // Plane origin
             float nx, float ny, float nz) // Plane normal
{
  float vx = px - dx;
  float vy = py - dy;
  float vz = pz - dz;
  return (myDot(vx, vy, vz, nx, ny, nz) >= 0);
}

#define SMALL_NUM   0.00000001

void storeXYZBorderRegions(unsigned int* imageOut, const char* filename,
	int cols, int rows, int depth,
	int nbRegions, unsigned int* regionsId, unsigned int* closedBlocks, unsigned int* regionsColor) {

	FILE* xyzfile;
	xyzfile = fopen(filename, "wb");

	int nbCells = 8; // 8 corners

	for (int k = 0; k < depth; ++k)
		for (int j = 0; j < rows; j++)
			for (int i = 0; i < cols; i++) {
				if (((i == 0) && (j == 0) && (k == 0)) ||
					((i == cols - 1) && (j == 0) && (k == 0)) ||
					((i == 0) && (j == rows - 1) && (k == 0)) ||
					((i == cols - 1) && (j == rows - 1) && (k == 0)) ||
					((i == 0) && (j == 0) && (k == depth - 1)) ||
					((i == cols - 1) && (j == 0) && (k == depth - 1)) ||
					((i == 0) && (j == rows - 1) && (k == depth - 1)) ||
					((i == cols - 1) && (j == rows - 1) && (k == depth - 1))) {
					continue;
				}
				int pos = search(regionsId, nbRegions,
					imageOut[(k * rows + j) * cols + i]);
				if (pos != -1) {
					if (closedBlocks[pos] == 0) nbCells++;
				}
			}

	// Total number of "atoms" 
	fprintf(xyzfile, "%d\n\n", nbCells);

	int count = 0;
	for (int k = 0; k < depth; ++k)
		for (int j = 0; j < rows; j++)
			for (int i = 0; i < cols; i++) {
				if (((i == 0) && (j == 0) && (k == 0)) ||
					((i == cols - 1) && (j == 0) && (k == 0)) ||
					((i == 0) && (j == rows - 1) && (k == 0)) ||
					((i == cols - 1) && (j == rows - 1) && (k == 0)) ||
					((i == 0) && (j == 0) && (k == depth - 1)) ||
					((i == cols - 1) && (j == 0) && (k == depth - 1)) ||
					((i == 0) && (j == rows - 1) && (k == depth - 1)) ||
					((i == cols - 1) && (j == rows - 1) && (k == depth - 1))) {
					fprintf(xyzfile, "C\t 0\t %d \t %d \t %d\n", i, j, k);
					continue;
				}

				int pos = search(regionsId, nbRegions,
					imageOut[(k * rows + j) * cols + i]);
				if (pos != -1) {
					if (closedBlocks[pos] == 0)
						fprintf(xyzfile, "C\t %d\t %d \t %d \t %d\n",
							regionsColor[pos], i, j, k);
				}
			}
	fclose(xyzfile);
}