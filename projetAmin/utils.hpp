#ifndef __XYZSTORE__
#define __XYZSTORE__

void storeXYZnonZerovaluesWithTunnel(unsigned int* imageOut, const char* filename,
	int cols, int rows, int depth,
	int minx, int maxx, int miny, int maxy, int minz, int maxz);

void storeXYZnonZerovalues(unsigned int* imageOut, const char* filename,
	int cols, int rows, int depth);

void storeXYZnonOnevalues(unsigned int* imageOut, const char* filename,
	int cols, int rows, int depth);

void storeXYZregions(unsigned int* imageOut, const char* filename,
	int cols, int rows, int depth, int nbNonZero);

char* readSource(char* kernelPath);
void chk(cl_int status, const char* cmd);

int search(unsigned int* array, int length, int valueToFind);

//float min(float a, float b);
//float max(float a, float b);

int countIntersections(float x, float y, float z,
	float* posX, float* posY, float* posZ, int sizeRegion,
	unsigned int* i1, unsigned int* i2, unsigned int* i3,
	int nbTriangles);

void rotation(float t, float p, float shift, int i, float* inputDiscs);

void storeXYZBorderRegions(unsigned int* imageOut, const char* filename,
	int cols, int rows, int depth,
	int nbRegions, unsigned int* regionsId, unsigned int* closedBlocks, unsigned int* regionsColor);

#endif
