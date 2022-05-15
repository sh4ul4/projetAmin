#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <locale.h>
#include <float.h> 
#include <string.h>
#include "utils.hpp"

#define M_PI 3.14159265358979323846

//#define GATEDCC
#ifdef  GATEDCC
#define PLATFORM_TO_USE 3	// For gate.dcc.uchile.cl: 3 and 0, 1 or 2 (not 3 !!!)
#define DEVICE_TO_USE 1
#else
#define PLATFORM_TO_USE 0
#define DEVICE_TO_USE 0
#endif

#define KERNELCONNECT "C:/dataGPU/createDiskPixel_localM_fusion2.cl"
#define KERNELBLOCKS "C:/dataGPU/doublePassSquare_CB.cl"

//#define INPUTDISCS "C:/dataGPU/input/Bloque_2_Exp6_P32_5_Simulation_1.csv" //Bloque_2_Exp3_P32_5_Simulation_10.csv" //"C:/dataGPU/input/DFNright1.csv"
#define INPUTDISCS "C:/dataGPU/input/synData-model2.csv"
#define INPUTFRAGMENTATION "C:/dataGPU/input/fragmentationParameters.csv"
#define LIMITLINES 500000      // Number of lines taken from the input file
#define WITHTUNNEL 1          // 0 or 1
#define REMOVENONCLOSEDBLOCKS    0  // For non tunnel experiments only (to show the closed poluhedral with block 1 = just show closed blocks/ 0 = not show )
#define SHOWONLYTOPRIGHTBLOCKS   1  // For tunnel experiments only => 20/12/21: now selects blocks touching Left, Right, Bottom and Top faces

// Control flow 
#define ONLYDISCS             0
#define ONLYDISCSWITHTUNNEL   0
#define ONLYCONNECTIVITY      0 
#define STORECONNECTIVIY      0
#define STOREREGIONS          1
#define STOREREGIONSCSV       1 
#define STORESTATS            1
// Here Ones can have a control on the faces of the block*********
#define FACELEFT  0
#define FACERIGHT 1
#define FACEDOWN  1
#define FACEUP    0
#define FACEFRONT 0
#define FACEBACK  0


#define OUTPUTRESULTS "C:/dataGPU/output/results.csv"
#define OUTPUTREGIONS "C:/dataGPU/output/regions"
#define OUTPUTCONNECT "C:/dataGPU/output/connectivity.xyz"
#define CSVFILE "C:/dataGPU/output/fragCurve"
#define STATSFILE "C:/dataGPU/output/stats"
#define HISTFILE "C:/dataGPU/output/histogram"

// Resolution of the voxel grid (read in in the input file)
int     ImgSizeX;
int     ImgSizeY;
int     ImgSizeZ;

// Parameters to translate original data to domain [0 IMGSIZEX]
// Depend on minX and maxX values in the input file
double Domain_Size;  // (20.0) //(100.0)
double Domain_Shift;                                          // #define SHIFT (10) //(50)
#define MULT (ImgSizeX/Domain_Size)
#define REDUCTION (((Domain_Size)*(Domain_Size)*(Domain_Size))/((ImgSizeX)*(ImgSizeX)*(ImgSizeX)))  
         // Used to compute sizes in m^3

// Density read from input file
float Density;

// Tunnel defined as propoportions in X, Y and Z
// => for example (0.2*ImgSizeX) means the left wall will be located at 20%
#define TUNNELMINX ((int)(0.2*ImgSizeX)) //((int)(0.45*IMGSIZEX)) //((int)10*MULT/4.0)           //(3*MULT))
#define TUNNELMAXX ((int)(0.4*ImgSizeX)) //((int)(0.55*IMGSIZEX)) //((int)30*MULT/4.0)           //(7*MULT))
#define TUNNELMINY (0)
#define TUNNELMAXY (ImgSizeX)
#define TUNNELMINZ ((int)(0.4*ImgSizeX))
#define TUNNELMAXZ ((int)(0.6*ImgSizeX))

// OpenCL blocks of threads (with local memory use) for connectivity computation
#define WPSIZE  8             
#define MAXDISCSPERBLOCK 10000   // Estimated number but should not be too high...

// OpenCL blocks of threads for flood computation
#ifdef  GATEDCC
#define NBBLOCKS 128				
#else
#define NBBLOCKS 32           
#endif
#define BLOCKSIZEX (ImgSizeX/NBBLOCKS)
#define BLOCKSIZEY (ImgSizeX/NBBLOCKS)
#define BLOCKSIZEZ (ImgSizeX/NBBLOCKS)	

int main(int argc, char* argv[])
{

    time_t timer;
    char buffer[26];
    struct tm* tm_info;
    clock_t start, end;

    start = clock();

    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    puts(buffer);

    // Reading fragmentation parameters
    FILE* f = fopen(INPUTFRAGMENTATION, "r");
    if (f == NULL) {
        printf("Error opening input file %s\n", INPUTFRAGMENTATION);
        exit(2);
    }
    printf("Reading file: %s\n", INPUTFRAGMENTATION);
    if (fscanf(f, "%d\n", &ImgSizeX) == EOF) {
        printf("Error reading input file %s\n", INPUTFRAGMENTATION);
        exit(2);
    }
    printf("Resolution: %d\n", ImgSizeX);
    ImgSizeY = ImgSizeX;
    ImgSizeZ = ImgSizeX;
    if (fscanf(f, "%f\n", &Density) == EOF) {
        printf("Error reading input file %s\n", INPUTFRAGMENTATION);
        exit(2);
    }
    printf("Density: %f\n", Density);
    fclose(f);

    /*
    // InputDisks
    f = fopen(INPUTDISCS, "r");
    if (f == NULL) {
          printf("Error opening input file %s\n", INPUTDISCS);
          exit(2);
    }
    int minX, maxX;
    printf("Reading file: %s\n", INPUTDISCS);
    if (fscanf(f, "%d; %d\n", &minX, &maxX) == EOF) {
          printf("Error reading input file %s\n", INPUTDISCS);
          exit(2);
    }
    Domain_Size = maxX - minX;
    Domain_Shift = -minX;// + maxX)/2;
    printf("Domain_Size: %f / Domain_Shift: %f / Mult: %f\n",
       Domain_Size, Domain_Shift, MULT);
    int nbDisks;
    if (fscanf(f, "%d\n", &nbDisks) == EOF) {
          printf("Error reading input file %s\n", INPUTDISCS);
          exit(2);
    }
    if (nbDisks > LIMITLINES) nbDisks = LIMITLINES;
    printf("nbDisks: %d\n", nbDisks);
    float px, py, pz, nx, ny, nz, radius;
    int setId;
    size_t dataSizeDisks = nbDisks*7*sizeof(float);
    float* inputDisks = (float*)malloc(dataSizeDisks);
    for (int i = 0; i < nbDisks; i++) {
     fscanf(f, "%f;%f;%f;%f;%f;%f;%f;%d\n",
          &px, &py, &pz, &radius, &nx, &ny, &nz, &setId);
       {
          px = (px + Domain_Shift)*MULT;
          py = (py + Domain_Shift)*MULT;
          pz = (pz + Domain_Shift)*MULT;
          radius *= MULT;
          inputDisks[i*7] = px;
          inputDisks[i*7 + 1] = py;
          inputDisks[i*7 + 2] = pz;
          inputDisks[i*7 + 3] = nx;
          inputDisks[i*7 + 4] = ny;
          inputDisks[i*7 + 5] = nz;
          inputDisks[i*7 + 6] = radius;

          if (i == 0) printf("Disc %d: [%f, %f, %f] / %f / [%f, %f, %f]\n",
                     i, px, py, pz, radius, nx, ny, nz);
          if (i >= LIMITLINES) break;
       }
    }
    fclose(f);
    */

    // InputDisks
    f = fopen(INPUTDISCS, "r");
    if (f == NULL) {
        printf("Error opening input file %s\n", INPUTDISCS);
        exit(2);
    }
    int minX, maxX;
    printf("Reading file: %s\n", INPUTDISCS);
    if (fscanf(f, "%d; %d\n", &minX, &maxX) == EOF) {
        printf("Error reading input file %s\n", INPUTDISCS);
        exit(2);
    }
    Domain_Size = maxX - minX;
    Domain_Shift = -minX;// + maxX)/2;
    printf("Domain_Size: %f / Domain_Shift: %f / Mult: %f\n",
        Domain_Size, Domain_Shift, MULT);
    int nbDisks;
    if (fscanf(f, "%d\n", &nbDisks) == EOF) {
        printf("Error reading input file %s\n", INPUTDISCS);
        exit(2);
    }
    if (nbDisks > LIMITLINES) nbDisks = LIMITLINES;
    printf("nbDisks: %d\n", nbDisks);
    float px, py, pz, nx, ny, nz, radius;
    int setId;
    size_t dataSizeDisks = nbDisks * 7 * sizeof(float);
    float* inputDisks = (float*)malloc(dataSizeDisks);
    int cptFile = 2;
    for (int i = 0; i < nbDisks; i++) {
        if (fscanf(f, "%f;%f;%f;%f;%f;%f;%f;%d\n",
            &px, &py, &pz, &radius, &nx, &ny, &nz, &setId) == EOF) {
            // Load next file
            fclose(f);
            char nextFile[100], nextEnd[10];
            sprintf(nextFile, "%s", INPUTDISCS);
            int l = strlen(nextFile);
            nextFile[l - 4] = 0;
            sprintf(nextEnd, "-%d.csv", cptFile++);
            strcat(nextFile, nextEnd);
            f = fopen(nextFile, "r");
            if (f == NULL) {
                printf("Error opening input file %s\n", nextFile);
                exit(2);
            }
            printf("Reading file: %s\n", nextFile);
        }

        px = (px + Domain_Shift) * MULT;
        py = (py + Domain_Shift) * MULT;
        pz = (pz + Domain_Shift) * MULT;
        radius *= MULT;
        inputDisks[i * 7] = px;
        inputDisks[i * 7 + 1] = py;
        inputDisks[i * 7 + 2] = pz;
        inputDisks[i * 7 + 3] = nx;
        inputDisks[i * 7 + 4] = ny;
        inputDisks[i * 7 + 5] = nz;
        inputDisks[i * 7 + 6] = radius;

        if (i == 0) printf("Disc %d: [%f, %f, %f] / %f / [%f, %f, %f]\n",
            i, px, py, pz, radius, nx, ny, nz);
        if (i >= LIMITLINES) break;
    }
    fclose(f);

    // Set up the OpenCL environment
    cl_int status;

    // Discovery platform
    cl_platform_id platforms[100];
    cl_platform_id platform;
    status = clGetPlatformIDs(100, platforms, NULL);
    chk(status, "clGetPlatformIDs");
    platform = platforms[PLATFORM_TO_USE];

    // Discover device
    cl_device_id devices[100];
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 100, devices, NULL);
    chk(status, "clGetDeviceIDs");

    cl_device_id device = devices[DEVICE_TO_USE];

    // Create context
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platform), 0 };
    cl_context context;
    context = clCreateContext(props, 1, &device, NULL, NULL, &status);
    chk(status, "clCreateContext");

    // Create command queue
    cl_command_queue queue;
    queue = clCreateCommandQueue(context, device, 0, &status);
    chk(status, "clCreateCommandQueue");

    // Create the input and output buffers on the device
    cl_mem d_inputDisks;
    d_inputDisks = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSizeDisks, NULL,
        &status);
    chk(status, "clCreateBuffer d_inputDisks");

    size_t dataSizeImage = ImgSizeX * ImgSizeY * ImgSizeZ * sizeof(unsigned int);
    unsigned int* connectImage = (unsigned int*)malloc(dataSizeImage);
    cl_mem d_connectImage;
    d_connectImage = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSizeImage, NULL,
        &status);
    chk(status, "clCreateBuffer d_connectImage");

    // Copy the input disks to the device
    status = clEnqueueWriteBuffer(queue, d_inputDisks, CL_TRUE, 0, dataSizeDisks,
        inputDisks, 0, NULL, NULL);
    chk(status, "clEnqueueWriteBuffer inputDisks");

    // Create a program object with source and build it
    const char* source;
    source = readSource((char*)KERNELCONNECT);

    cl_program program;
    program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    chk(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (status != 0) {
        printf("%s failed (%d)\n", "clBuildProgram", status);
        size_t len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char* buffer = (char*)malloc(len);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        printf("%s", buffer);
        exit(1);
    }
    else printf("clBuildProgram succeeded\n");

    // Create the kernel object
    cl_kernel kernel;
    kernel = clCreateKernel(program, "create_disk_pixel_localM_fusion2", &status);
    chk(status, "clCreateKernel");

    // Set the kernel arguments
    int maxNbDiscsPerBlock = MAXDISCSPERBLOCK;
    int sizeForLocalMemory = sizeof(unsigned int) * maxNbDiscsPerBlock;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_connectImage);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_inputDisks);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &ImgSizeX);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &ImgSizeY);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &nbDisks);
    status |= clSetKernelArg(kernel, 5, sizeForLocalMemory, NULL);
    status |= clSetKernelArg(kernel, 6, sizeof(int), &maxNbDiscsPerBlock);

    chk(status, "clSetKernelArg");

    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    puts(buffer);


    // Run the first kernel
    //size_t globalSize = nbDisks;
    size_t globalSize[3] = { ImgSizeX, ImgSizeY, ImgSizeZ };
    size_t localSize[3] = { WPSIZE, WPSIZE, WPSIZE };
    status = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalSize,
        localSize, 0, NULL, NULL);
    chk(status, "clEnqueueNDRange");
    status = clFlush(queue);
    chk(status, "clFlush");
    status = clFinish(queue);
    chk(status, "clFinish");
    printf("Kernel create_disk: done\n");

    status = clEnqueueReadBuffer(queue, d_connectImage, CL_TRUE, 0, dataSizeImage,
        connectImage, 0, NULL, NULL);
    chk(status, "clEnqueueReadBuffer");
    status = clFlush(queue);
    chk(status, "clFlush");
    status = clFinish(queue);
    chk(status, "clFinish");

    if (STORECONNECTIVIY) {
        if (WITHTUNNEL)
            storeXYZnonZerovaluesWithTunnel(connectImage, "connectivity.xyz",
                ImgSizeX, ImgSizeY, ImgSizeZ,
                TUNNELMINX, TUNNELMAXX, TUNNELMINY, TUNNELMAXY, TUNNELMINZ, TUNNELMAXZ);
        else
            storeXYZnonZerovalues(connectImage, "connectivity.xyz",
                ImgSizeX, ImgSizeY, ImgSizeZ);
    }

    // Free the memory on the host for unnecessary objects
    //free(inputDisks);

    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    puts(buffer);

    if (ONLYCONNECTIVITY)
        return 0;

    ///////////////////////////////////////////////////: START THE FLOOD

       // Work image buffer
    unsigned int* workImage = (unsigned int*)malloc(dataSizeImage);
    if (WITHTUNNEL) {
        for (int k = 0; k < ImgSizeZ; ++k) {
            for (int j = 0; j < ImgSizeY; ++j) {
                for (int i = 0; i < ImgSizeX; ++i) {
                    // O value is for "tunnel" regions
                    workImage[(k * ImgSizeY + j) * ImgSizeX + i] = 0;
                    // 1 is used to mark voxels on the edges
                    if ((i == 0) || (i == ImgSizeX - 1) || (j == 0) || (j == ImgSizeY - 1)
                        || (k == 0) || (k == ImgSizeZ - 1))
                        workImage[(k * ImgSizeY + j) * ImgSizeX + i] = 1;
                    else if (((k < TUNNELMINZ) || (j < TUNNELMINY) || (i < TUNNELMINX) ||
                        (k > TUNNELMAXZ) || (j > TUNNELMAXY) || (i > TUNNELMAXX))) {
                        workImage[(k * ImgSizeY + j) * ImgSizeX + i] =
                            (k * ImgSizeY + j) * ImgSizeX + i + 1;     // 1 is added to preserve 0 val
                    }
                }
            }
        }
        printf("Tunnel: [%d, %d, %d, %d, %d, %d]\n", (int)TUNNELMINZ, (int)TUNNELMAXZ,
            (int)TUNNELMINY, (int)TUNNELMAXY, (int)TUNNELMINZ, (int)TUNNELMAXZ);
    }
    else {
        for (int k = 0; k < ImgSizeZ; ++k) {
            for (int j = 0; j < ImgSizeY; ++j) {
                for (int i = 0; i < ImgSizeX; ++i) {
                    workImage[(k * ImgSizeY + j) * ImgSizeX + i] =
                        (k * ImgSizeY + j) * ImgSizeX + i + 1;
                }
            }
        }
    }
    cl_mem d_workImage;
    d_workImage = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSizeImage, NULL,
        &status);
    chk(status, "clCreateBuffer d_workImage");

    // Copy the initial image values to the device
    status = clEnqueueWriteBuffer(queue, d_workImage, CL_TRUE, 0, dataSizeImage,
        workImage, 0, NULL, NULL);
    chk(status, "clEnqueueWriteBuffer workImage");

    // Double pass kernel
    source = readSource((char*)KERNELBLOCKS);
    program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    chk(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (status != 0) {
        printf("%s failed (%d)\n", "clBuildProgram", status);
        size_t len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char* buffer = (char*)malloc(len);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        printf("%s", buffer);
        exit(1);
    }
    else printf("clBuildProgram succeeded\n");

    cl_kernel kernel_dp;
    kernel_dp = clCreateKernel(program, "double_pass_square_CB", &status);
    chk(status, "clCreateKernel");

    int blockSizeX = BLOCKSIZEX;
    int blockSizeY = BLOCKSIZEY;
    int blockSizeZ = BLOCKSIZEZ;
    size_t modifWidth = ImgSizeX / blockSizeX;
    size_t modifHeight = ImgSizeY / blockSizeY;
    size_t modifDepth = ImgSizeZ / blockSizeZ;
    size_t dataSizeModif = modifWidth * modifHeight * modifDepth * sizeof(unsigned int);
    unsigned int* workModif = NULL;
    workModif = (unsigned int*)malloc(dataSizeModif);
    cl_mem d_modif;
    d_modif = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSizeModif, NULL,
        &status);
    chk(status, "clCreateBuffer");
    status = clEnqueueWriteBuffer(queue, d_modif, CL_TRUE, 0, dataSizeModif,
        workModif, 0, NULL, NULL);
    chk(status, "clEnqueueWriteBuffer");

    status = clSetKernelArg(kernel_dp, 0, sizeof(cl_mem), &d_workImage);
    status |= clSetKernelArg(kernel_dp, 1, sizeof(cl_mem), &d_modif);
    status |= clSetKernelArg(kernel_dp, 2, sizeof(cl_mem), &d_connectImage);
    status |= clSetKernelArg(kernel_dp, 3, sizeof(int), &ImgSizeX);
    status |= clSetKernelArg(kernel_dp, 4, sizeof(int), &ImgSizeY);
    status |= clSetKernelArg(kernel_dp, 5, sizeof(int), &ImgSizeZ);
    status |= clSetKernelArg(kernel_dp, 6, sizeof(int), &blockSizeX);
    status |= clSetKernelArg(kernel_dp, 7, sizeof(int), &blockSizeY);
    status |= clSetKernelArg(kernel_dp, 8, sizeof(int), &blockSizeZ);
    chk(status, "clSetKernelArg");

    // Finally !
    int nbTotalModif = 1;
    int nbPasses = 0;

    printf("Running %ld threads...\n", modifWidth * modifHeight * modifDepth);
    while (nbTotalModif > 0) {

        nbTotalModif = 0;
        size_t globalSize_dp[3] = { modifWidth, modifHeight, modifDepth };
        status = clEnqueueNDRangeKernel(queue, kernel_dp, 3, NULL, globalSize_dp, NULL, 0,
            NULL, NULL);
        chk(status, "clEnqueueNDRange");
        status = clFlush(queue);
        chk(status, "clFlush");
        status = clFinish(queue);
        chk(status, "clFinish");

        status = clEnqueueReadBuffer(queue, d_modif, CL_TRUE, 0, dataSizeModif,
            workModif, 0, NULL, NULL);
        chk(status, "clEnqueueReadBuffer");

        for (int cpt = 0; cpt < modifWidth * modifHeight * modifDepth; cpt++)
            nbTotalModif += workModif[cpt];
        printf("nbTotalModif-dp: %d\n", nbTotalModif);

        printf("Nbpasses: %d\n", ++nbPasses);
    }

    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    puts(buffer);


    // Read the image back to the host
    status = clEnqueueReadBuffer(queue, d_workImage, CL_TRUE, 0, dataSizeImage,
        workImage, 0, NULL, NULL);
    chk(status, "clEnqueueReadBuffer");
    status = clFlush(queue);
    chk(status, "clFlush");
    status = clFinish(queue);
    chk(status, "clFinish");

    // Free the memory on the host for unnecessary objects
    free(workModif);
    free(connectImage);

    return 0;
}