#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <locale.h>
#include <float.h> 
#include <string.h>
#include "utils.cuh"

#include "../dataC/createDiskPixel_localM_fusion2.cuh"
#include "../dataC/doublePassSquare_CB.cuh"

#define M_PI 3.14159265358979323846

//#define GATEDCC
#ifdef  GATEDCC
#define PLATFORM_TO_USE 3	// For gate.dcc.uchile.cl: 3 and 0, 1 or 2 (not 3 !!!)
#define DEVICE_TO_USE 1
#else
#define PLATFORM_TO_USE 0
#define DEVICE_TO_USE 0
#endif

//#define INPUTDISCS "C:/dataGPU/input/Bloque_2_Exp6_P32_5_Simulation_1.csv" //Bloque_2_Exp3_P32_5_Simulation_10.csv" //"C:/dataGPU/input/DFNright1.csv"
#define INPUTDISCS "../dataC/input/synData-model2.csv"
#define INPUTFRAGMENTATION "../dataC/input/fragmentationParameters.csv"
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


#define OUTPUTRESULTS "../dataC/output/results.csv"
#define OUTPUTREGIONS "../dataC/output/regions"
#define OUTPUTCONNECT "../dataC/output/connectivity.xyz"
#define CSVFILE "../dataC/output/fragCurve"
#define STATSFILE "../dataC/output/stats"
#define HISTFILE "../dataC/output/histogram"

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

    /*
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
    */

    // Create the input and output buffers on the device
    /*
    cl_mem d_inputDisks;
    d_inputDisks = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSizeDisks, NULL,
        &status);
    chk(status, "clCreateBuffer d_inputDisks");
    */
    float* d_inputDisks = nullptr;
    CU_CHECK(cudaMalloc((void**)&d_inputDisks, dataSizeDisks));

    /*cl_mem d_connectImage;
    d_connectImage = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSizeImage, NULL,
        &status);
    */
    size_t dataSizeImage = ImgSizeX * ImgSizeY * ImgSizeZ * sizeof(unsigned int);
    unsigned int* connectImage = (unsigned int*)malloc(dataSizeImage);
    unsigned int* d_connectImage = nullptr;
    CU_CHECK(cudaMalloc((void**)&d_connectImage, dataSizeImage));

    /*
    // Copy the input disks to the device
    status = clEnqueueWriteBuffer(queue, d_inputDisks, CL_TRUE, 0, dataSizeDisks,
        inputDisks, 0, NULL, NULL);
    chk(status, "clEnqueueWriteBuffer inputDisks");
    */
    CU_CHECK(cudaMemcpy(d_inputDisks, inputDisks, dataSizeDisks, cudaMemcpyHostToDevice));

    /*
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
    */

    // Create the kernel object
    /*
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
    */
    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    puts(buffer);


    // Run the first kernel
    /*
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
    */
    dim3 globalSize = dim3(ImgSizeX, ImgSizeY, ImgSizeZ);
    dim3 numBlocks = dim3(WPSIZE, WPSIZE, WPSIZE);
    callCreateDiskPixel_localM_fusion2(globalSize, numBlocks, d_connectImage, d_inputDisks, ImgSizeX, ImgSizeY, nbDisks, MAXDISCSPERBLOCK);
    CU_CHECK(cudaFree(d_inputDisks));

    /*
    status = clEnqueueReadBuffer(queue, d_connectImage, CL_TRUE, 0, dataSizeImage,
        connectImage, 0, NULL, NULL);
    chk(status, "clEnqueueReadBuffer");
    status = clFlush(queue);
    chk(status, "clFlush");
    status = clFinish(queue);
    chk(status, "clFinish");
    */
    CU_CHECK(cudaMemcpy(connectImage, d_connectImage, dataSizeImage, cudaMemcpyDeviceToHost));

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
    free(inputDisks);

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

    /*cl_mem d_workImage;
    d_workImage = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSizeImage, NULL,
        &status);
    chk(status, "clCreateBuffer d_workImage");*/
    unsigned int* d_workImage = nullptr;
    CU_CHECK(cudaMalloc((void**)&d_workImage, dataSizeImage));

    // Copy the initial image values to the device
    /*status = clEnqueueWriteBuffer(queue, d_workImage, CL_TRUE, 0, dataSizeImage,
        workImage, 0, NULL, NULL);
    chk(status, "clEnqueueWriteBuffer workImage");*/
    CU_CHECK(cudaMemcpy(d_workImage, workImage, dataSizeImage, cudaMemcpyHostToDevice));

    // Double pass kernel
    /*source = readSource((char*)KERNELBLOCKS);
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
    chk(status, "clCreateKernel");*/

    int blockSizeX = BLOCKSIZEX;
    int blockSizeY = BLOCKSIZEY;
    int blockSizeZ = BLOCKSIZEZ;
    size_t modifWidth = ImgSizeX / blockSizeX;
    size_t modifHeight = ImgSizeY / blockSizeY;
    size_t modifDepth = ImgSizeZ / blockSizeZ;
    size_t dataSizeModif = modifWidth * modifHeight * modifDepth * sizeof(unsigned int);
    unsigned int* workModif = NULL;
    workModif = (unsigned int*)malloc(dataSizeModif);
    /*cl_mem d_modif;
    d_modif = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSizeModif, NULL,
        &status);
    chk(status, "clCreateBuffer");*/
    unsigned int* d_modif = nullptr;
    CU_CHECK(cudaMalloc((void**)&d_modif, dataSizeModif));
    /*status = clEnqueueWriteBuffer(queue, d_modif, CL_TRUE, 0, dataSizeModif,
        workModif, 0, NULL, NULL);
    chk(status, "clEnqueueWriteBuffer");*/
    CU_CHECK(cudaMemcpy(d_modif, workModif, dataSizeModif, cudaMemcpyHostToDevice));

    /*status = clSetKernelArg(kernel_dp, 0, sizeof(cl_mem), &d_workImage);
    status |= clSetKernelArg(kernel_dp, 1, sizeof(cl_mem), &d_modif);
    status |= clSetKernelArg(kernel_dp, 2, sizeof(cl_mem), &d_connectImage);
    status |= clSetKernelArg(kernel_dp, 3, sizeof(int), &ImgSizeX);
    status |= clSetKernelArg(kernel_dp, 4, sizeof(int), &ImgSizeY);
    status |= clSetKernelArg(kernel_dp, 5, sizeof(int), &ImgSizeZ);
    status |= clSetKernelArg(kernel_dp, 6, sizeof(int), &blockSizeX);
    status |= clSetKernelArg(kernel_dp, 7, sizeof(int), &blockSizeY);
    status |= clSetKernelArg(kernel_dp, 8, sizeof(int), &blockSizeZ);
    chk(status, "clSetKernelArg");*/

    // Finally !
    int nbTotalModif = 1;
    int nbPasses = 0;

    printf("Running %ld threads...\n", (long)(modifWidth * modifHeight * modifDepth));
    while (nbTotalModif > 0) {

        nbTotalModif = 0;
        /*size_t globalSize_dp[3] = {modifWidth, modifHeight, modifDepth};
        status = clEnqueueNDRangeKernel(queue, kernel_dp, 3, NULL, globalSize_dp, NULL, 0,
            NULL, NULL);
        chk(status, "clEnqueueNDRange");
        status = clFlush(queue);
        chk(status, "clFlush");
        status = clFinish(queue);
        chk(status, "clFinish");*/
        dim3 numBlocks = dim3(WPSIZE, WPSIZE, WPSIZE);
        callDoublePassSquare_CB(numBlocks, modifWidth, modifHeight, modifDepth, d_workImage, d_modif, d_connectImage,
            ImgSizeX, ImgSizeY, ImgSizeZ, blockSizeX, blockSizeY, blockSizeZ);

        /*status = clEnqueueReadBuffer(queue, d_modif, CL_TRUE, 0, dataSizeModif,
            workModif, 0, NULL, NULL);
        chk(status, "clEnqueueReadBuffer");*/
        CU_CHECK(cudaMemcpy(workModif, d_modif, dataSizeModif, cudaMemcpyDeviceToHost));

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
    /*status = clEnqueueReadBuffer(queue, d_workImage, CL_TRUE, 0, dataSizeImage,
        workImage, 0, NULL, NULL);
    chk(status, "clEnqueueReadBuffer");*/
    CU_CHECK(cudaMemcpy(workImage, d_workImage, dataSizeImage, cudaMemcpyDeviceToHost));

    /*status = clFlush(queue);
    chk(status, "clFlush");
    status = clFinish(queue);
    chk(status, "clFinish");*/

    // Free the memory on the host for unnecessary objects
    /*free(workModif);
    free(connectImage);*/
    CU_CHECK(cudaFree(d_workImage));
    CU_CHECK(cudaFree(d_modif));
    CU_CHECK(cudaFree(d_connectImage));


    /*********************************************************************************************************/
       // Count the number of regions EXCLUDING 0 (tunnel) and 1 (outer region)
    int nbRegionsTotal = 0;
    for (int k = 0; k < ImgSizeZ; ++k)
        for (int j = 0; j < ImgSizeY; ++j)
            for (int i = 0; i < ImgSizeX; ++i)
                if (workImage[(k * ImgSizeY + j) * ImgSizeX + i] ==
                    1 + (k * ImgSizeY + j) * ImgSizeX + i) {
                    if (WITHTUNNEL) {
                        if ((workImage[(k * ImgSizeY + j) * ImgSizeX + i] != 0) &&
                            (workImage[(k * ImgSizeY + j) * ImgSizeX + i] != 1))
                            nbRegionsTotal++;
                    }
                    else nbRegionsTotal++;
                }
    printf("nbRegions total = %d\n", nbRegionsTotal);

    // Store the id of each region EXCLUDING 0 (tunnel) and 1 (outer region)
    int dataSizeRegions = nbRegionsTotal * sizeof(unsigned int);
    unsigned int* regionsIdAll = (unsigned int*)malloc(dataSizeRegions);
    int countRTotal = 0;
    for (int k = 0; k < ImgSizeZ; ++k)
        for (int j = 0; j < ImgSizeY; ++j)
            for (int i = 0; i < ImgSizeX; ++i)
                if (workImage[(k * ImgSizeY + j) * ImgSizeX + i] ==
                    1 + (k * ImgSizeY + j) * ImgSizeX + i) {
                    if (WITHTUNNEL) {
                        if ((workImage[(k * ImgSizeY + j) * ImgSizeX + i] != 0) &&
                            (workImage[(k * ImgSizeY + j) * ImgSizeX + i] != 1))
                            regionsIdAll[countRTotal++] = 1 + (k * ImgSizeY + j) * ImgSizeX + i;
                    }
                    else regionsIdAll[countRTotal++] = 1 + (k * ImgSizeY + j) * ImgSizeX + i;
                }

    // Array regionsID should be automatically sorted 
    if (countRTotal != nbRegionsTotal) {
        printf("Error: countRTotal != nbRegionsTotal\n");
        exit(1);
    }
    int isSorted = 1;
    for (int i = 0; i < nbRegionsTotal - 1; i++)
        if (regionsIdAll[i] > regionsIdAll[i + 1]) isSorted = 0;
    if (isSorted == 0) {
        printf("Error: isSorted = 0\n");
        exit(1);
    }

    // Mark regions that touch the tunnel
    unsigned int* regionsMark = (unsigned int*)malloc(dataSizeRegions);
    for (int i = 0; i < nbRegionsTotal; i++)
        if (WITHTUNNEL) regionsMark[i] = 0;
        else regionsMark[i] = 1;

    if (WITHTUNNEL) {
        for (int k = 0; k < ImgSizeZ; ++k)
            for (int j = 0; j < ImgSizeY; ++j)
                for (int i = 0; i < ImgSizeX; ++i) {
                    if ((workImage[(k * ImgSizeY + j) * ImgSizeX + i] != 0) &&
                        (workImage[(k * ImgSizeY + j) * ImgSizeX + i] != 1) &&
                        ((((k + 1 == TUNNELMINZ) || (k - 1 == TUNNELMAXZ))
                            && (j >= TUNNELMINY) && (j <= TUNNELMAXY)
                            && (i >= TUNNELMINX) && (i <= TUNNELMAXX)) ||
                            (((j + 1 == TUNNELMINY) || (j - 1 == TUNNELMAXY))
                                && (k >= TUNNELMINZ) && (k <= TUNNELMAXZ)
                                && (i >= TUNNELMINX) && (i <= TUNNELMAXX)) ||
                            (((i + 1 == TUNNELMINX) || (i - 1 == TUNNELMAXX))
                                && (j >= TUNNELMINY) && (j <= TUNNELMAXY)
                                && (k >= TUNNELMINZ) && (k <= TUNNELMAXZ)))
                        ) {
                        int pos = search(regionsIdAll, nbRegionsTotal,
                            workImage[(k * ImgSizeY + j) * ImgSizeX + i]);
                        if (pos == -1) {
                            printf("Region %u not found, exiting\n",
                                workImage[(k * ImgSizeY + j) * ImgSizeX + i]);
                            exit(1);
                        }
                        regionsMark[pos] = 1;
                    }
                }
    }

    // Only keep marked regions
    int nbRegions = 0;
    for (int i = 0; i < nbRegionsTotal; i++)
        if (regionsMark[i] == 1) nbRegions++;
    if (WITHTUNNEL)
        printf("nbRegions touching the tunnel = %d\n", nbRegions);
    dataSizeRegions = nbRegions * sizeof(unsigned int);
    unsigned int* regionsId = (unsigned int*)malloc(dataSizeRegions);
    int countRTunnel = 0;
    for (int i = 0; i < countRTotal; i++)
        if (regionsMark[i] == 1) regionsId[countRTunnel++] = regionsIdAll[i];
    if (countRTunnel != nbRegions) {
        printf("Error: countRTunnel != nbRegions\n");
        exit(1);
    }

    // Compute the size of each region and store max value
    //    (possibly mark irrelevant regions as 1)
    // 08/05/2020: Also compute center of mass
    unsigned int* regionsSize = (unsigned int*)malloc(dataSizeRegions);
    int dataSizeRegionsFloat = nbRegions * sizeof(float);
    float* cMassRegionX = (float*)malloc(dataSizeRegionsFloat);
    float* cMassRegionY = (float*)malloc(dataSizeRegionsFloat);
    float* cMassRegionZ = (float*)malloc(dataSizeRegionsFloat);
    for (int i = 0; i < nbRegions; i++) {
        //printf("%d: %u\n", i, regionsId[i]);
        regionsSize[i] = 0;
        cMassRegionX[i] = 0;
        cMassRegionY[i] = 0;
        cMassRegionZ[i] = 0;
    }
    int posMaxSize = 0;
    for (int k = 0; k < ImgSizeZ; ++k)
        for (int j = 0; j < ImgSizeY; ++j)
            for (int i = 0; i < ImgSizeX; ++i) {
                int pos = -1;
                if (WITHTUNNEL) {
                    if ((workImage[(k * ImgSizeY + j) * ImgSizeX + i] != 0) &&
                        (workImage[(k * ImgSizeY + j) * ImgSizeX + i] != 1)) {
                        pos = search(regionsId, nbRegions,
                            workImage[(k * ImgSizeY + j) * ImgSizeX + i]);
                        if (pos == -1) {
                            workImage[(k * ImgSizeY + j) * ImgSizeX + i] = 1;
                            continue;
                        }
                    }
                }
                else {
                    pos = search(regionsId, nbRegions,
                        workImage[(k * ImgSizeY + j) * ImgSizeX + i]);
                    if (pos == -1) {
                        workImage[(k * ImgSizeY + j) * ImgSizeX + i] = 1;
                        continue;
                    }
                }
                if (pos != -1) {
                    regionsSize[pos]++;
                    cMassRegionX[pos] += i;
                    cMassRegionY[pos] += j;
                    cMassRegionZ[pos] += k;
                    if (regionsSize[pos] > regionsSize[posMaxSize]) posMaxSize = pos;
                }
            }

    // Compute min, mean and center of mass
    int maxSize = regionsSize[posMaxSize];
    double avgSize = 0;
    int minSize = ImgSizeX * ImgSizeX * ImgSizeX;
    for (int i = 0; i < nbRegions; i++) {
        avgSize += regionsSize[i];
        cMassRegionX[i] /= regionsSize[i];
        cMassRegionY[i] /= regionsSize[i];
        cMassRegionZ[i] /= regionsSize[i];
        // Map back to original domain
        cMassRegionX[i] = (cMassRegionX[i] / MULT) - Domain_Shift;
        cMassRegionY[i] = (cMassRegionY[i] / MULT) - Domain_Shift;
        cMassRegionZ[i] = (cMassRegionZ[i] / MULT) - Domain_Shift;
        if (regionsSize[i] < minSize) minSize = regionsSize[i];
    }
    if (nbRegions > 0) avgSize /= nbRegions;
    printf("AvgSize = %f (%f in m^3)\n", avgSize, avgSize * REDUCTION);
    printf("MinSize = %u (%f in m^3)\n", minSize, (minSize * REDUCTION));
    printf("MaxSize = %u (%f in m^3)\n", maxSize, maxSize * REDUCTION);

    // Compute P32 and circular variance
    float P32 = 0;
    float cVar[] = { 0, 0, 0 };
    for (int i = 0; i < nbDisks; i++) {
        float radius = inputDisks[i * 7 + 6] / MULT;
        P32 += M_PI * radius * radius;
        cVar[0] += inputDisks[i * 7 + 3];
        cVar[1] += inputDisks[i * 7 + 4];
        cVar[2] += inputDisks[i * 7 + 5];
    }
    P32 /= (Domain_Size * Domain_Size * Domain_Size);//(ImgSizeX*ImgSizeY*ImgSizeZ);
    float CV = sqrt(cVar[0] * cVar[0] + cVar[1] * cVar[1] + cVar[2] * cVar[2]);
    CV = 1 - (CV / nbDisks);
    for (int i = 0; i < 3; i++) cVar[i] /= nbDisks;
    printf("P32 = %f / CVar = %f\n", P32, CV);

    if (STORESTATS) {
        FILE* f;
        f = fopen(OUTPUTRESULTS, "wb");
        fprintf(f, "NbBlocks;%d\n", nbRegions);
        fprintf(f, "AvgSize;%f\n", avgSize * REDUCTION);
        fprintf(f, "MinSize;%f\n", (minSize * REDUCTION));
        fprintf(f, "MaxSize;%f\n", maxSize * REDUCTION);
        fprintf(f, "P32;%f\n", P32);
        fprintf(f, "cVar;%f\n", CV);
        fclose(f);
        printf("Global stats saved in %s\n", OUTPUTRESULTS);
    }

    if (STOREREGIONSCSV)
    {
        FILE* f;
        char outputRegions[100];
        sprintf(outputRegions, "%s.%d.csv", OUTPUTREGIONS, ImgSizeX);
        f = fopen(outputRegions, "wb");
        fprintf(f, "volume;mass;center of mass X;center of mass Y;center of mass Z\n");
        for (int i = 0; i < nbRegions; i++)
            fprintf(f, "%f;%f;%f;%f;%f\n", regionsSize[i] * REDUCTION,
                regionsSize[i] * REDUCTION * Density,
                cMassRegionX[i], cMassRegionY[i], cMassRegionZ[i]);
        fclose(f);
        printf("Volume, mass and center of mass saved in %s\n", outputRegions);
    }
    unsigned int* closedBlocks = (unsigned int*)malloc(dataSizeRegions);
    // 31/10: If tunnel, find the number of blocks that touch each face
    unsigned int* touchTop = (unsigned int*)malloc(dataSizeRegions);
    unsigned int* touchBottom = (unsigned int*)malloc(dataSizeRegions);
    unsigned int* touchLeft = (unsigned int*)malloc(dataSizeRegions);
    unsigned int* touchRight = (unsigned int*)malloc(dataSizeRegions);
    if (WITHTUNNEL) {
        for (int i = 0; i < nbRegions; i++) {
            touchTop[i] = 0;
            touchBottom[i] = 0;
            touchLeft[i] = 0;
            touchRight[i] = 0;
        }
        for (int k = 0; k < ImgSizeZ; ++k)
            for (int j = 0; j < ImgSizeY; ++j)
                for (int i = 0; i < ImgSizeX; ++i) {
                    if ((workImage[(k * ImgSizeY + j) * ImgSizeX + i] == 0) ||
                        (workImage[(k * ImgSizeY + j) * ImgSizeX + i] == 1))
                        continue;
                    int pos = search(regionsId, nbRegions,
                        workImage[(k * ImgSizeY + j) * ImgSizeX + i]);
                    if (pos == -1) {
                        printf("Region %u not found, exiting\n",
                            workImage[(k * ImgSizeY + j) * ImgSizeX + i]);
                        exit(1);
                    }
                    if ((k + 1 == TUNNELMINZ)
                        && (j >= TUNNELMINY) && (j <= TUNNELMAXY)
                        && (i >= TUNNELMINX) && (i <= TUNNELMAXX))
                        touchBottom[pos] = 1;
                    if ((k - 1 == TUNNELMAXZ)
                        && (j >= TUNNELMINY) && (j <= TUNNELMAXY)
                        && (i >= TUNNELMINX) && (i <= TUNNELMAXX))
                        touchTop[pos] = 1;
                    if ((i + 1 == TUNNELMINX)
                        && (j >= TUNNELMINY) && (j <= TUNNELMAXY)
                        && (k >= TUNNELMINZ) && (k <= TUNNELMAXZ))
                        touchLeft[pos] = 1;
                    if ((i - 1 == TUNNELMAXX)
                        && (j >= TUNNELMINY) && (j <= TUNNELMAXY)
                        && (k >= TUNNELMINZ) && (k <= TUNNELMAXZ))
                        touchRight[pos] = 1;
                }
        int nbTop = 0, nbBottom = 0, nbLeft = 0, nbRight = 0;
        int nbTopLeft = 0, nbBottomLeft = 0, nbTopRight = 0, nbBottomRight = 0;
        for (int i = 0; i < nbRegions; i++) {
            if ((touchTop[i]) && (touchLeft[i])) {
                nbTopLeft++; continue;
            }
            if ((touchTop[i]) && (touchRight[i])) {
                nbTopRight++; continue;
            }
            if ((touchBottom[i]) && (touchLeft[i])) {
                nbBottomLeft++; continue;
            }
            if ((touchBottom[i]) && (touchRight[i])) {
                nbBottomRight++; continue;
            }
            if (touchTop[i]) nbTop++;
            if (touchBottom[i]) nbBottom++;
            if (touchLeft[i]) nbLeft++;
            if (touchRight[i]) nbRight++;
        }
        printf("Top Bottom Left Right: %d %d %d %d\n", nbTop, nbBottom, nbLeft, nbRight);
        printf("TL BL TR BR: %d %d %d %d\n", nbTopLeft, nbBottomLeft, nbTopRight, nbBottomRight);
    }
    // 30/10: If no tunnel, find the regions that touch one border
    else {

        /*
       for (int i = 0; i < nbRegions; i++) closedBlocks[i] = 1;
       for (int k = 0; k < ImgSizeZ; ++k)
          for (int j = 0; j < ImgSizeY; ++j)
             for (int i = 0; i < ImgSizeX; ++i) {
                if ((i == 0) || (i == ImgSizeX-1) || (j == 0) || (j == ImgSizeY-1)
                      || (k == 0) || (k == ImgSizeZ-1)) {
                   int pos = search(regionsId, nbRegions,
                      workImage[(k*ImgSizeY + j)*ImgSizeX + i]);
                   if (pos == -1) {
                      printf("Error: region %d not found when finding closed blocks\n",
                         workImage[(k*ImgSizeY + j)*ImgSizeX + i]);
                      exit(2);
                   }
                   closedBlocks[pos] = 0;
                }
             }
         */

         /// 23/09/2020: mark border blocks
        unsigned int* borderBlocks = (unsigned int*)malloc(dataSizeRegions);
        for (int i = 0; i < nbRegions; i++) borderBlocks[i] = 0;
        for (int k = 0; k < ImgSizeZ; ++k)
            for (int j = 0; j < ImgSizeY; ++j)
                for (int i = 0; i < ImgSizeX; ++i) {
                    if ((i == 0) || (i == ImgSizeX - 1) || (j == 0) || (j == ImgSizeY - 1)
                        || (k == 0) || (k == ImgSizeZ - 1)) {
                        int pos = search(regionsId, nbRegions,
                            workImage[(k * ImgSizeY + j) * ImgSizeX + i]);
                        if (pos == -1) {
                            printf("Error #739: region %d not found when finding closed blocks\n",
                                workImage[(k * ImgSizeY + j) * ImgSizeX + i]);
                            exit(2);
                        }
                        borderBlocks[pos] = 1;
                    }
                }

        /// 23/09/2020: if a border block touches an "active" face, it becomes "closed"
 /// (and non border blocks are automatically "closed")
        for (int i = 0; i < nbRegions; i++) closedBlocks[i] = (borderBlocks[i] + 1) % 2;
        for (int k = 0; k < ImgSizeZ; ++k)
            for (int j = 0; j < ImgSizeY; ++j)
                for (int i = 0; i < ImgSizeX; ++i) {
                    if (((i == 0) && (FACELEFT)) ||
                        ((i == ImgSizeX - 1) && (FACERIGHT)) ||
                        ((j == 0) && (FACEDOWN)) ||
                        ((j == ImgSizeY - 1) && (FACEUP)) ||
                        ((k == 0) && (FACEFRONT)) ||
                        ((k == ImgSizeZ - 1) && (FACEBACK))) {
                        int pos = search(regionsId, nbRegions,
                            workImage[(k * ImgSizeY + j) * ImgSizeX + i]);
                        if (pos == -1) {
                            printf("Error #736: region %d not found when finding closed blocks\n",
                                workImage[(k * ImgSizeY + j) * ImgSizeX + i]);
                            exit(2);
                        }
                        closedBlocks[pos] = 1;
                    }
                }

        ///// 23/09/2020: store border regions (ie non-closed)
        if ((REMOVENONCLOSEDBLOCKS) && (STOREREGIONS)) {
            unsigned int* regionsColor = (unsigned int*)malloc(dataSizeRegions);
            for (int i = 0; i < nbRegions; i++) regionsColor[i] = 2 + rand() % 1022;
            char outputRegions[100];
            sprintf(outputRegions, "%s.%d-border.xyz", OUTPUTREGIONS, ImgSizeX);
            storeXYZBorderRegions(workImage, outputRegions, ImgSizeX, ImgSizeY, ImgSizeZ,
                nbRegions, regionsId, closedBlocks, regionsColor);
            printf("Border regions saved in %s\n", outputRegions);
        }


        double sizeClosedBlocks = 0;
        for (int i = 0; i < nbRegions; i++)
            if (closedBlocks[i] == 1) sizeClosedBlocks += regionsSize[i];
        printf("---- sizeClosedBlocks = %d (%f in m^3)\n", (int)sizeClosedBlocks,
            sizeClosedBlocks * REDUCTION);
        printf("---- totalSize = %d (%f in m^3)\n", (int)(ImgSizeX * ImgSizeY * ImgSizeZ),
            (double)(ImgSizeX * ImgSizeY * ImgSizeZ) * REDUCTION);
        printf("---- Percentage: %f\n", sizeClosedBlocks / (ImgSizeX * ImgSizeY * ImgSizeZ));
        sizeClosedBlocks = 0;
        for (int k = 0; k < ImgSizeZ; ++k)
            for (int j = 0; j < ImgSizeY; ++j)
                for (int i = 0; i < ImgSizeX; ++i) {
                    int pos = search(regionsId, nbRegions,
                        workImage[(k * ImgSizeY + j) * ImgSizeX + i]);
                    if (pos == -1) {
                        printf("Error: region %d not found when finding closed blocks\n",
                            workImage[(k * ImgSizeY + j) * ImgSizeX + i]);
                        exit(2);
                    }
                    if ((REMOVENONCLOSEDBLOCKS) && (closedBlocks[pos] == 0))
                        workImage[(k * ImgSizeY + j) * ImgSizeX + i] = 0;
                    else sizeClosedBlocks++;
                }
        printf("---- DoubleCheck: sizeClosedBlocks = %d (%f in m^3)\n", (int)sizeClosedBlocks,
            sizeClosedBlocks * REDUCTION);
    }

    if (STOREREGIONS) {
        unsigned int* regionsColor = (unsigned int*)malloc(dataSizeRegions);
        for (int i = 0; i < nbRegions; i++) regionsColor[i] = 2 + rand() % 1022;
        for (int k = 0; k < ImgSizeZ; ++k)
            for (int j = 0; j < ImgSizeY; ++j)
                for (int i = 0; i < ImgSizeX; ++i) {
                    // Tunnel
                    if ((WITHTUNNEL) && (workImage[(k * ImgSizeY + j) * ImgSizeX + i] == 0)) {
                        if ((k != TUNNELMINZ) && (k != TUNNELMAXZ) &&
                            (j != TUNNELMINY) && (j != TUNNELMAXY) &&
                            (i != TUNNELMINX) && (i != TUNNELMAXX))
                            workImage[(k * ImgSizeY + j) * ImgSizeX + i] = 1;
                        continue;
                    }
                    if ((WITHTUNNEL) && (workImage[(k * ImgSizeY + j) * ImgSizeX + i] == 1))
                        continue;
                    if ((REMOVENONCLOSEDBLOCKS) && (workImage[(k * ImgSizeY + j) * ImgSizeX + i] == 0))
                        continue;
                    int pos = search(regionsId, nbRegions,
                        workImage[(k * ImgSizeY + j) * ImgSizeX + i]);
                    if (pos == -1) {
                        printf("Error: region %d not found when storing regions\n",
                            workImage[(k * ImgSizeY + j) * ImgSizeX + i]);
                        exit(2);
                        workImage[(k * ImgSizeY + j) * ImgSizeX + i] = 1;
                        continue;
                    }
                    // Keep only max region
                    //if (pos != posMaxSize) {
                    //   workImage[(k*ImgSizeY + j)*ImgSizeX + i] = 1;   
                    //   continue;
                    //}
                    // Keep only size 1 regions
                    //if (regionsSize[pos] > 1) {
                    //   workImage[(k*ImgSizeY + j)*ImgSizeX + i] = 1;   
                    //   continue;
                    //}

                    // Keep only top right regions
                    // 20/12/21: Now extended to faces Up, Right, Left and Down
//                    if ((WITHTUNNEL) && (SHOWONLYTOPRIGHTBLOCKS)) {
//                        if ((touchTop[pos] == 1) && (touchRight[pos] == 1))
                    if (WITHTUNNEL) {
                        if (
                            ((touchTop[pos] == 1) && (FACEUP == 1)) ||
                            ((touchRight[pos] == 1) && (FACERIGHT == 1)) ||
                            ((touchBottom[pos] == 1) && (FACEDOWN == 1)) ||
                            ((touchLeft[pos] == 1) && (FACELEFT == 1))
                            )
                            workImage[(k * ImgSizeY + j) * ImgSizeX + i] = regionsColor[pos];
                        else
                            workImage[(k * ImgSizeY + j) * ImgSizeX + i] = 1;
                    }
                    else {
                        workImage[(k * ImgSizeY + j) * ImgSizeX + i] = regionsColor[pos];
                    }

                }

        //storeXYZregions(workImage, OUTPUTREGIONS, imageHeight, imageWidth, imageDepth, 
        //      imageHeight * imageWidth * imageDepth);
        char outputRegions[100];
        if (WITHTUNNEL) {
            //sprintf(outputRegions, "%s.%s.%d-tunnel.xyz", INPUTDISCS, OUTPUTREGIONS, ImgSizeX);
            //20/12/21: fixed bug for filename
            sprintf(outputRegions, "%s.%d-tunnel.xyz", OUTPUTREGIONS, ImgSizeX);
            storeXYZnonOnevalues(workImage, outputRegions, ImgSizeX, ImgSizeY, ImgSizeZ);
        }
        else {
            if (REMOVENONCLOSEDBLOCKS)
                sprintf(outputRegions, "%s.%d.CB.xyz", OUTPUTREGIONS, ImgSizeX);
            else
                //sprintf(outputRegions, "%s.%s.%d.xyz", INPUTDISCS, OUTPUTREGIONS, ImgSizeX); 
                sprintf(outputRegions, "%s.%d.xyz", OUTPUTREGIONS, ImgSizeX);
            storeXYZnonZerovalues(workImage, outputRegions, ImgSizeX, ImgSizeY, ImgSizeZ);
        }
        printf("Regions saved in %s\n", outputRegions);
        //http://skmf.eu/tutorial-for-ovito/
    }

    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    puts(buffer);
    if (REMOVENONCLOSEDBLOCKS) {

        int nbClosedBlocks = 0;
        maxSize = 0;
        avgSize = 0;
        minSize = ImgSizeX * ImgSizeX * ImgSizeX;
        double deviationClosed = 0;
        for (int i = 0; i < nbRegions; i++) {
            if (closedBlocks[i] == 0) continue;
            nbClosedBlocks++;
            avgSize += regionsSize[i];
            if (regionsSize[i] < minSize) minSize = regionsSize[i];
            if (regionsSize[i] > maxSize) maxSize = regionsSize[i];
        }
        printf("---- nbClosedBlocks = %d\n", nbClosedBlocks);
        if (nbClosedBlocks > 0) {
            avgSize /= nbClosedBlocks;
            for (int i = 0; i < nbRegions; i++)
                if (closedBlocks[i] == 1)
                    deviationClosed += (regionsSize[i] - avgSize) * (regionsSize[i] - avgSize);
            deviationClosed /= nbClosedBlocks;
            deviationClosed = sqrt(deviationClosed);
            printf("AvgSizeClosed = %f (%f in m^3)\n", avgSize, avgSize * REDUCTION);
            printf("DeviationClosed = %f (%f in m^3)\n", deviationClosed, deviationClosed * REDUCTION);
            printf("MinSizeClosed = %u (%f in m^3)\n", minSize, (minSize * REDUCTION));
            printf("MaxSizeClosed = %u (%f in m^3)\n", maxSize, maxSize * REDUCTION);
        }

        if (STORESTATS) {
            FILE* f;
            f = fopen(OUTPUTRESULTS, "ab");
            fprintf(f, "NbClosedBlocks;%d\n", nbClosedBlocks);
            fprintf(f, "AvgSizeClosed;%f\n", avgSize * REDUCTION);
            fprintf(f, "DeviationClosed;%f\n", deviationClosed * REDUCTION);
            fprintf(f, "MinSizeClosed;%f\n", (minSize * REDUCTION));
            fprintf(f, "MaxSize;%f\n", maxSize * REDUCTION);
            printf("Global stats updated in %s\n", OUTPUTRESULTS);
        }
    }

    if (STOREREGIONSCSV) {
        FILE* f;
        char outputRegions[100];
        sprintf(outputRegions, "%s.%d-closed.csv", OUTPUTREGIONS, ImgSizeX);
        f = fopen(outputRegions, "wb");
        fprintf(f, "volume;mass;center of mass X;center of mass Y;center of mass Z\n");
        for (int i = 0; i < nbRegions; i++)
            if (closedBlocks[i] == 1)
                fprintf(f, "%f;%f;%f;%f;%f\n", regionsSize[i] * REDUCTION,
                    regionsSize[i] * REDUCTION * Density,
                    cMassRegionX[i], cMassRegionY[i], cMassRegionZ[i]);
        fclose(f);
        printf("Volume, mass and center of mass saved in %s\n", outputRegions);
    }
    // 06/10/2020: store CSV and global stats for border regions
    if (REMOVENONCLOSEDBLOCKS) {

        int nbBorderBlocks = 0;
        maxSize = 0;
        avgSize = 0;
        minSize = ImgSizeX * ImgSizeX * ImgSizeX;
        double deviationBorder = 0;
        for (int i = 0; i < nbRegions; i++) {
            if (closedBlocks[i] == 1) continue;
            nbBorderBlocks++;
            avgSize += regionsSize[i];
            if (regionsSize[i] < minSize) minSize = regionsSize[i];
            if (regionsSize[i] > maxSize) maxSize = regionsSize[i];
        }
        printf("---- nbBorderBlocks = %d\n", nbBorderBlocks);
        if (nbBorderBlocks > 0) {
            avgSize /= nbBorderBlocks;
            for (int i = 0; i < nbRegions; i++)
                if (closedBlocks[i] == 0)
                    deviationBorder += (regionsSize[i] - avgSize) * (regionsSize[i] - avgSize);
            deviationBorder /= nbBorderBlocks;
            deviationBorder = sqrt(deviationBorder);
            printf("AvgSizeBorder = %f (%f in m^3)\n", avgSize, avgSize * REDUCTION);
            printf("DeviationBorder = %f (%f in m^3)\n", deviationBorder, deviationBorder * REDUCTION);
            printf("MinSizeBorder = %u (%f in m^3)\n", minSize, (minSize * REDUCTION));
            printf("MaxSizeBorder = %u (%f in m^3)\n", maxSize, maxSize * REDUCTION);
        }

        if (STORESTATS) {
            FILE* f;
            f = fopen(OUTPUTRESULTS, "ab");
            fprintf(f, "NbBorderBlocks;%d\n", nbBorderBlocks);
            fprintf(f, "AvgSizeBorder;%f\n", avgSize * REDUCTION);
            fprintf(f, "DeviationBorder;%f\n", deviationBorder * REDUCTION);
            fprintf(f, "MinSizeBorder;%f\n", (minSize * REDUCTION));
            fprintf(f, "MaxSizeBorder;%f\n", maxSize * REDUCTION);
            printf("Global stats updated in %s\n", OUTPUTRESULTS);
        }

        if (STOREREGIONSCSV) {
            FILE* f;
            char outputRegions[100];
            sprintf(outputRegions, "%s.%d-border.csv", OUTPUTREGIONS, ImgSizeX);
            f = fopen(outputRegions, "wb");
            fprintf(f, "volume;mass;center of mass X;center of mass Y;center of mass Z\n");
            for (int i = 0; i < nbRegions; i++)
                if (closedBlocks[i] == 0)
                    fprintf(f, "%f;%f;%f;%f;%f\n", regionsSize[i] * REDUCTION,
                        regionsSize[i] * REDUCTION * Density,
                        cMassRegionX[i], cMassRegionY[i], cMassRegionZ[i]);
            fclose(f);
            printf("Volume, mass and center of mass saved in %s\n", outputRegions);
        }
    }

    return 0;
}
