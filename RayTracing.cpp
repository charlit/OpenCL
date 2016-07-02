#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/gl.h>
#include <ctime>
#include <windows.h>
#include <CL/cl.h>
#include "ocl_macro.h"

#include "./common/cpu_bitmap.h"

//Common defines
#define VENDOR_NAME "AMD"
#define DEVICE_TYPE CL_DEVICE_TYPE_CPU

#define DIM 800
#define nbSphere 8

typedef struct sphere {
	int r, b, g;
	int x, y, z;
	float rayon;
} sphere;

sphere sphereArray[nbSphere];

const char *sphere_kernel =
"#define DIM 800                                                        \n"
"#define nbSphere 8                                                   \n"
"#define shadow_factor 3                                                \n"
"typedef struct sphere {                                                \n"
"	int r, b, g;                                                        \n"
"	int x, y, z;                                                        \n"
"	float rayon;                                                        \n"
"} sphere;                                                              \n"
"                                                                       \n"
" //Envoie de rayons sur la sphère                                      \n"
"float spherehit(sphere _sphere, int ox, int oy){	                    \n"
"	float out = -1;                                                     \n"
"	float distX = (ox - _sphere.x) * (ox - _sphere.x);                  \n"
"	float distY = (oy - _sphere.y) * (oy - _sphere.y);                  \n"
"	float dist = sqrt(distX + distY);                                   \n"
"	if (dist <= (float) _sphere.rayon){                                 \n"
"	    float xca = (_sphere.x - ox) * (_sphere.x - ox);                \n"
"	    float yca = ( _sphere.y - oy) * ( _sphere.y - oy);              \n"
"       out = sqrt(xca + yca);                                          \n"
"	}                                                                   \n"
"	return out;                                                         \n"
"}                                                                      \n"
"                                                                       \n"
"__kernel                                                               \n"
"void kernelcpu(__global unsigned char *ptr, __global sphere *sphereArray){ \n"
" int i = get_global_id(0);                                             \n"
" int j = get_global_id(1);                                             \n"
"			float hit = -1;                                             \n"
"			int color0 = 0;                                             \n"
"			int color1 = 0;                                             \n"
"			int color2 = 0;                                             \n"
"			for (int z = 0; z < nbSphere; z++)                          \n"
"			{                                                           \n"
"				hit = spherehit(sphereArray[z],i,j);                    \n"
"				if (hit != -1){  // Si il y a une sphere                                    \n"
"					if ((sphereArray[z].r - (hit * shadow_factor)) - (sphereArray[z].z) > 0 && (sphereArray[z].r - (hit * shadow_factor)) - (sphereArray[z].z) < 255) \n"
"						color0 = (sphereArray[z].r - (hit * shadow_factor)) - (sphereArray[z].z); \n"
"					if ((sphereArray[z].g - (hit * shadow_factor)) - (sphereArray[z].z) > 0 && (sphereArray[z].g - (hit * shadow_factor)) - (sphereArray[z].z) < 255) \n"
"						color1 = (sphereArray[z].g - (hit * shadow_factor)) - (sphereArray[z].z); \n"
"					if ((sphereArray[z].b - (hit * shadow_factor)) - (sphereArray[z].z) > 0 && (sphereArray[z].b - (hit * shadow_factor)) - (sphereArray[z].z) < 255) \n"
"						color2 = (sphereArray[z].b - (hit * shadow_factor)) - (sphereArray[z].z); \n"
"				}                                                       \n"
"			}                                                           \n"
"			int index = (j + (i * DIM)) * 4;                            \n"
"			ptr[index] = color0;                                        \n"
"			ptr[index + 1] = color1;                                    \n"
"			ptr[index + 2] = color2;                                    \n"
"			ptr[index + 3] = 255;                                      \n"
"}                                                                      \n";


int main(void) {
	srand(time(NULL));

	CPUBitmap bitmap(DIM, DIM);
	unsigned char *ptr = bitmap.get_ptr();

	cl_int clStatus; //Keeps track of the error values returned.

	// Get platform and device information
	cl_platform_id * platforms = NULL;

	// Set up the Platform. Take a look at the MACROs used in this file.
	// These are defined in common/ocl_macros.h
	OCL_CREATE_PLATFORMS(platforms);

	// Get the devices list and choose the type of device you want to run on
	cl_device_id *device_list = NULL;
	OCL_CREATE_DEVICE(platforms[1], DEVICE_TYPE, device_list);

	// Create OpenCL context for devices in device_list
	cl_context context;
	cl_context_properties props[3] =
	{
		CL_CONTEXT_PLATFORM,(cl_context_properties)platforms[0],0
	};

	// An OpenCL context can be associated to multiple devices, either CPU or GPU
	// based on the value of DEVICE_TYPE defined above.
	context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);
	LOG_OCL_ERROR(clStatus, "clCreateContext Failed...");

	// Create a command queue for the first device in device_list
	cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);
	LOG_OCL_ERROR(clStatus, "clCreateCommandQueue Failed...");

	long datasize = bitmap.image_size();

	for (int i = 0; i < nbSphere; i++)
	{
	    int randX = rand() % DIM;
        int randY = rand() % DIM;
        int randZ = rand() % 100;

        int randR = rand() % 255;
        int randG = rand() % 255;
        int randB = rand() % 255;

        int randRa = rand() % 255;

        sphere sphere = { randR, randG, randB, randX, randY, randZ, randRa };

		sphereArray[i] = sphere;
	}

	size_t arrsize = sizeof(sphere) * nbSphere;

	// Create memory buffers on the device for each vector
	cl_mem ptr_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &clStatus);

	cl_mem arr_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, arrsize, NULL, &clStatus);

	// Copy the Buffer A and B to the device. We do a blocking write to the device buffer.
	clStatus = clEnqueueWriteBuffer(command_queue, ptr_clmem, CL_TRUE, 0, datasize, ptr, 0, NULL, NULL);
	LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer Failed...");

	clStatus = clEnqueueWriteBuffer(command_queue, arr_clmem, CL_TRUE, 0, arrsize, sphereArray, 0, NULL, NULL);
	LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer Failed...");

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&sphere_kernel, NULL, &clStatus);
	LOG_OCL_ERROR(clStatus, "clCreateProgramWithSource Failed...");

	// Build the program
	clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
	if (clStatus != CL_SUCCESS)
		LOG_OCL_COMPILER_ERROR(program, device_list[0]);

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "kernelcpu", &clStatus);

	// Set the arguments of the kernel. Take a look at the kernel definition in sum_event
	// variable. First parameter is a constant and the other three are buffers.
	clStatus |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&ptr_clmem);
	LOG_OCL_ERROR(clStatus, "clSetKernelArg Failed...");

	clStatus |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&arr_clmem);
	LOG_OCL_ERROR(clStatus, "clSetKernelArg Failed...");

	// Execute the OpenCL kernel on the list
	size_t global_size[2] = { DIM, DIM };
	size_t local_size = NULL;
	cl_event sum_event;
	clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, NULL, 0, NULL, &sum_event);
	LOG_OCL_ERROR(clStatus, "clEnqueueNDRangeKernel Failed...");

	// Read the memory buffer C_clmem on the device to the host allocated buffer C
	// This task is invoked only after the completion of the event sum_event
	clStatus = clEnqueueReadBuffer(command_queue, ptr_clmem, CL_TRUE, 0, datasize, ptr, 1, &sum_event, NULL);
	LOG_OCL_ERROR(clStatus, "clEnqueueReadBuffer Failed...");

	// Clean up and wait for all the comands to complete.
	clStatus = clFinish(command_queue);

	bitmap.display_and_exit();

	// Finally release all OpenCL objects and release the host buffers.
	clStatus = clReleaseKernel(kernel);
	clStatus = clReleaseProgram(program);
	clStatus = clReleaseMemObject(ptr_clmem);
	clStatus = clReleaseMemObject(arr_clmem);
	clStatus = clReleaseCommandQueue(command_queue);
	clStatus = clReleaseContext(context);
	free(ptr);
	free(platforms);
	free(device_list);

	system("pause");
	return 0;
}
