#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include "ocl_macros.h"

#include "../../common/cpu_bitmap.h"


#define DIM 1024
//Common defines 
#define VENDOR_NAME "AMD"
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU

const char *julia_kernel =
"#define DIM 1024															\n"
"typedef struct cuComplex {													\n"
"	float   r;																\n"
"	float   i;																\n"
"} cuComplex;																\n"
"																			\n"
"cuComplex createComplex(float _r, float _i) {								\n"
"	struct cuComplex tmp;													\n"
"	tmp.r = _r;																\n"
"	tmp.i = _i;																\n"
"																			\n"
"	return tmp;																\n"
"}																			\n"
"																			\n"
"float magnitude2(cuComplex z) {											\n"
"	return z.r * z.r + z.i * z.i;											\n"
"}																			\n"
"																			\n"
"cuComplex multiply(cuComplex a, cuComplex b) {								\n"
"	return createComplex(a.r * b.r - a.i * b.i, a.i * b.r + a.r * b.i);		\n"
"}																			\n"
"																			\n"
"cuComplex add(cuComplex a, cuComplex b) {									\n"
"	return createComplex(a.r + b.r, a.i + b.i);								\n"
"}																			\n"
"																			\n"
"																			\n"
"int julia(int x, int y) {											\n"
"	const float scale = 1.5;												\n"
"	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);					\n"
"	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);					\n"
"																			\n"
"	cuComplex c = createComplex(-0.8, 0.156);								\n"
"	cuComplex a = createComplex(jx, jy);									\n"
"																			\n"
"	int i = 0;																\n"
"	for (i = 0; i<200; i++) {												\n"
"		a = add(multiply(a, a), c);											\n"
"		if (magnitude2(a) > 1000)											\n"
"			return 0;											\n"
"	}															\n"
"																\n"
"	return 1;													\n"
"}																\n"
"																\n"
"__kernel void julia_kernel( __global unsigned char *ptr){ 		\n"
"			int y = get_global_id(1);							\n"
"			int x = get_global_id(0);							\n"
"			int offset = x + y * DIM;							\n"
"		    int juliaValue = julia(x, y);						\n"
"			ptr[offset * 4 + 0] = 255 * juliaValue;				\n"
"			ptr[offset * 4 + 1] = 0;							\n"
"			ptr[offset * 4 + 2] = 0;							\n"
"			ptr[offset * 4 + 3] = 255;							\n"	
"																\n"
"																\n"
"}																\n";

int main(void) {
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
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platforms[0],
		0
	};
	// An OpenCL context can be associated to multiple devices, either CPU or GPU
	// based on the value of DEVICE_TYPE defined above.
	context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);
	LOG_OCL_ERROR(clStatus, "clCreateContext Failed...");

	// Create a command queue for the first device in device_list
	cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);
	LOG_OCL_ERROR(clStatus, "clCreateCommandQueue Failed...");

	// Create memory buffer
	cl_mem julia_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, bitmap.image_size() * sizeof(unsigned char), NULL, &clStatus);

	
	for (int u = 0; u < DIM * DIM; u++) {
		ptr[u * 4] = 0;
		ptr[u * 4 + 1] = 0;
		ptr[u * 4 + 2] = 0;
		ptr[u * 4 + 3] = 0;
	}	

	// Enqueue a write buffer
	clStatus = clEnqueueWriteBuffer(command_queue, julia_clmem, CL_TRUE, 0, bitmap.image_size() * sizeof(unsigned char), ptr, 0, NULL, NULL);
	LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer Failed...");

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&julia_kernel, NULL, &clStatus);
	LOG_OCL_ERROR(clStatus, "clCreateProgramWithSource Failed...");

	// Build the program
	clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
	if (clStatus != CL_SUCCESS)
		LOG_OCL_COMPILER_ERROR(program, device_list[0]);

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "julia_kernel", &clStatus);

	clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&julia_clmem);
	LOG_OCL_ERROR(clStatus, "clSetKernelArg Failed...");
	
	// Execute the OpenCL kernel on the list
	size_t global_size[2] = { DIM, DIM };
	size_t local_size[2] = { 1, 1 };
	cl_event julia_event;
	clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &julia_event);
	LOG_OCL_ERROR(clStatus, "clEnqueueNDRangeKernel Failed...");

	// Read the memory buffer C_clmem on the device to the host allocated buffer
	unsigned char *A = (unsigned char*)malloc(bitmap.image_size() * sizeof(unsigned char));
	clStatus = clEnqueueReadBuffer(command_queue, julia_clmem, CL_TRUE, 0, bitmap.image_size() * sizeof(unsigned char), ptr, 1, &julia_event, NULL);
	LOG_OCL_ERROR(clStatus, "clEnqueueReadBuffer Failed...");
	
	// Clean up and wait for all the comands to complete.
	clStatus = clFinish(command_queue);

	// Finally release all OpenCL objects and release the host buffers.
	clStatus = clReleaseKernel(kernel);
	clStatus = clReleaseProgram(program);
	clStatus = clReleaseMemObject(julia_clmem);
	clStatus = clReleaseCommandQueue(command_queue);
	clStatus = clReleaseContext(context);	
	free(platforms);
	free(device_list);

	bitmap.display_and_exit();

	return 0;
}