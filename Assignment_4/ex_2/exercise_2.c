#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <CL/cl.h>
#include <sys/time.h>

#define CHK_ERROR(err) if ((err) != CL_SUCCESS) fprintf(stderr, "Error:%s, line %d\n", clGetErrorString(err), __LINE__);

#define ARRAY_SZ 100000000

const char *clGetErrorString(int);

const char *saxpy_kernel =
"__kernel void saxpy(int size, __global float *x, __global float *y, float a) {\n"
"   int thread_idx = get_global_id(0);\n"
"   if (thread_idx >= size) return;\n"
"   y[thread_idx] += a * x[thread_idx];}\n";

double time_now()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


void saxpy_cpu(int size, const float *x, float *y, float a) {
    for (int i = 0; i < size; ++i)
        y[i] += a * x[i];
}

void init_array(int size, float *arr) {
    for (int i = 0; i < size; ++i) {
        arr[i] = 100.0f * (rand() / (float) RAND_MAX);
    }
}

bool compare_results(int size, float *truth, float *res) {
    for (int i = 0; i < size; ++i) {
        const float diff = fabs(truth[i] - res[i]);
        if (diff > 1e-6) {
            printf("Arrays differ! idx: %d, diff: %f, true val: %f, res: %f\n", i, diff, truth[i], res[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    cl_platform_id *platforms;
    cl_uint n_platforms;

    cl_int err = clGetPlatformIDs(0, NULL, &n_platforms); CHK_ERROR(err);
    platforms = malloc(n_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(n_platforms, platforms, NULL); CHK_ERROR(err);

    cl_device_id *devices;
    cl_uint n_devices;
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices); CHK_ERROR(err);
    devices = malloc(n_devices * sizeof(cl_device_id));
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, n_devices, devices, NULL); CHK_ERROR(err);
    
    cl_context context = clCreateContext(NULL, n_devices, devices, NULL, NULL, &err); CHK_ERROR(err);
    cl_command_queue cmd_queue = clCreateCommandQueue(context, devices[0], 0, &err); CHK_ERROR(err);

    cl_mem x_device = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * ARRAY_SZ, NULL, &err); CHK_ERROR(err);
    cl_mem y_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * ARRAY_SZ, NULL, &err); CHK_ERROR(err);

    float *x = malloc(ARRAY_SZ * sizeof(float));
    float *y = malloc(ARRAY_SZ * sizeof(float));
    float *y_res_cpu = malloc(ARRAY_SZ * sizeof(float));
    float *y_res_gpu = malloc(ARRAY_SZ * sizeof(float));

    init_array(ARRAY_SZ, x);
    init_array(ARRAY_SZ, y);
    memcpy(y_res_cpu, y, sizeof(float) * ARRAY_SZ);

    err = clEnqueueWriteBuffer(cmd_queue, x_device, CL_TRUE, 0, sizeof(float) * ARRAY_SZ, x, 0, NULL, NULL); CHK_ERROR(err);
    err = clEnqueueWriteBuffer(cmd_queue, y_device, CL_TRUE, 0, sizeof(float) * ARRAY_SZ, y, 0, NULL, NULL); CHK_ERROR(err);

    cl_program program = clCreateProgramWithSource(context, 1, &saxpy_kernel, NULL, &err); CHK_ERROR(err);
    err = clBuildProgram(program, 1, devices, NULL, NULL, NULL); CHK_ERROR(err);
    cl_kernel kernel = clCreateKernel(program, "saxpy", &err); CHK_ERROR(err);

    const size_t ndims = 1;
    const size_t wg_size = 256;
    const size_t num_workitems = ARRAY_SZ + wg_size - (ARRAY_SZ % wg_size);

    float a = 1.0;

    const int array_size = ARRAY_SZ;
    double t1 = time_now();
    err = clSetKernelArg(kernel, 0, sizeof(int), (void*) &array_size); CHK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &x_device); CHK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &y_device); CHK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(float), (void*) &a); CHK_ERROR(err);

    err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &num_workitems, &wg_size, 0, NULL, NULL); CHK_ERROR(err);
    err = clFinish(cmd_queue); CHK_ERROR(err);
    double t2 = time_now();
    err = clEnqueueReadBuffer(cmd_queue, y_device, CL_TRUE, 0, sizeof(float) * ARRAY_SZ, y_res_gpu, 0, NULL, NULL); CHK_ERROR(err);

    err = clFinish(cmd_queue); CHK_ERROR(err);
    double t3 = time_now();
    saxpy_cpu(ARRAY_SZ, x, y_res_cpu, a);
    double t4 = time_now();
    compare_results(ARRAY_SZ, y_res_cpu, y_res_gpu);

    printf("CPU time: %f s, GPU time %f s\n", t4 - t3, t2 - t1);

    err = clReleaseCommandQueue(cmd_queue); CHK_ERROR(err);
    err = clReleaseContext(context); CHK_ERROR(err);

    free(platforms);
    free(devices);

    free(x);
    free(y);
    free(y_res_cpu);
    free(y_res_gpu);

    return 0;
}


// The source for this particular version is from: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* clGetErrorString(int errorCode) {
  switch (errorCode) {
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
  case -69: return "CL_INVALID_PIPE_SIZE";
  case -70: return "CL_INVALID_DEVICE_QUEUE";
  case -71: return "CL_INVALID_SPEC_ID";
  case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
  case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
  case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
  case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
  case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
  case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
  case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
  case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
  case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
  case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
  case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
  case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
  case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
  case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
  case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
  case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
  case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
  case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
  default: return "CL_UNKNOWN_ERROR";
  }
}
