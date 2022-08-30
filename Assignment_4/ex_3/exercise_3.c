#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>

#define ARRAY_SZ 1000000000

double time_now()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
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

void init_array(int size, float *arr) {
    for (int i = 0; i < size; ++i) {
        arr[i] = 100.0f * (rand() / (float) RAND_MAX);
    }
}

void saxpy_cpu(int size, float *x, float *y, float a) {
    for (int i = 0; i < size; ++i) {
        y[i] += a * x[i];
    }
}

void saxpy_gpu(int size, float * restrict x, float * restrict y, float a) {
    #pragma acc parallel loop copyin(x[0:size]), copy(y[0:size])
    for (int i = 0; i < size; ++i) {
        y[i] += a * x[i];
    }
}

int main() {
    float *x = malloc(ARRAY_SZ * sizeof(float));
    float *y = malloc(ARRAY_SZ * sizeof(float));

    float *y_cpu = malloc(ARRAY_SZ * sizeof(float));

    init_array(ARRAY_SZ, x);
    init_array(ARRAY_SZ, y);

    memcpy(y_cpu, y, ARRAY_SZ * sizeof(float));

    double t1 = time_now();
    saxpy_gpu(ARRAY_SZ, x, y, 1.0);
    double t2 = time_now();
    saxpy_cpu(ARRAY_SZ, x, y_cpu, 1.0);
    double t3 = time_now();

    compare_results(ARRAY_SZ, y_cpu, y);

    printf("CPU time: %f s, GPU time %f s\n", t3 - t2, t2 - t1);

    free(x);
    free(y);
    free(y_cpu);

    return 0;
}

