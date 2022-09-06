#ifndef UTILS_H
#define UTILS_H

#include "raytracing.h"

#include <iostream>

#define cuda_err(call) cuda_check((call), __FILE__, __LINE__)
inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "Cuda error: " << cudaGetErrorString(err)
                  << " at line: " << line << " in " << file << "\n";
        exit(err);
    }
}

#define MAKE_OP(op) \
inline __device__ float3 operator op (const float3& lhs, const float3& rhs) { \
    float3 res; \
    res.x = lhs.x op rhs.x; \
    res.y = lhs.y op rhs.y; \
    res.z = lhs.z op rhs.z; \
    return res; \
}

MAKE_OP(+)
MAKE_OP(-)
MAKE_OP(*)
MAKE_OP(/)

#undef MAKE_OP

inline __device__ float3 operator*(const float3& v, float a) {
    float3 res;
    res.x = v.x * a;
    res.y = v.y * a;
    res.z = v.z * a;
    return res;
}

inline __device__ float3 operator*(float a, const float3& v) {
    return v * a;
}

inline __device__ float dot_product(const float3& v1, const float3& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline __device__ float3 normalize(const float3& v) {
    const float norm = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    float3 res = v;
    res.x /= norm;
    res.y /= norm;
    res.z /= norm;

    return res;
}

inline __device__ float3 clip(const float3& v, float lo, float hi) {
    float3 res = v;
    if (res.x < lo) res.x = lo;
    if (res.x > hi) res.x = hi;

    if (res.y < lo) res.y = lo;
    if (res.y > hi) res.x = hi;

    if (res.z < lo) res.z = lo;
    if (res.z > hi) res.z = hi;
    return res;
}

#endif
