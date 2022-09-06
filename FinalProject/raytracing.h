#ifndef RAYTRACING_H
#define RAYTRACING_H

#include <string>

struct SceneObject;

extern __constant__ float d_screen_edges[];

struct SceneInfo {
    //Location of camera in the scene
    float3 camera_pos;
    float3 light_pos;

    //Image height and width in pixels
    int height, width;
};

struct Ray {
    __device__ Ray(const float3& dir, const float3& origin) {
        this->direction.x = dir.x;
        this->direction.y = dir.y;
        this->direction.z = dir.z;

        this->origin.x = origin.x;
        this->origin.y = origin.y;
        this->origin.z = origin.z;
    }
    float3 direction;
    float3 origin;
};

struct DeviceImage {
    float3* pixels;
    int width, height;

    __device__ void set_color(int x, int y, float3 col) {
        pixels[(height - y - 1) * width + x].x = col.x;
        pixels[(height - y - 1) * width + x].y = col.y;
        pixels[(height - y - 1) * width + x].z = col.z;
    }
};


__global__ void trace_rays(SceneInfo scene,
                           int n_objects, SceneObject** objects,
                           DeviceImage img);
#endif
