#include <cstdlib>
#include <vector>
#include <cuda.h>

#include "geometry.h"
#include "raytracing.h"
#include "utils.h"
#include "export.h"

void init_scene(int height, int width, DeviceImage& image, SceneInfo& scene) {
    const float aspect_ratio = width / static_cast<float>(height);
    image.height = height;
    image.width  = width;

    cuda_err(cudaMalloc(&image.pixels, sizeof(float3) * width * height));

    scene.height = height;
    scene.width = width;

    float host_screen_edges[4];

    host_screen_edges[0]= -1.0f;
    host_screen_edges[1]= -1.0f / aspect_ratio + 0.25f;
    host_screen_edges[2]= 1.0f;
    host_screen_edges[3]= 1.0f / aspect_ratio + 0.25f;

    cuda_err(cudaMemcpyToSymbol(d_screen_edges, host_screen_edges, sizeof(host_screen_edges)));

    scene.camera_pos = make_float3(0.0f, 0.35f, -1.0f);
    scene.light_pos = make_float3(5.0f, 5.0f, -10.0f);
}

__global__ void alloc_objects(SceneObject** objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        objects[0] = new Sphere(make_float3(0.75f, 0.1f, 1.0f), 0.6f, make_float3(0.0f, 0.0f, 1.0f));
        objects[1] = new Sphere(make_float3(-0.75f, 0.1f, 2.25f), 0.6f, make_float3(0.5f, 0.223f, 0.5f));
        objects[2] = new Sphere(make_float3(-2.75f, 0.1f, 3.5f), 0.6f, make_float3(1.0f, 0.572f, 0.184f));
        objects[3] = new Plane(make_float3(0.0f, -0.5f, 0.0f), make_float3(0.0f, 1.0f, 0.0f));
    }
}

__global__ void free_objects(SceneObject** objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete objects[0];
        delete objects[1];
        delete objects[2];
        delete objects[3];
    }
}

int main(int argc, char* argv[])
{
    constexpr int n_objects = 4;
    int img_width = 1200;
    int img_height = 800;

    if (argc == 3) {
        img_width = std::atoi(argv[1]);
        img_height = std::atoi(argv[1]);
    }

    SceneObject** objects;

    cuda_err(cudaMalloc((void**) &objects, n_objects * sizeof(void*)));
    alloc_objects<<<1,1>>>(objects);
    cuda_err(cudaGetLastError());
    cuda_err(cudaDeviceSynchronize());


    DeviceImage img;
    SceneInfo scene;

    init_scene(img_height, img_width, img, scene);
    constexpr int tx = 8;
    constexpr int ty = 8;

    dim3 blocks(img_width / tx, img_height / ty);
    dim3 threads(tx, ty);

    trace_rays<<<blocks, threads>>>(scene, n_objects, objects, img);
    std::vector<float3> pixels(img_width * img_height);

    cuda_err(cudaMemcpy(pixels.data(), img.pixels, img_width * img_height * sizeof(float3), cudaMemcpyDeviceToHost));
    std::vector<float> flattened_pixels(img_width * img_height * 3);
    int idx = 0;
    for (float3 vec : pixels) {
        //A bit of a hack: OpenCV works by default in BGR, so swap x and z elements here
        flattened_pixels[idx] = vec.z;
        flattened_pixels[idx + 1] = vec.y;
        flattened_pixels[idx + 2] = vec.x;

        idx += 3;
    }

    export_image(img_height, img_width, flattened_pixels.data());

    free_objects<<<1,1>>>(objects);
    cudaFree((void*) objects);

    return 0;
}
