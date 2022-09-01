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

    scene.screen_edges[0]= -1.0f;
    scene.screen_edges[1]= -1.0f / aspect_ratio + 0.25f;
    scene.screen_edges[2]= 1.0f;
    scene.screen_edges[3]= 1.0f / aspect_ratio + 0.25f;

    scene.camera_pos = make_float3(0.0f, 0.35f, -1.0f);
    scene.light_pos = make_float3(5.0f, 5.0f, -10.0f);
}

__global__ void alloc_objects(SceneObject** objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        objects[0] = new Sphere(make_float3(0.75f, 0.1f, 1.0f), 0.6, make_float3(0.0f, 0.0f, 1.0f));
        objects[1] = new Sphere(make_float3(-0.75f, 0.1f, 2.25f), 0.6, make_float3(0.5f, 0.223f, 0.5f));
        objects[2] = new Sphere(make_float3(-2.75f, 0.1f, 3.5f), 0.6, make_float3(0.0f, 1.0f, 0.0f));
    }
}

int main(int argc, char* argv[])
{
    int img_width = 400;
    int img_height = 400;

    if (argc == 3) {
        img_width = std::atoi(argv[1]);
        img_height = std::atoi(argv[1]);
    }

    SceneObject** objects;

    cuda_err(cudaMalloc((void**) &objects, 3 * sizeof(void*)));
    alloc_objects<<<1,1>>>(objects);

    DeviceImage img;
    SceneInfo scene;

    init_scene(img_height, img_width, img, scene);
    constexpr int tx = 8;
    constexpr int ty = 8;

    dim3 blocks(img_width / tx, img_height / ty);
    dim3 threads(tx, ty);

    trace_rays<<<blocks, threads>>>(scene, 3, objects, img);
    std::vector<float3> pixels(img_width * img_height);

    cuda_err(cudaMemcpy(pixels.data(), img.pixels, img_width * img_height * sizeof(float3), cudaMemcpyDeviceToHost));
    std::vector<float> flattened_pixels(img_width * img_height * 3);
    int idx = 0;
    for (float3 vec : pixels) {
        if (idx % 3 == 0)
            flattened_pixels[idx] = vec.x;
        else if (idx % 3 == 1)
            flattened_pixels[idx] = vec.y;
        else if (idx % 3 == 2)
            flattened_pixels[idx] = vec.z;

        ++idx;
    }

    export_image(img_height, img_width, flattened_pixels.data());


    return 0;
}
