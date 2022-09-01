#include "raytracing.h"
#include "geometry.h"
#include "utils.h"

//Some constants in terms of color and lighting in the scene;
constexpr float ambient_col = 0.05f;
constexpr float default_diffusion = 1.0f;
constexpr float default_specular = 1.0f;
constexpr float default_exponent = 50.0f;

__device__ float3 trace_ray(const SceneInfo& scene, const Ray& ray, int n_objects, const SceneObject** objects) {
    int closest_obj_idx = -1;
    float closest_dist = 1000000.0f;
    for (int i = 0; i < n_objects; ++i) {
        float dist = objects[i]->ray_intersection(ray);
        if (dist > 0.0f && dist < closest_dist) {
            closest_dist = dist;
            closest_obj_idx = i;
        }
    }
    
    //ray_intersection returns a negative value if no intersection is found
    if (closest_obj_idx == -1) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    const SceneObject* obj = objects[closest_obj_idx];
    const float3 intersect_point = ray.origin + ray.direction * closest_dist;
    const float3 normal_vec = obj->normal(intersect_point);

    const float3 to_light = normalize(scene.light_pos - intersect_point);
    const float3 to_camera = normalize(scene.camera_pos - intersect_point);

    float3 col = make_float3(ambient_col, ambient_col, ambient_col);
    col = col + default_diffusion * max(dot_product(normal_vec, to_light), 0.0f) * obj->get_color();
    float specular_factor = default_specular *
                            max(dot_product(normal_vec, normalize(to_light + to_camera)), 0.0f);
    specular_factor = pow(specular_factor, default_exponent);
    col = col + specular_factor * make_float3(1.0f, 1.0f, 1.0f);

    return col;
}

__global__ void trace_rays(const SceneInfo& scene,
                           int n_objects, SceneObject** objects,
                           DeviceImage img) {
    const int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    const float pixel_sz_x = scene.screen_edges[2] - scene.screen_edges[0];
    const float pixel_sz_y = scene.screen_edges[1] - scene.screen_edges[3];

    const float x = scene.screen_edges[0] + x_idx * pixel_sz_x;
    const float y = scene.screen_edges[0] + y_idx * pixel_sz_y;

    const float3 ray_dir = normalize(make_float3(x - scene.camera_pos.x,
                                                 y - scene.camera_pos.y,
                                                 0.0f - scene.camera_pos.z));

    Ray ray{ray_dir, scene.camera_pos};

    float3 col = trace_ray(scene, ray, n_objects, objects);
    img.set_color(x_idx, y_idx, col);
}


