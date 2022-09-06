#include "raytracing.h"
#include "geometry.h"
#include "utils.h"


__constant__ float d_screen_edges[4];

__device__ float3 trace_ray(const SceneInfo& scene, const Ray& ray,
                            int n_objects, SceneObject** objects,
                            Ray& new_ray,
                            int& closest_idx) {
    int closest_obj_idx = -1;
    float closest_dist = 1000000.0f;
    for (int i = 0; i < n_objects; ++i) {
        float dist = objects[i]->ray_intersection(ray);
        if (dist > 0.0f && dist < closest_dist) {
            closest_dist = dist;
            closest_obj_idx = i;
        }
    }

    closest_idx = closest_obj_idx;
    
    //no intersection found
    if (closest_obj_idx == -1) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }


    const SceneObject* obj = objects[closest_obj_idx];
    const float3 intersect_point = ray.origin + ray.direction * closest_dist;
    const float3 normal_vec = obj->normal(intersect_point);

    const float3 to_light = normalize(scene.light_pos - intersect_point);
    const float3 to_camera = normalize(scene.camera_pos - intersect_point);

    float3 col = make_float3(ambient_col, ambient_col, ambient_col);
    col = col + obj->get_diffuse_coeff() *
                max(dot_product(normal_vec, to_light), 0.0f) *
                obj->get_color(intersect_point);
    float specular_factor = max(dot_product(normal_vec, normalize(to_light + to_camera)), 0.0f);
    specular_factor = obj->get_specular_coeff() * pow(specular_factor, default_exponent);
    col = col + specular_factor * make_float3(1.0f, 1.0f, 1.0f);

    //Compute the reflecting ray:
    const float3 new_ray_orig = intersect_point + 0.0001f * normal_vec;
    const float3 new_ray_dir = normalize(ray.direction - 2.0f *
                                         dot_product(ray.direction, normal_vec) *
                                         normal_vec);

    new_ray.origin = new_ray_orig;
    new_ray.direction = new_ray_dir;

    return col;
}

__global__ void trace_rays(SceneInfo scene,
                           int n_objects, SceneObject** objects,
                           DeviceImage img) {

    constexpr int max_depth = 5;
    const int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    const float pixel_sz_x = static_cast<float>(d_screen_edges[2] - d_screen_edges[0])
                             / static_cast<float>(img.width);
    const float pixel_sz_y = static_cast<float>(d_screen_edges[3] - d_screen_edges[1])
                             / static_cast<float>(img.height);

    const float x = d_screen_edges[0] + (static_cast<float>(x_idx) + 0.5f) * pixel_sz_x;
    const float y = d_screen_edges[1] + (static_cast<float>(y_idx) + 0.5f) * pixel_sz_y;

    const float3 ray_dir = normalize(make_float3(x - scene.camera_pos.x,
                                                 y - scene.camera_pos.y,
                                                 0.0f - scene.camera_pos.z));

    Ray ray{ray_dir, scene.camera_pos};

    int depth = 0;
    float3 col = make_float3(0.0f, 0.0f, 0.0f);
    int closest_idx = -1;
    float reflection = 1.0f;
    Ray refl_ray{make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f)};
    while (depth < max_depth) {
        const float3 col_ray = trace_ray(scene, ray, n_objects, objects, refl_ray, closest_idx);
        if (closest_idx < 0) break;
        col = col + reflection * col_ray;
        reflection *= objects[closest_idx]->get_reflection_coeff();
        ray = refl_ray;
        ++depth;
    }
    const float3 clipped_col = clip(col, 0.0f, 1.0f);
    img.set_color(x_idx, y_idx, clipped_col);
}


