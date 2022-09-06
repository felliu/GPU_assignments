#include "utils.h"
#include "geometry.h"
#include "raytracing.h"

__device__ float Sphere::ray_intersection(const Ray& ray) const {
    const float3 v = ray.origin - this->origin;
    const float a = dot_product(ray.direction, ray.direction);
    const float b = 2.0f * dot_product(ray.direction, v);
    const float c = dot_product(v, v) - this->radius * this->radius;
    const float discriminant = b * b - 4 * a * c;
    //printf("Discriminant: %f\n", discriminant);

    //Is there an intersection in the direction of the ray?
    if (discriminant > 0) {
        //Solve for the intersecting points
        const float root = (b < 0) ?
                           (-b - sqrt(discriminant)) / 2.0f :
                           (-b + sqrt(discriminant)) / 2.0f;
        float t0 = root / a;
        float t1 = c / root;
        if (t0 > t1) {
            const float tmp = t1;
            t1 = t0;
            t0 = tmp;
        }

        if (t1 >= 0)
            return (t0 < 0) ? t1 : t0;
    }

    return -1.0f;
}

__device__ float3 Sphere::normal(const float3& v) const {
    return normalize(v - this->origin);
}

//The position argument doesn't matter for spheres in our scene
__device__ float3 Sphere::get_color(const float3& _) const {
    return this->col;
}

__device__ float Plane::ray_intersection(const Ray& ray) const {
    const float denominator = dot_product(ray.direction, this->normal_vec);
    if (fabs(denominator) < 1e-6) {
        return -1.0f;
    }

    const float d = dot_product(this->point - ray.origin, this->normal_vec) / denominator;
    if (d < 0.0f) {
        return -1.0f;
    }

    return d;
}

__device__ float3 Plane::get_color(const float3& point) const {
    const float3 color0 = make_float3(0.0f, 0.0f, 0.0f);
    const float3 color1 = make_float3(1.0f, 1.0f, 1.0f);
    //Way to do "true mod" as is implemented in Python
    const int x_mod = ((static_cast<int>(point.x * 2.0f) % 2) + 2) % 2;
    const int z_mod = ((static_cast<int>(point.z * 2.0f) % 2) + 2) % 2;

    if (x_mod == z_mod) {
        return color0;
    }
    return color1;
}

