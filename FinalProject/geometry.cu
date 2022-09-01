#include "utils.h"
#include "geometry.h"
#include "raytracing.h"

__device__ float Sphere::ray_intersection(const Ray& ray) const {
    const float3 v = ray.origin - this->origin;
    const float a = dot_product(ray.direction, ray.direction);
    const float b = 2.0f * dot_product(ray.direction, v);
    const float c = dot_product(v, v) - this->radius * this->radius;
    const float discriminant = b * b - 4 * a * c;

    //Is there an intersection in the direction of the ray?
    if (discriminant > 0) {
        //Solve for the intersecting points
        const float root = (b < 0) ?
                           (-b - sqrt(discriminant)) / 2.0f :
                           (-b + sqrt(discriminant)) / 2.0f;
        float t0 = root / a;
        float t1 = c / root;
        if (t1 > t0) {
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

__device__ float3 Sphere::get_color() const {
    return this->col;
}


