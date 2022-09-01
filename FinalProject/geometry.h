#ifndef GEOMETRY_H
#define GEOMETRY_H

struct Ray;

struct SceneObject {
    __device__ virtual float ray_intersection(const Ray& r) const = 0;
    __device__ virtual float3 get_color() const = 0;
    __device__ virtual float3 normal(const float3& v) const = 0;
};

struct Sphere : SceneObject {
    __device__ Sphere(float3 pos, float r, float3 color) :
        origin(pos), radius(r), col(color) {}
    float3 origin;
    float radius;

    float3 col;

    __device__ float ray_intersection(const Ray& r) const override;
    __device__ float3 normal(const float3& v) const override;
    __device__ float3 get_color() const override;
};

struct Plane : SceneObject {
    float3 point;
    float3 normal_vec;

    __device__ float ray_intersection(const Ray& r) const override;
    __device__ float3 normal(const float3& v) const override;
};

#endif
