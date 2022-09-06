#ifndef GEOMETRY_H
#define GEOMETRY_H

struct Ray;

//Some constants in terms of color and lighting in the scene;
constexpr float ambient_col = 0.05f;
constexpr float default_diffusion = 1.0f;
constexpr float default_specular = 1.0f;
constexpr float default_exponent = 50.0f;
constexpr float default_reflection = 1.0f;

struct SceneObject {
    __device__ virtual float ray_intersection(const Ray& r) const = 0;
    __device__ virtual float3 get_color(const float3& point) const = 0;
    __device__ virtual float3 normal(const float3& v) const = 0;

    __device__ virtual float get_diffuse_coeff() const = 0;
    __device__ virtual float get_specular_coeff() const = 0;
    __device__ virtual float get_reflection_coeff() const = 0;

    virtual ~SceneObject() = default;
};

struct Sphere : SceneObject {
    __device__ Sphere(const float3& pos, float r, const float3& color) :
        origin(pos), radius(r), col(color) {}
    float3 origin;
    float radius;

    float3 col;

    __device__ float ray_intersection(const Ray& r) const override;
    __device__ float3 normal(const float3& v) const override;
    __device__ float3 get_color(const float3& point) const override;
    __device__ float get_diffuse_coeff() const override { return default_diffusion; }
    __device__ virtual float get_specular_coeff() const override { return default_specular; }
    __device__ virtual float get_reflection_coeff() const override { return default_reflection; }
};

struct Plane : SceneObject {
    __device__ Plane(const float3& point, const float3& normal) :
        point(point), normal_vec(normal) {}

    float3 point;
    float3 normal_vec;

    __device__ float3 get_color(const float3& point) const override;
    __device__ float ray_intersection(const Ray& r) const override;
    __device__ float3 normal(const float3& v) const override { return this->normal_vec; }

    __device__ float get_diffuse_coeff() const override { return 0.75f; }
    __device__ float get_specular_coeff() const override { return 0.5f; }
    __device__ float get_reflection_coeff() const override { return 0.25f; }
};

#endif
