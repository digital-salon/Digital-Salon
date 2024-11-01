#pragma once

#include <math.h>
#include <limits>
#include <stdio.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>

const float math_pi = 3.1415926536f;
const float math_2pi = 2 * math_pi;
const float math_radians = math_pi / 180.0f;
const float math_degrees = 180.0f / math_pi;
const float math_epsilon = 1e-5f;
const float math_maxfloat = (std::numeric_limits<float>::max)();
const float math_minfloat = -(std::numeric_limits<float>::max)();

class Matrix4x4;
class Matrix3x3;
class Vector4;
class Vector3;
class Vector2;

typedef unsigned int uint;
typedef Matrix4x4 mat4;
typedef Matrix3x3 mat3;
typedef Vector4 vec4;
typedef Vector3 vec3;
typedef Vector2 vec2;

template <class T> T fract(const T& x) { return x - floor(x); }
template <class T> T round(const T& x) { return floor(x + T(0.5)); }
template <class TA, class TB, class TC> TA clamp(const TA& x, const TB& minv, const TC& maxv) { if (x < (TA)minv) return (TA)minv; else if (x > (TA)maxv) return (TA)maxv; return x; }
template <class T> T sqr(const T& x) { return x * x; }
template <class T> T cube(const T& x) { return x * x * x; }
template <class T> T sign(const T& x) { if (x > 0) return T(1); else if (x < 0) return T(-1); return T(0); }
template <class T> T deg2rad(const T& x) { return x * T(math_pi / 180.0); }
template <class T> T rad2deg(const T& x) { return x * T(180.0 / math_pi); }
template <class T> T g_min(const T& a, const T& b) { return (a < b ? a : b); }
template <class T> T g_min(const T& a, const T& b, const T& c) { return min(min(a, b), c); };
template <class T> T g_min(const T& a, const T& b, const T& c, const T& d) { return min(min(a, b), min(c, d)); };
template <class T> T g_max(const T& a, const T& b) { return (a > b ? a : b); }
template <class T> T g_max(const T& a, const T& b, const T& c) { return max(max(a, b), c); }
template <class T> T g_max(const T& a, const T& b, const T& c, const T& d) { return max(max(a, b), max(c, d)); }
template <class T> T smoothStep(const T& l, const T& u, const T& x) { T t = clamp((x - l) / (u - l), (T)0, (T)1); return t * t * (3 - 2 * t); }
template <class T> T lerp(const T& a, const T& b, const float& s) { T t = b * s + (1 - s) * a; return t; }
template <class T> void loop(T& a, const T& low, const T& high, const T& inc = 1.0) { if (a >= high) a = low; else a += inc; }
template <class T> T rand(const T& low, const T& high) { return rand() / (static_cast<float>(RAND_MAX) + 1.0) * (high - low) + low; }


__host__ __device__  float cross(const Vector2& a, const Vector2& b);
__host__ __device__  Vector2 operator *(float s, const Vector2& v);
__host__ __device__ Vector2 normalize(Vector2& v);
__host__ __device__ float   dot(const Vector2& a, const Vector2& b);
__host__ __device__ float   length(const Vector2& v);
__host__ __device__ Vector2 vec2_rnd(float low, float high);

class Vector2
{

public:
    Vector2();
    Vector2(float x, float y);
    Vector2(const Vector2& v);
    Vector2(const float* v);
    ~Vector2();

    __host__ __device__ float length() const;
    float dot(const Vector2& vec);
    float angle(Vector2& vec);
    float cross(const Vector2& vec);
    Vector2 normalized();
    void normalize();
    void set(float x, float y);
    void set(float* vec);

    Vector2 mulMatrix(float* matrix);
    Vector2& get();

    void print();

    //Assignment
    Vector2& operator =  (const Vector2& a);
    Vector2& operator += (const Vector2& a);
    Vector2& operator += (float s);
    Vector2& operator -= (const Vector2& a);
    Vector2& operator -= (float s);
    Vector2& operator *= (const Vector2& a);
    Vector2& operator *= (float s);
    Vector2& operator /= (const Vector2& a);
    Vector2& operator /= (float s);

    //Arithmetic
    friend Vector2 operator + (const Vector2& a, const Vector2& b);
    friend Vector2 operator + (const Vector2& a, float s);
    friend Vector2 operator + (float s, const Vector2& a);
    friend Vector2 operator - (const Vector2& a, const Vector2& b);
    friend Vector2 operator - (const Vector2& a, float s);
    friend Vector2 operator - (const Vector2& a);
    friend Vector2 operator * (const Vector2& a, const Vector2& b);
    friend Vector2 operator * (const Vector2& a, float s);
    friend Vector2 operator * (float s, const Vector2& a);
    friend Vector2 operator / (const Vector2& a, const Vector2& b);
    friend Vector2 operator / (const Vector2& a, float s);

    //Comparison
    friend bool operator == (const Vector2& a, const Vector2& b);
    friend bool operator != (const Vector2& a, const Vector2& b);
    friend bool operator <= (const Vector2& a, const Vector2& b);
    friend bool operator <  (const Vector2& a, const Vector2& b);
    friend bool operator >= (const Vector2& a, const Vector2& b);
    friend bool operator >  (const Vector2& a, const Vector2& b);

    float x;
    float y;
};

Vector3 cross(const Vector3& a, const Vector3& b);
Vector3 operator *(float s, const Vector3& v);
Vector3 normalize(const Vector3& v);
float   dot(const Vector3& a, const Vector3& b);
float   length(const Vector3& v);
float   lengthSq(const Vector3& v);
Vector3 vec3_rnd(float low, float high);
Vector3 vec3_rnd_pos_y(float low, float high);
vec3 abs(const Vector3& v);
vec3 step(const vec3& edge, const vec3& x);
vec3 reflect(const vec3& I, const vec3& N);

class Vector3
{

public:
    Vector3();
    Vector3(float x, float y, float z);
    Vector3(const Vector3& v);
    Vector3(const Vector4& v);
    Vector3(const float* v);
    explicit Vector3(float x);
    ~Vector3();

    float length() const;
    float dot(const Vector3& vec);
    float angle(Vector3& vec);
    Vector3 cross(const Vector3& vec);
    Vector3 normalized();
    void normalize();
    void set(float x, float y, float z);
    void set(float* vec);

    Vector3 mulMatrix(float* matrix);

    Vector3& get();

    void print();

    //Assignment
    Vector3& operator =  (const Vector3& a);
    Vector3& operator += (const Vector3& a);
    Vector3& operator += (float s);
    Vector3& operator -= (const Vector3& a);
    Vector3& operator -= (float s);
    Vector3& operator *= (const Vector3& a);
    Vector3& operator *= (float s);
    Vector3& operator /= (const Vector3& a);
    Vector3& operator /= (float s);

    //Arithmetic
    friend Vector3 operator + (const Vector3& a, const Vector3& b);
    friend Vector3 operator + (const Vector3& a, float s);
    friend Vector3 operator + (float s, const Vector3& a);
    friend Vector3 operator - (const Vector3& a, const Vector3& b);
    friend Vector3 operator - (const Vector3& a, float s);
    friend Vector3 operator - (const Vector3& a);
    friend Vector3 operator * (const Vector3& a, const Vector3& b);
    friend Vector3 operator * (const Vector3& a, float s);
    friend Vector3 operator * (float s, const Vector3& a);
    friend Vector3 operator / (const Vector3& a, const Vector3& b);
    friend Vector3 operator / (const Vector3& a, float s);

    //Comparison
    friend bool operator == (const Vector3& a, const Vector3& b);
    friend bool operator != (const Vector3& a, const Vector3& b);
    friend bool operator <= (const Vector3& a, const Vector3& b);
    friend bool operator <  (const Vector3& a, const Vector3& b);
    friend bool operator >= (const Vector3& a, const Vector3& b);
    friend bool operator >  (const Vector3& a, const Vector3& b);

    float x;
    float y;
    float z;
};


Vector4 cross(const Vector4& a, const Vector4& b);
Vector4 operator *(float s, const Vector4& v);
Vector4 normalize(Vector4& v);
float   dot(const Vector4& a, const Vector4& b);
float   length(const Vector4& v);
vec4 abs(const Vector4& v);
vec4 lessThan(const Vector4& v, const Vector4& w);
vec4 floor(const Vector4& v);
Vector4 vec4_rnd(float low, float high);

class Vector4
{

public:
    Vector4();
    Vector4(float x, float y, float z, float w);
    Vector4(const Vector4& vec);
    Vector4(const Vector3& vec);
    Vector4(const Vector3& vec, float v);
    Vector4(const float3& vec, float v);
    Vector4(const float* vec);
    explicit Vector4(float x);
    ~Vector4();

    float length() const;
    float dot(const Vector4& vec);
    float angle(Vector4& vec);
    Vector4 cross(const Vector4& vec);
    Vector4 normalized();
    void normalize();
    void set(float x, float y, float z, float w);
    void set(float* vec);

    Vector4& get();

    void print();

    //Assignment
    Vector4& operator =  (const Vector4& a);
    Vector4& operator += (const Vector4& a);
    Vector4& operator += (float s);
    Vector4& operator -= (const Vector4& a);
    Vector4& operator -= (float s);
    Vector4& operator *= (const Vector4& a);
    Vector4& operator *= (float s);
    Vector4& operator /= (const Vector4& a);
    Vector4& operator /= (float s);

    //Arithmetic
    friend Vector4 operator + (const Vector4& a, const Vector4& b);
    friend Vector4 operator + (const Vector4& a, float s);
    friend Vector4 operator + (float s, const Vector4& a);
    friend Vector4 operator - (const Vector4& a, const Vector4& b);
    friend Vector4 operator - (const Vector4& a, float s);
    friend Vector4 operator - (const Vector4& a);
    friend Vector4 operator * (const Vector4& a, const Vector4& b);
    friend Vector4 operator * (const Vector4& a, float s);
    friend Vector4 operator * (float s, const Vector4& a);
    friend Vector4 operator / (const Vector4& a, const Vector4& b);
    friend Vector4 operator / (const Vector4& a, float s);

    //Comparison
    friend bool operator == (const Vector4& a, const Vector4& b);
    friend bool operator != (const Vector4& a, const Vector4& b);
    friend bool operator <= (const Vector4& a, const Vector4& b);
    friend bool operator <  (const Vector4& a, const Vector4& b);
    friend bool operator >= (const Vector4& a, const Vector4& b);
    friend bool operator >  (const Vector4& a, const Vector4& b);

    float x;
    float y;
    float z;
    float w;
};


//Matrix4x4 operator *(float s, const Matrix4x4& m);
Vector4 operator *(const Vector4& v, const Matrix4x4& m);
float4 MatVecHost(const Matrix4x4& m, const float4& v);
float3 MatVecHost(const Matrix4x4& m, const float3& v);
Vector3 operator *(const Matrix4x4& m, const Vector3& v);
Matrix4x4 transpose(const Matrix4x4& m);
Matrix4x4 inverse(const Matrix4x4& m);
inline vec3 float2vec(const float3& u) { return vec3(u.x, u.y, u.z); }
inline float3 vec2float(const vec3& u) { return make_float3(u.x, u.y, u.z); }
float anglefloat3(const float3& u, const float3& v);

class Matrix4x4
{

public:
    Matrix4x4();
    Matrix4x4(float a11, float a12, float a13, float a14, float a21, float a22, float a23, float a24,
        float a31, float a32, float a33, float a34, float a41, float a42, float a43, float a44);
    Matrix4x4(const Matrix4x4& m);
    Matrix4x4(const Matrix3x3& m);
    Matrix4x4(float d);

    Matrix4x4(const float* vec);
    ~Matrix4x4();

    void set(float a11, float a12, float a13, float a14, float a21, float a22, float a23, float a24,
        float a31, float a32, float a33, float a34, float a41, float a42, float a43, float a44);
    void set(float* mat);

    Matrix4x4& get();
    void data(float* vec) const;

    Matrix4x4 inverse();
    Matrix4x4 transpose();

    void setToIdentity();
    void setToZero();

    float determinant();

    void print();

    //Assignment
    Matrix4x4& operator =  (const Matrix4x4& a);
    Matrix4x4& operator += (const Matrix4x4& a);
    Matrix4x4& operator += (float s);
    Matrix4x4& operator -= (const Matrix4x4& a);
    Matrix4x4& operator -= (float s);
    Matrix4x4& operator *= (const Matrix4x4& m);
    Matrix4x4& operator *= (float s);
    Matrix4x4& operator /= (float s);

    //Arithmetic
    friend Matrix4x4 operator + (const Matrix4x4& a, const Matrix4x4& b);
    friend Matrix4x4 operator + (const Matrix4x4& a, float s);
    friend Matrix4x4 operator + (float s, const Matrix4x4& a);
    friend Matrix4x4 operator - (const Matrix4x4& a, const Matrix4x4& b);
    friend Matrix4x4 operator - (const Matrix4x4& a, float s);
    friend Matrix4x4 operator - (const Matrix4x4& a);
    friend Matrix4x4 operator * (const Matrix4x4& a, const Matrix4x4& b);
    friend Matrix4x4 operator * (const Matrix4x4& a, float s);
    friend Matrix4x4 operator * (float s, const Matrix4x4& a);
    friend Matrix4x4 operator / (const Matrix4x4& a, float s);

    //Comparison
    friend bool operator == (const Matrix4x4& a, const Matrix4x4& b);
    friend bool operator != (const Matrix4x4& a, const Matrix4x4& b);


    //OpenGL
    static const Matrix4x4 zero();
    static const Matrix4x4 identitiy();
    static const Matrix4x4 rotateX(float angle);
    static const Matrix4x4 rotateY(float angle);
    static const Matrix4x4 rotateZ(float angle);
    static const Matrix4x4 rotate(float angle, float x, float y, float z);
    static const Matrix4x4 rotate(float angle, const Vector3& n);
    static const Matrix4x4 scale(const Vector3& s);
    static const Matrix4x4 scale(float x, float y, float z);
    static const Matrix4x4 translate(const Vector3& t);
    static const Matrix4x4 translate(float x, float y, float z);
    static const Matrix4x4 orthographic(float left, float right, float bottom, float top, float zNear, float zFar);
    static const Matrix4x4 perspective(float fov, float ratio, float zNear, float zFar);
    static const Matrix4x4 lookAt(const Vector3& position, const Vector3& center, const Vector3& up);
    static const Matrix4x4 lookAt(float px, float py, float pz, float cx, float cy, float cz, float ux, float uy, float uz);
    static const Matrix3x3 normalMatrix(const Matrix4x4& model);

    float a11;
    float a21;
    float a31;
    float a41;

    float a12;
    float a22;
    float a32;
    float a42;

    float a13;
    float a23;
    float a33;
    float a43;

    float a14;
    float a24;
    float a34;
    float a44;

private:

};


Matrix3x3 operator *(float s, const Matrix3x3& m);
Vector3 operator *(const Vector3& v, const Matrix3x3& m);
Vector3 operator *(const Matrix3x3& m, const Vector3& v);
Matrix3x3 transpose(const Matrix3x3& m);
Matrix3x3 inverse(const Matrix3x3& m);

class Matrix3x3
{

public:
    Matrix3x3();
    Matrix3x3(float u);
    Matrix3x3(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33);
    Matrix3x3(const Matrix3x3& m);
    Matrix3x3(const Matrix4x4& m);
    Matrix3x3(const float* vec);
    ~Matrix3x3();

    void set(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33);
    void set(float* mat);

    Matrix3x3& get();
    void data(float* vec) const;

    Matrix3x3 inverse();
    Matrix3x3 transpose();

    void setToIdentity();
    void SetRow(const int& row, const float3& u);
    void setToZero();

    float determinant();

    void print();

    //Assignment
    Matrix3x3& operator =  (const Matrix3x3& a);
    Matrix3x3& operator += (const Matrix3x3& a);
    Matrix3x3& operator += (float s);
    Matrix3x3& operator -= (const Matrix3x3& a);
    Matrix3x3& operator -= (float s);
    Matrix3x3& operator *= (const Matrix3x3& m);
    Matrix3x3& operator *= (float s);
    Matrix3x3& operator /= (float s);

    //Arithmetic
    friend Matrix3x3 operator + (const Matrix3x3& a, const Matrix3x3& b);
    friend Matrix3x3 operator + (const Matrix3x3& a, float s);
    friend Matrix3x3 operator + (float s, const Matrix3x3& a);
    friend Matrix3x3 operator - (const Matrix3x3& a, const Matrix3x3& b);
    friend Matrix3x3 operator - (const Matrix3x3& a, float s);
    friend Matrix3x3 operator - (const Matrix3x3& a);
    friend Matrix3x3 operator * (const Matrix3x3& a, const Matrix3x3& b);
    friend Matrix3x3 operator * (const Matrix3x3& a, float s);
    friend Matrix3x3 operator * (float s, const Matrix3x3& a);
    friend Matrix3x3 operator / (const Matrix3x3& a, float s);

    //Comparison
    friend bool operator == (const Matrix3x3& a, const Matrix3x3& b);
    friend bool operator != (const Matrix3x3& a, const Matrix3x3& b);

    float a11;
    float a21;
    float a31;

    float a12;
    float a22;
    float a32;

    float a13;
    float a23;
    float a33;

private:

};


#define QUATERNION_PI 3.141592654
#define QUATERNION_ERROR 1e-6

class Quaternion
{

public:
    Quaternion();
    Quaternion(const float x, const float y, const float z, const float w);
    Quaternion(const Vector3& v, const float s);
    Quaternion(const float* f);
    Quaternion(const Quaternion& q);
    ~Quaternion();

    float length();
    float length2();
    void normalize();
    Quaternion normalized();

    void set(const float x, const float y, const float z, const float w);
    void set(const Vector3& v, const float s);
    void set(const float* f);
    void set(const Quaternion& q);

    Quaternion& setEuler(const float Yaw, const float Pitch, const float Roll);
    Quaternion slerp(const Quaternion& from, const Quaternion& to, const float t);

    void toMatrix(float* m);
    void fromAxisAngle(float x, float y, float z, float degrees);

    Quaternion& get();

    void print();

    //Assignment
    Quaternion& operator =  (const Quaternion& q);
    Quaternion& operator += (const Quaternion& q);
    Quaternion& operator -= (const Quaternion& q);
    Quaternion& operator *= (const Quaternion& q);
    Quaternion& operator *= (float s);
    Quaternion& operator /= (float s);

    //Arithmetic
    Quaternion operator *(const Quaternion& q);

    friend Quaternion operator - (const Quaternion& a);
    friend Quaternion operator + (const Quaternion& a, const Quaternion& b);
    friend Quaternion operator - (const Quaternion& a, const Quaternion& b);
    friend Quaternion operator * (const Quaternion& a, const float f);
    friend Quaternion operator * (const float f, const Quaternion& a);
    friend Quaternion operator * (const Quaternion& a, const Quaternion& b);
    friend Quaternion operator / (const Quaternion& a, const float f);

    //Comparison
    friend bool operator == (const Quaternion& a, const Quaternion& b);
    friend bool operator != (const Quaternion& a, const Quaternion& b);

    float x;
    float y;
    float z;
    float w;

private:
};