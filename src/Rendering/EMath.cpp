#include "EMath.h"

Vector2::Vector2()
    : x(0.0),
    y(0.0)
{
}

Vector2::Vector2(float x, float y)
{
    this->x = x;
    this->y = y;
}

Vector2::Vector2(const Vector2& v)
{
    this->x = v.x;
    this->y = v.y;
}

Vector2::Vector2(const float* v)
{
    this->x = v[0];
    this->y = v[1];
}

Vector2::~Vector2()
{
}

__host__ __device__ float Vector2::length() const
{
    float length = 0.0f;

    length = sqrtf((this->x * this->x) + (this->y * this->y));

    return length;
}

float Vector2::dot(const Vector2& vec)
{
    float dot = 0.0f;

    dot = (this->x * vec.x) + (this->y * vec.y);

    return dot;
}

float Vector2::angle(Vector2& vec)
{
    float d = this->dot(vec);

    float a = this->length();
    float b = vec.length();

    float s = d / (a * b);
    float angle = (float)acos((double)s);

    return angle;
}

float Vector2::cross(const Vector2& vec)
{
    return (this->x * vec.y - this->y * vec.x);
}

void Vector2::normalize()
{
    float len = this->length();

    if (len > 0.0f)
    {
        this->x = this->x / len;
        this->y = this->y / len;
    }
}

Vector2 Vector2::normalized()
{
    float len = this->length();
    Vector2 normalized;

    if (len > 0.0)
    {
        normalized.x = this->x / len;
        normalized.y = this->y / len;
    }

    return normalized;
}

void Vector2::set(float x, float y)
{
    this->x = x;
    this->y = y;
}

void Vector2::set(float* vec)
{
    this->x = vec[0];
    this->y = vec[1];
}

Vector2& Vector2::get()
{
    return *this;
}

void Vector2::print()
{
    printf("x: %f, y: %f\n", x, y);
}

Vector2& Vector2::operator =(const Vector2& a)
{
    this->x = a.x;
    this->y = a.y;

    return *this;
}

Vector2& Vector2::operator +=(const Vector2& a)
{
    this->x += a.x;
    this->y += a.y;

    return *this;
}

Vector2& Vector2::operator +=(float s)
{
    this->x += s;
    this->y += s;

    return *this;
}

Vector2& Vector2::operator -=(const Vector2& a)
{
    this->x -= a.x;
    this->y -= a.y;

    return *this;
}

Vector2& Vector2::operator -=(float s)
{
    this->x -= s;
    this->y -= s;

    return *this;
}

Vector2& Vector2::operator *=(const Vector2& a)
{
    this->x *= a.x;
    this->y *= a.y;

    return *this;
}

Vector2& Vector2::operator *=(float s)
{
    this->x *= s;
    this->y *= s;

    return *this;
}

Vector2& Vector2::operator /=(const Vector2& a)
{
    this->x /= a.x;
    this->y /= a.y;

    return *this;
}

Vector2& Vector2::operator /=(float s)
{
    this->x /= s;
    this->y /= s;

    return *this;
}

Vector2 operator +(const Vector2& a, const Vector2& b)
{
    Vector2 r;

    r.x = a.x + b.x;
    r.y = a.y + b.y;

    return r;
}

Vector2 operator +(const Vector2& a, float s)
{
    Vector2 r;

    r.x = a.x + s;
    r.y = a.y + s;

    return r;
}

Vector2 operator +(float s, const Vector2& a)
{
    Vector2 r;

    r.x = a.x + s;
    r.y = a.y + s;

    return r;
}

Vector2 operator -(const Vector2& a, const Vector2& b)
{
    Vector2 r;

    r.x = a.x - b.x;
    r.y = a.y - b.y;

    return r;
}

Vector2 operator -(const Vector2& a, float s)
{
    Vector2 r;

    r.x = a.x - s;
    r.y = a.y - s;

    return r;
}

Vector2 operator -(const Vector2& a)
{
    Vector2 r;

    r.x = -a.x;
    r.y = -a.y;

    return r;
}

Vector2 operator *(const Vector2& a, float s)
{
    Vector2 r;

    r.x = a.x * s;
    r.y = a.y * s;

    return r;
}

Vector2 operator *(const Vector2& a, const Vector2& b)
{
    Vector2 r;

    r.x = a.x * b.x;
    r.y = a.y * b.y;

    return r;
}

Vector2 operator /(const Vector2& a, float s)
{
    Vector2 r;

    r.x = a.x / s;
    r.y = a.y / s;

    return r;
}

Vector2 operator /(const Vector2& a, const Vector2& b)
{
    Vector2 r;

    r.x = a.x / b.x;
    r.y = a.y / b.y;

    return r;
}

bool operator == (const Vector2& a, const Vector2& b)
{
    return(a.x == b.x && a.y == b.y);
}

bool operator != (const Vector2& a, const Vector2& b)
{
    return(a.x != b.x || a.y != b.y);
}

bool operator <= (const Vector2& a, const Vector2& b)
{
    return(a.x <= b.x && a.y <= b.y);
}

bool operator < (const Vector2& a, const Vector2& b)
{
    return(a.x < b.x&& a.y < b.y);
}

bool operator >= (const Vector2& a, const Vector2& b)
{
    return(a.x >= b.x && a.y >= b.y);
}

bool operator > (const Vector2& a, const Vector2& b)
{
    return(a.x > b.x && a.y > b.y);
}

Vector2 Vector2::mulMatrix(float* matrix)
{
    Vector2 result;

    result.x = matrix[0] * this->x + matrix[1] * this->y;
    result.y = matrix[2] * this->x + matrix[3] * this->y;

    return result;
}

__host__ __device__  float cross(const Vector2& a, const Vector2& b)
{
    return (a.x * b.y - a.y * b.x);
}

__host__ __device__ Vector2 normalize(const Vector2& v)
{
    float len = v.length();
    Vector2 normalized;

    if (len > 0.0)
    {
        normalized.x = v.x / len;
        normalized.y = v.y / len;
    }

    return normalized;
}

__host__ __device__ float dot(const Vector2& a, const Vector2& b)
{
    float dot = 0.0f;

    dot = (a.x * b.x) + (a.y * b.y);

    return dot;
}

__host__ __device__ float length(const Vector2& v)
{
    float length = 0.0f;

    length = sqrtf((v.x * v.x) + (v.y * v.y));

    return length;
}

__host__ __device__ Vector2 operator *(float s, const Vector2& v)
{
    return (v * s);
}

__host__ __device__ Vector2 vec2_rnd(float low, float high)
{
    vec2 v;

    v.x = (rand() / (static_cast<float>(RAND_MAX) + 1.0f)) * (high - low) + low;
    v.y = (rand() / (static_cast<float>(RAND_MAX) + 1.0f)) * (high - low) + low;

    return v;
}


Vector3::Vector3()
    : x(0.0),
    y(0.0),
    z(0.0)
{
}

Vector3::Vector3(float x, float y, float z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}

Vector3::Vector3(const Vector3& v)
{
    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
}

Vector3::Vector3(const Vector4& v)
{
    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
}

Vector3::Vector3(const float* v)
{
    this->x = v[0];
    this->y = v[1];
    this->z = v[2];
}

Vector3::Vector3(float x)
{
    this->x = x;
    this->y = x;
    this->z = x;
}

Vector3::~Vector3()
{
}

float Vector3::length() const
{
    float length = 0.0f;

    length = sqrtf((this->x * this->x) + (this->y * this->y) + (this->z * this->z));

    return length;
}

float Vector3::dot(const Vector3& vec)
{
    float dot = 0.0f;

    dot = (this->x * vec.x) + (this->y * vec.y) + (this->z * vec.z);

    return dot;
}

float Vector3::angle(Vector3& vec)
{
    float d = this->dot(vec);

    float a = this->length();
    float b = vec.length();

    float s = d / (a * b);
    float angle = (float)acos((double)s);

    return angle;
}

Vector3 Vector3::cross(const Vector3& vec)
{
    Vector3 cross;

    cross.x = (this->y * vec.z) - (this->z * vec.y);
    cross.y = (this->z * vec.x) - (this->x * vec.z);
    cross.z = (this->x * vec.y) - (this->y * vec.x);

    return cross;
}

void Vector3::normalize()
{
    float len = this->length();

    if (len > 0.0f)
    {
        this->x = this->x / len;
        this->y = this->y / len;
        this->z = this->z / len;
    }
}

Vector3 Vector3::normalized()
{
    float len = this->length();
    Vector3 normalized;

    if (len > 0.0)
    {
        normalized.x = this->x / len;
        normalized.y = this->y / len;
        normalized.z = this->z / len;
    }

    return normalized;
}

void Vector3::set(float x, float y, float z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}

void Vector3::set(float* vec)
{
    this->x = vec[0];
    this->y = vec[1];
    this->z = vec[2];
}

Vector3& Vector3::get()
{
    return *this;
}

void Vector3::print()
{
    printf("x: %f, y: %f, z: %f\n", x, y, z);
}

Vector3& Vector3::operator =(const Vector3& a)
{
    this->x = a.x;
    this->y = a.y;
    this->z = a.z;

    return *this;
}

Vector3& Vector3::operator +=(const Vector3& a)
{
    this->x += a.x;
    this->y += a.y;
    this->z += a.z;

    return *this;
}

Vector3& Vector3::operator +=(float s)
{
    this->x += s;
    this->y += s;
    this->z += s;

    return *this;
}

Vector3& Vector3::operator -=(const Vector3& a)
{
    this->x -= a.x;
    this->y -= a.y;
    this->z -= a.z;

    return *this;
}

Vector3& Vector3::operator -=(float s)
{
    this->x -= s;
    this->y -= s;
    this->z -= s;

    return *this;
}

Vector3& Vector3::operator *=(const Vector3& a)
{
    this->x *= a.x;
    this->y *= a.y;
    this->z *= a.z;

    return *this;
}

Vector3& Vector3::operator *=(float s)
{
    this->x *= s;
    this->y *= s;
    this->z *= s;

    return *this;
}

Vector3& Vector3::operator /=(const Vector3& a)
{
    this->x /= a.x;
    this->y /= a.y;
    this->z /= a.z;

    return *this;
}

Vector3& Vector3::operator /=(float s)
{
    this->x /= s;
    this->y /= s;
    this->z /= s;

    return *this;
}

Vector3 operator +(const Vector3& a, const Vector3& b)
{
    Vector3 r;

    r.x = a.x + b.x;
    r.y = a.y + b.y;
    r.z = a.z + b.z;

    return r;
}

Vector3 operator +(const Vector3& a, float s)
{
    Vector3 r;

    r.x = a.x + s;
    r.y = a.y + s;
    r.z = a.z + s;

    return r;
}

Vector3 operator +(float s, const Vector3& a)
{
    Vector3 r;

    r.x = a.x + s;
    r.y = a.y + s;
    r.z = a.z + s;

    return r;
}

Vector3 operator -(const Vector3& a, const Vector3& b)
{
    Vector3 r;

    r.x = a.x - b.x;
    r.y = a.y - b.y;
    r.z = a.z - b.z;

    return r;
}

Vector3 operator -(const Vector3& a, float s)
{
    Vector3 r;

    r.x = a.x - s;
    r.y = a.y - s;
    r.z = a.z - s;

    return r;
}

Vector3 operator -(const Vector3& a)
{
    Vector3 r;

    r.x = -a.x;
    r.y = -a.y;
    r.z = -a.z;

    return r;
}

Vector3 operator *(const Vector3& a, float s)
{
    Vector3 r;

    r.x = a.x * s;
    r.y = a.y * s;
    r.z = a.z * s;

    return r;
}

Vector3 operator *(const Vector3& a, const Vector3& b)
{
    Vector3 r;

    r.x = a.x * b.x;
    r.y = a.y * b.y;
    r.z = a.z * b.z;

    return r;
}

Vector3 operator /(const Vector3& a, float s)
{
    Vector3 r;

    r.x = a.x / s;
    r.y = a.y / s;
    r.z = a.z / s;

    return r;
}

Vector3 operator /(const Vector3& a, const Vector3& b)
{
    Vector3 r;

    r.x = a.x / b.x;
    r.y = a.y / b.y;
    r.z = a.z / b.z;

    return r;
}

bool operator == (const Vector3& a, const Vector3& b)
{
    return(a.x == b.x && a.y == b.y && a.z == b.z);
}

bool operator != (const Vector3& a, const Vector3& b)
{
    return(a.x != b.x || a.y != b.y || a.z != b.z);
}

bool operator <= (const Vector3& a, const Vector3& b)
{
    return(a.x <= b.x && a.y <= b.y && a.z <= b.z);
}

bool operator < (const Vector3& a, const Vector3& b)
{
    return(a.x < b.x&& a.y < b.y&& a.z < b.z);
}

bool operator >= (const Vector3& a, const Vector3& b)
{
    return(a.x >= b.x && a.y >= b.y && a.z >= b.z);
}

bool operator > (const Vector3& a, const Vector3& b)
{
    return(a.x > b.x && a.y > b.y && a.z > b.z);
}

Vector3 Vector3::mulMatrix(float* matrix)
{
    Vector3 result;

    result.x = matrix[0] * this->x + matrix[1] * this->y + matrix[2] * this->z;
    result.y = matrix[3] * this->x + matrix[4] * this->y + matrix[5] * this->z;
    result.z = matrix[6] * this->x + matrix[7] * this->y + matrix[8] * this->z;

    return result;
}

Vector3 cross(const Vector3& a, const Vector3& b)
{
    Vector3 c;

    c.x = (a.y * b.z) - (a.z * b.y);
    c.y = (a.z * b.x) - (a.x * b.z);
    c.z = (a.x * b.y) - (a.y * b.x);

    return c;
}

Vector3 normalize(const Vector3& v)
{
    float len = v.length();
    Vector3 normalized;

    if (len > 0.0)
    {
        normalized.x = v.x / len;
        normalized.y = v.y / len;
        normalized.z = v.z / len;
    }

    return normalized;
}

float dot(const Vector3& a, const Vector3& b)
{
    float dot = 0.0f;

    dot = (a.x * b.x) + (a.y * b.y) + (a.z * b.z);

    return dot;
}

float length(const Vector3& v)
{
    float length = 0.0f;

    length = sqrtf((v.x * v.x) + (v.y * v.y) + (v.z * v.z));

    return length;
}

float lengthSq(const vec3& v)
{
    return (v.x * v.x) + (v.y * v.y) + (v.z * v.z);
}

vec3 abs(const Vector3& v)
{
    vec3 r;

    r.x = abs(r.x);
    r.y = abs(r.y);
    r.z = abs(r.z);

    return r;
}

Vector3 operator *(float s, const Vector3& v)
{
    return (v * s);
}

Vector3 vec3_rnd(float low, float high)
{
    vec3 v;

    v.x = (rand() / (static_cast<float>(RAND_MAX) + 1.0f)) * (high - low) + low;
    v.y = (rand() / (static_cast<float>(RAND_MAX) + 1.0f)) * (high - low) + low;
    v.z = (rand() / (static_cast<float>(RAND_MAX) + 1.0f)) * (high - low) + low;

    return v;
}

Vector3 vec3_rnd_pos_y(float low, float high)
{
    vec3 v;

    v.x = (rand() / (static_cast<float>(RAND_MAX) + 1.0f)) * (high - low) + low;
    v.y = (rand() / (static_cast<float>(RAND_MAX) + 1.0f)) * (2 * high);
    v.z = (rand() / (static_cast<float>(RAND_MAX) + 1.0f)) * (high - low) + low;

    return v;
}

Vector3 step(const vec3& edge, const vec3& x)
{
    vec3 r(1.0f, 1.0f, 1.0f);

    if (x.x < edge.x)
        r.x = 0.0;

    if (x.y < edge.y)
        r.y = 0.0;

    if (x.z < edge.z)
        r.z = 0.0;

    return r;
}

Vector3 reflect(const vec3& I, const vec3& N)
{
    return I - 2.0 * dot(N, I) * N;
}


Vector4::Vector4()
    : x(0.0),
    y(0.0),
    z(0.0),
    w(0.0)
{
}

Vector4::Vector4(float x, float y, float z, float w)
{
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}

Vector4::Vector4(const Vector4& vec)
{
    this->x = vec.x;
    this->y = vec.y;
    this->z = vec.z;
    this->w = vec.w;
}

Vector4::Vector4(const Vector3& vec)
{
    this->x = vec.x;
    this->y = vec.y;
    this->z = vec.z;
    this->w = 1;
}

Vector4::Vector4(const Vector3& vec, float v)
{
    this->x = vec.x;
    this->y = vec.y;
    this->z = vec.z;
    this->w = v;
}

Vector4::Vector4(const float3& vec, float v)
{
    this->x = vec.x;
    this->y = vec.y;
    this->z = vec.z;
    this->w = v;
}

Vector4::Vector4(const float* vec)
{
    this->x = vec[0];
    this->y = vec[1];
    this->z = vec[2];
    this->w = vec[3];
}

Vector4::Vector4(float x)
{
    this->x = x;
    this->y = x;
    this->z = x;
    this->w = x;
}

Vector4::~Vector4()
{
}

float Vector4::length() const
{
    float length = 0.0f;

    length = sqrtf((this->x * this->x) + (this->y * this->y) + (this->z * this->z) + (this->w * this->w));

    return length;
}

float Vector4::dot(const Vector4& vec)
{
    float dot = 0.0f;

    dot = (this->x * vec.x) + (this->y * vec.y) + (this->z * vec.z) + (this->w * vec.w);

    return dot;
}

float Vector4::angle(Vector4& vec)
{
    float d = this->dot(vec);

    float a = this->length();
    float b = vec.length();

    float s = d / (a * b);
    float angle = (float)acos((double)s);

    return angle;
}

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Vector4 Vector4::cross(const Vector4& vec)
{
    Vector4 cross;

    cross.x = (this->y * vec.z) - (this->z * vec.y);
    cross.y = (this->z * vec.x) - (this->x * vec.z);
    cross.z = (this->x * vec.y) - (this->y * vec.x);
    cross.w = 1.0f;

    return cross;
}

void Vector4::normalize()
{
    float len = this->length();

    if (len > 0.0f)
    {
        this->x = this->x / len;
        this->y = this->y / len;
        this->z = this->z / len;
        this->w = this->w / len;
    }
}

Vector4 Vector4::normalized()
{
    float len = this->length();
    Vector4 normalized;

    if (len > 0.0)
    {
        normalized.x = this->x / len;
        normalized.y = this->y / len;
        normalized.z = this->z / len;
        normalized.w = this->w / len;
    }

    return normalized;
}

void Vector4::set(float x, float y, float z, float w)
{
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}

void Vector4::set(float* vec)
{
    this->x = vec[0];
    this->y = vec[1];
    this->z = vec[2];
    this->w = vec[3];
}

Vector4& Vector4::get()
{
    return *this;
}

void Vector4::print()
{
    printf("x: %f, y: %f, z: %f, w: %f\n", x, y, z, w);
}

Vector4& Vector4::operator =(const Vector4& a)
{
    this->x = a.x;
    this->y = a.y;
    this->z = a.z;
    this->w = a.w;

    return *this;
}

Vector4& Vector4::operator +=(const Vector4& a)
{
    this->x += a.x;
    this->y += a.y;
    this->z += a.z;
    this->w += a.w;

    return *this;
}

Vector4& Vector4::operator +=(float s)
{
    this->x += s;
    this->y += s;
    this->z += s;
    this->w += s;

    return *this;
}

Vector4& Vector4::operator -=(const Vector4& a)
{
    this->x -= a.x;
    this->y -= a.y;
    this->z -= a.z;
    this->w -= a.w;

    return *this;
}

Vector4& Vector4::operator -=(float s)
{
    this->x -= s;
    this->y -= s;
    this->z -= s;
    this->w -= s;

    return *this;
}

Vector4& Vector4::operator *=(const Vector4& a)
{
    this->x *= a.x;
    this->y *= a.y;
    this->z *= a.z;
    this->w *= a.w;

    return *this;
}

Vector4& Vector4::operator *=(float s)
{
    this->x *= s;
    this->y *= s;
    this->z *= s;
    this->w *= s;

    return *this;
}

Vector4& Vector4::operator /=(const Vector4& a)
{
    this->x /= a.x;
    this->y /= a.y;
    this->z /= a.z;
    this->w /= a.w;

    return *this;
}

Vector4& Vector4::operator /=(float s)
{
    this->x /= s;
    this->y /= s;
    this->z /= s;
    this->w /= s;

    return *this;
}

Vector4 operator +(const Vector4& a, const Vector4& b)
{
    Vector4 r;

    r.x = a.x + b.x;
    r.y = a.y + b.y;
    r.z = a.z + b.z;
    r.w = a.w + b.w;

    return r;
}

Vector4 operator +(const Vector4& a, float s)
{
    Vector4 r;

    r.x = a.x + s;
    r.y = a.y + s;
    r.z = a.z + s;
    r.w = a.w + s;

    return r;
}

Vector4 operator +(float s, const Vector4& a)
{
    Vector4 r;

    r.x = a.x + s;
    r.y = a.y + s;
    r.z = a.z + s;
    r.w = a.w + s;

    return r;
}

Vector4 operator -(const Vector4& a, const Vector4& b)
{
    Vector4 r;

    r.x = a.x - b.x;
    r.y = a.y - b.y;
    r.z = a.z - b.z;
    r.w = a.w - b.w;

    return r;
}

Vector4 operator -(const Vector4& a, float s)
{
    Vector4 r;

    r.x = a.x - s;
    r.y = a.y - s;
    r.z = a.z - s;
    r.w = a.w - s;

    return r;
}

Vector4 operator -(const Vector4& a)
{
    Vector4 r;

    r.x = -a.x;
    r.y = -a.y;
    r.z = -a.z;
    r.w = -a.w;

    return r;
}

Vector4 operator *(const Vector4& a, float s)
{
    Vector4 r;

    r.x = a.x * s;
    r.y = a.y * s;
    r.z = a.z * s;
    r.w = a.w * s;

    return r;
}

Vector4 operator *(const Vector4& a, const Vector4& b)
{
    Vector4 r;

    r.x = a.x * b.x;
    r.y = a.y * b.y;
    r.z = a.z * b.z;
    r.w = a.w * b.w;

    return r;
}

Vector4 operator /(const Vector4& a, float s)
{
    Vector4 r;

    r.x = a.x / s;
    r.y = a.y / s;
    r.z = a.z / s;
    r.w = a.w / s;

    return r;
}

Vector4 operator /(const Vector4& a, const Vector4& b)
{
    Vector4 r;

    r.x = a.x / b.x;
    r.y = a.y / b.y;
    r.z = a.z / b.z;
    r.w = a.w / b.w;

    return r;
}

bool operator == (const Vector4& a, const Vector4& b)
{
    return(a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w);
}

bool operator != (const Vector4& a, const Vector4& b)
{
    return(a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w);
}

bool operator <= (const Vector4& a, const Vector4& b)
{
    return(a.x <= b.x && a.y <= b.y && a.z <= b.z && a.w <= b.w);
}

bool operator < (const Vector4& a, const Vector4& b)
{
    return(a.x < b.x&& a.y < b.y&& a.z < b.z&& a.w < b.w);
}

bool operator >= (const Vector4& a, const Vector4& b)
{
    return(a.x >= b.x && a.y >= b.y && a.z >= b.z && a.w >= b.w);
}

bool operator > (const Vector4& a, const Vector4& b)
{
    return(a.x > b.x && a.y > b.y && a.z > b.z && a.w > b.w);
}

Vector4 cross(const Vector4& a, const Vector4& b)
{
    Vector4 c;

    c.x = (a.y * b.z) - (a.z * b.y);
    c.y = (a.z * b.x) - (a.x * b.z);
    c.z = (a.x * b.y) - (a.y * b.x);
    c.w = 1.0f;

    return c;
}

Vector4 normalize(const Vector4& v)
{
    float len = v.length();
    Vector4 normalized;

    if (len > 0.0)
    {
        normalized.x = v.x / len;
        normalized.y = v.y / len;
        normalized.z = v.z / len;
        normalized.w = v.w / len;
    }

    return normalized;
}

float dot(const Vector4& a, const Vector4& b)
{
    float dot = 0.0f;

    dot = (a.x * b.x) + (a.y * b.y) + (a.z * b.z) + (a.w * b.w);

    return dot;
}

float length(const Vector4& v)
{
    float length = 0.0f;

    length = sqrtf((v.x * v.x) + (v.y * v.y) + (v.z * v.z) + (v.w * v.w));

    return length;
}

vec4 abs(const Vector4& v)
{
    vec4 r;

    r.x = abs(r.x);
    r.y = abs(r.y);
    r.z = abs(r.z);
    r.w = abs(r.w);

    return r;
}

vec4 lessThan(const Vector4& v, const Vector4& w)
{
    vec4 r(0.0f, 0.0f, 0.0f, 0.0f);

    if (v.x < w.x)
        r.x = 1;

    if (v.y < w.y)
        r.y = 1;

    if (v.z < w.z)
        r.z = 1;

    if (v.w < w.w)
        r.w = 1;

    return r;
}

Vector4 floor(const Vector4& v)
{
    vec4 r;

    r.x = (float)floor(v.x);
    r.y = (float)floor(v.y);
    r.z = (float)floor(v.z);
    r.w = (float)floor(v.w);

    return r;
}

Vector4 vec4_rnd(float low, float high)
{
    vec4 v;

    v.x = (rand() / (static_cast<float>(RAND_MAX) + 1.0f)) * (high - low) + low;
    v.y = (rand() / (static_cast<float>(RAND_MAX) + 1.0f)) * (high - low) + low;
    v.z = (rand() / (static_cast<float>(RAND_MAX) + 1.0f)) * (high - low) + low;
    v.w = (rand() / (static_cast<float>(RAND_MAX) + 1.0f)) * (high - low) + low;

    return v;
}

Vector4 operator *(float s, const Vector4& v)
{
    return (v * s);
}



Matrix4x4::Matrix4x4()
    : a11(0.0f),
    a21(0.0f),
    a31(0.0f),
    a41(0.0f),
    a12(0.0f),
    a22(0.0f),
    a32(0.0f),
    a42(0.0f),
    a13(0.0f),
    a23(0.0f),
    a33(0.0f),
    a43(0.0f),
    a14(0.0f),
    a24(0.0f),
    a34(0.0f),
    a44(0.0f)
{
}

Matrix4x4::Matrix4x4(float a11, float a12, float a13, float a14,
    float a21, float a22, float a23, float a24,
    float a31, float a32, float a33, float a34,
    float a41, float a42, float a43, float a44)
{
    this->a11 = a11; this->a12 = a12; this->a13 = a13; this->a14 = a14;
    this->a21 = a21; this->a22 = a22; this->a23 = a23; this->a24 = a24;
    this->a31 = a31; this->a32 = a32; this->a33 = a33; this->a34 = a34;
    this->a41 = a41; this->a42 = a42; this->a43 = a43; this->a44 = a44;
}

Matrix4x4::Matrix4x4(const Matrix4x4& m)
{
    this->a11 = m.a11; this->a12 = m.a12; this->a13 = m.a13; this->a14 = m.a14;
    this->a21 = m.a21; this->a22 = m.a22; this->a23 = m.a23; this->a24 = m.a24;
    this->a31 = m.a31; this->a32 = m.a32; this->a33 = m.a33; this->a34 = m.a34;
    this->a41 = m.a41; this->a42 = m.a42; this->a43 = m.a43; this->a44 = m.a44;
}

Matrix4x4::Matrix4x4(const Matrix3x3& m)
{
    this->a11 = m.a11; this->a12 = m.a12; this->a13 = m.a13; this->a14 = 0.0f;
    this->a21 = m.a21; this->a22 = m.a22; this->a23 = m.a23; this->a24 = 0.0f;
    this->a31 = m.a31; this->a32 = m.a32; this->a33 = m.a33; this->a34 = 0.0f;
    this->a41 = 0.0f;  this->a42 = 0.0f;  this->a43 = 0.0f;  this->a44 = 1.0f;
}

Matrix4x4::Matrix4x4(float d)
{
    this->a11 = d;    this->a12 = 0.0f; this->a13 = 0.0f; this->a14 = 0.0f;
    this->a21 = 0.0f; this->a22 = d;    this->a23 = 0.0f; this->a24 = 0.0f;
    this->a31 = 0.0f; this->a32 = 0.0f; this->a33 = d;    this->a34 = 0.0f;
    this->a41 = 0.0f; this->a42 = 0.0f; this->a43 = 0.0f; this->a44 = d;
}

Matrix4x4::Matrix4x4(const float* vec)
{
    this->a11 = vec[0];  this->a12 = vec[1];  this->a13 = vec[2];  this->a14 = vec[3];
    this->a21 = vec[4];  this->a22 = vec[5];  this->a23 = vec[6];  this->a24 = vec[7];
    this->a31 = vec[8];  this->a32 = vec[9];  this->a33 = vec[10]; this->a34 = vec[11];
    this->a41 = vec[12]; this->a42 = vec[13]; this->a43 = vec[14]; this->a44 = vec[15];
}

Matrix4x4::~Matrix4x4()
{
}

void Matrix4x4::set(float a11, float a12, float a13, float a14, float a21, float a22, float a23, float a24,
    float a31, float a32, float a33, float a34, float a41, float a42, float a43, float a44)
{
    this->a11 = a11; this->a12 = a12; this->a13 = a13; this->a14 = a14;
    this->a21 = a21; this->a22 = a22; this->a23 = a23; this->a24 = a24;
    this->a31 = a31; this->a32 = a32; this->a33 = a33; this->a34 = a34;
    this->a41 = a41; this->a42 = a42; this->a43 = a43; this->a44 = a44;
}

void Matrix4x4::set(float* vec)
{
    this->a11 = vec[0];  this->a12 = vec[1];  this->a13 = vec[2];  this->a14 = vec[3];
    this->a21 = vec[4];  this->a22 = vec[5];  this->a23 = vec[6];  this->a24 = vec[7];
    this->a31 = vec[8];  this->a32 = vec[9];  this->a33 = vec[10]; this->a34 = vec[11];
    this->a41 = vec[12]; this->a42 = vec[13]; this->a43 = vec[14]; this->a44 = vec[15];
}

Matrix4x4& Matrix4x4::get()
{
    return *this;
}

void Matrix4x4::data(float* vec) const
{
    vec[0] = a11; vec[1] = a12; vec[2] = a13; vec[3] = a14;
    vec[4] = a21; vec[5] = a22; vec[6] = a23; vec[7] = a24;
    vec[8] = a31; vec[9] = a32; vec[10] = a33; vec[11] = a34;
    vec[12] = a41; vec[13] = a42; vec[14] = a43; vec[15] = a44;
}

Matrix4x4 Matrix4x4::inverse()
{
    float D = determinant();

    D = (D == 0) ? 1 : D;


    Matrix3x3 m11(a22, a23, a24, a32, a33, a34, a42, a43, a44);
    Matrix3x3 m12(a21, a23, a24, a31, a33, a34, a41, a43, a44);
    Matrix3x3 m13(a21, a22, a24, a31, a32, a34, a41, a42, a44);
    Matrix3x3 m14(a21, a22, a23, a31, a32, a33, a41, a42, a43);

    Matrix3x3 m21(a12, a13, a14, a32, a33, a34, a42, a43, a44);
    Matrix3x3 m22(a11, a13, a14, a31, a33, a34, a41, a43, a44);
    Matrix3x3 m23(a11, a12, a14, a31, a32, a34, a41, a42, a44);
    Matrix3x3 m24(a11, a12, a13, a31, a32, a33, a41, a42, a43);

    Matrix3x3 m31(a12, a13, a14, a22, a23, a24, a42, a43, a44);
    Matrix3x3 m32(a11, a13, a14, a21, a23, a24, a41, a43, a44);
    Matrix3x3 m33(a11, a12, a14, a21, a22, a24, a41, a42, a44);
    Matrix3x3 m34(a11, a12, a13, a21, a22, a23, a41, a42, a43);

    Matrix3x3 m41(a12, a13, a14, a22, a23, a24, a32, a33, a34);
    Matrix3x3 m42(a11, a13, a14, a21, a23, a24, a31, a33, a34);
    Matrix3x3 m43(a11, a12, a14, a21, a22, a24, a31, a32, a34);
    Matrix3x3 m44(a11, a12, a13, a21, a22, a23, a31, a32, a33);

    return Matrix4x4(
        m11.determinant() / D,
        -m21.determinant() / D,
        m31.determinant() / D,
        -m41.determinant() / D,

        -m12.determinant() / D,
        m22.determinant() / D,
        -m32.determinant() / D,
        m42.determinant() / D,

        m13.determinant() / D,
        -m23.determinant() / D,
        m33.determinant() / D,
        -m43.determinant() / D,

        -m14.determinant() / D,
        m24.determinant() / D,
        -m34.determinant() / D,
        m44.determinant() / D
    );
}

Matrix4x4 Matrix4x4::transpose()
{
    return Matrix4x4(a11, a21, a31, a41,
        a12, a22, a32, a42,
        a13, a23, a33, a43,
        a14, a24, a34, a44);
}

void Matrix4x4::setToIdentity()
{
    this->a11 = 1.0f; this->a12 = 0.0f; this->a13 = 0.0f; this->a14 = 0.0f;
    this->a21 = 0.0f; this->a22 = 1.0f; this->a23 = 0.0f; this->a24 = 0.0f;
    this->a31 = 0.0f; this->a31 = 0.0f; this->a33 = 1.0f; this->a34 = 0.0f;
    this->a41 = 0.0f; this->a41 = 0.0f; this->a43 = 0.0f; this->a44 = 1.0f;
}

void Matrix4x4::setToZero()
{
    this->a11 = 0.0f; this->a12 = 0.0f; this->a13 = 0.0f; this->a14 = 0.0f;
    this->a21 = 0.0f; this->a22 = 0.0f; this->a23 = 0.0f; this->a24 = 0.0f;
    this->a31 = 0.0f; this->a32 = 0.0f; this->a33 = 0.0f; this->a34 = 0.0f;
    this->a41 = 0.0f; this->a42 = 0.0f; this->a43 = 0.0f; this->a44 = 0.0f;
}

//!!!!!!!!!!!!!!!!!!! TODO !!!!!!!!!!!!!!!!!!!!!!!!!!
float Matrix4x4::determinant()
{
    float D1 = (a22 * a33 * a44) + (a23 * a34 * a42) + (a24 * a32 * a43)
        - ((a24 * a33 * a42) + (a22 * a34 * a43) + (a23 * a32 * a44));

    float D2 = (a21 * a33 * a44) + (a23 * a34 * a41) + (a24 * a31 * a43)
        - ((a24 * a33 * a41) + (a21 * a34 * a43) + (a23 * a31 * a44));

    float D3 = (a21 * a32 * a44) + (a22 * a34 * a41) + (a24 * a31 * a42)
        - ((a24 * a32 * a41) + (a21 * a34 * a42) + (a22 * a31 * a44));

    float D4 = (a21 * a32 * a43) + (a22 * a33 * a41) + (a23 * a31 * a42)
        - ((a23 * a32 * a41) + (a21 * a33 * a42) + (a22 * a31 * a43));


    float D = a11 * D1 - a12 * D2 + a13 * D3 - a14 * D4;

    return D;
}

void Matrix4x4::print()
{
    printf("(%.5f %.5f %.5f %.5f\n"
        "%.5f %.5f %.5f %.5f\n"
        "%.5f %.5f %.5f %.5f\n"
        "%.5f %.5f %.5f %.5f)\n",
        a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, a41, a42, a43, a44);
}

Matrix4x4& Matrix4x4::operator =(const Matrix4x4& a)
{
    this->a11 = a.a11; this->a12 = a.a12; this->a13 = a.a13; this->a14 = a.a14;
    this->a21 = a.a21; this->a22 = a.a22; this->a23 = a.a23; this->a24 = a.a24;
    this->a31 = a.a31; this->a32 = a.a32; this->a33 = a.a33; this->a34 = a.a34;
    this->a41 = a.a41; this->a42 = a.a42; this->a43 = a.a43; this->a44 = a.a44;

    return *this;
}

Matrix4x4& Matrix4x4::operator +=(const Matrix4x4& a)
{
    this->a11 += a.a11; this->a12 += a.a12; this->a13 += a.a13; this->a14 += a.a14;
    this->a21 += a.a21; this->a22 += a.a22; this->a23 += a.a23; this->a24 += a.a24;
    this->a31 += a.a31; this->a32 += a.a32; this->a33 += a.a33; this->a34 += a.a34;
    this->a41 += a.a41; this->a42 += a.a42; this->a43 += a.a43; this->a44 += a.a44;

    return *this;
}

Matrix4x4& Matrix4x4::operator +=(float s)
{
    this->a11 += s; this->a12 += s; this->a13 += s; this->a14 += s;
    this->a21 += s; this->a22 += s; this->a23 += s; this->a24 += s;
    this->a31 += s; this->a32 += s; this->a33 += s; this->a34 += s;
    this->a41 += s; this->a42 += s; this->a43 += s; this->a44 += s;

    return *this;
}

Matrix4x4& Matrix4x4::operator -=(const Matrix4x4& a)
{
    this->a11 -= a.a11; this->a12 -= a.a12; this->a13 -= a.a13; this->a14 -= a.a14;
    this->a21 -= a.a21; this->a22 -= a.a22; this->a23 -= a.a23; this->a24 -= a.a24;
    this->a31 -= a.a31; this->a32 -= a.a32; this->a33 -= a.a33; this->a34 -= a.a34;
    this->a41 -= a.a41; this->a42 -= a.a42; this->a43 -= a.a43; this->a44 -= a.a44;

    return *this;
}

Matrix4x4& Matrix4x4::operator -=(float s)
{
    this->a11 -= s; this->a12 -= s; this->a13 -= s; this->a14 -= s;
    this->a21 -= s; this->a22 -= s; this->a23 -= s; this->a24 -= s;
    this->a31 -= s; this->a32 -= s; this->a33 -= s; this->a34 -= s;
    this->a41 -= s; this->a42 -= s; this->a43 -= s; this->a44 -= s;

    return *this;
}

Matrix4x4& Matrix4x4::operator *=(const Matrix4x4& m)
{
    Matrix4x4 c;

    c.a11 = this->a11 * m.a11 + this->a12 * m.a21 + this->a13 * m.a31 + this->a14 * m.a41;
    c.a12 = this->a11 * m.a12 + this->a12 * m.a22 + this->a13 * m.a32 + this->a14 * m.a42;
    c.a13 = this->a11 * m.a13 + this->a12 * m.a23 + this->a13 * m.a33 + this->a14 * m.a43;
    c.a14 = this->a11 * m.a14 + this->a12 * m.a24 + this->a13 * m.a34 + this->a14 * m.a44;

    c.a21 = this->a21 * m.a11 + this->a22 * m.a21 + this->a23 * m.a31 + this->a24 * m.a41;
    c.a22 = this->a21 * m.a12 + this->a22 * m.a22 + this->a23 * m.a32 + this->a24 * m.a42;
    c.a23 = this->a21 * m.a13 + this->a22 * m.a23 + this->a23 * m.a33 + this->a24 * m.a43;
    c.a24 = this->a21 * m.a14 + this->a22 * m.a24 + this->a23 * m.a34 + this->a24 * m.a44;

    c.a31 = this->a31 * m.a11 + this->a32 * m.a21 + this->a33 * m.a31 + this->a34 * m.a41;
    c.a32 = this->a31 * m.a12 + this->a32 * m.a22 + this->a33 * m.a32 + this->a34 * m.a42;
    c.a33 = this->a31 * m.a13 + this->a32 * m.a23 + this->a33 * m.a33 + this->a34 * m.a43;
    c.a34 = this->a31 * m.a14 + this->a32 * m.a24 + this->a33 * m.a34 + this->a34 * m.a44;

    c.a41 = this->a41 * m.a11 + this->a42 * m.a21 + this->a43 * m.a31 + this->a44 * m.a41;
    c.a42 = this->a41 * m.a12 + this->a42 * m.a22 + this->a43 * m.a32 + this->a44 * m.a42;
    c.a43 = this->a41 * m.a13 + this->a42 * m.a23 + this->a43 * m.a33 + this->a44 * m.a43;
    c.a44 = this->a41 * m.a14 + this->a42 * m.a24 + this->a43 * m.a34 + this->a44 * m.a44;


    this->a11 = c.a11;
    this->a21 = c.a21;
    this->a31 = c.a31;
    this->a41 = c.a41;

    this->a12 = c.a12;
    this->a22 = c.a22;
    this->a32 = c.a32;
    this->a42 = c.a42;

    this->a13 = c.a13;
    this->a23 = c.a23;
    this->a33 = c.a33;
    this->a43 = c.a43;

    this->a14 = c.a14;
    this->a24 = c.a24;
    this->a34 = c.a34;
    this->a44 = c.a44;

    return *this;
}

Matrix4x4& Matrix4x4::operator *=(float s)
{
    this->a11 *= s; this->a12 *= s; this->a13 *= s; this->a14 *= s;
    this->a21 *= s; this->a22 *= s; this->a23 *= s; this->a24 *= s;
    this->a31 *= s; this->a32 *= s; this->a33 *= s; this->a34 *= s;
    this->a41 *= s; this->a42 *= s; this->a43 *= s; this->a44 *= s;

    return *this;
}

Matrix4x4& Matrix4x4::operator /=(float s)
{
    this->a11 /= s; this->a12 /= s; this->a13 /= s; this->a14 /= s;
    this->a21 /= s; this->a22 /= s; this->a23 /= s; this->a24 /= s;
    this->a31 /= s; this->a32 /= s; this->a33 /= s; this->a34 /= s;
    this->a41 /= s; this->a42 /= s; this->a43 /= s; this->a44 /= s;

    return *this;
}

Matrix4x4 operator +(const Matrix4x4& a, const Matrix4x4& b)
{
    Matrix4x4 r;

    r.a11 = a.a11 + b.a11;  r.a12 = a.a12 + b.a12;  r.a13 = a.a13 + b.a13;  r.a14 = a.a14 + b.a14;
    r.a21 = a.a21 + b.a21;  r.a22 = a.a22 + b.a22;  r.a23 = a.a23 + b.a23;  r.a24 = a.a24 + b.a24;
    r.a31 = a.a31 + b.a31;  r.a32 = a.a32 + b.a32;  r.a33 = a.a33 + b.a33;  r.a34 = a.a34 + b.a34;
    r.a41 = a.a41 + b.a41;  r.a42 = a.a42 + b.a42;  r.a43 = a.a43 + b.a43;  r.a44 = a.a44 + b.a44;

    return r;
}

Matrix4x4 operator +(const Matrix4x4& a, float s)
{
    Matrix4x4 r;

    r.a11 = a.a11 + s;  r.a12 = a.a12 + s;  r.a13 = a.a13 + s;  r.a14 = a.a14 + s;
    r.a21 = a.a21 + s;  r.a22 = a.a22 + s;  r.a23 = a.a23 + s;  r.a24 = a.a24 + s;
    r.a31 = a.a31 + s;  r.a32 = a.a32 + s;  r.a33 = a.a33 + s;  r.a34 = a.a34 + s;
    r.a31 = a.a41 + s;  r.a42 = a.a42 + s;  r.a43 = a.a43 + s;  r.a44 = a.a44 + s;

    return r;
}

Matrix4x4 operator +(float s, const Matrix4x4& a)
{
    Matrix4x4 r;

    r.a11 = a.a11 + s;  r.a12 = a.a12 + s;  r.a13 = a.a13 + s;  r.a14 = a.a14 + s;
    r.a21 = a.a21 + s;  r.a22 = a.a22 + s;  r.a23 = a.a23 + s;  r.a24 = a.a24 + s;
    r.a31 = a.a31 + s;  r.a32 = a.a32 + s;  r.a33 = a.a33 + s;  r.a34 = a.a34 + s;
    r.a41 = a.a41 + s;  r.a42 = a.a42 + s;  r.a43 = a.a43 + s;  r.a44 = a.a44 + s;

    return r;
}

Matrix4x4 operator -(const Matrix4x4& a, const Matrix4x4& b)
{
    Matrix4x4 r;

    r.a11 = a.a11 - b.a11;  r.a12 = a.a12 - b.a12;  r.a13 = a.a13 - b.a13;  r.a14 = a.a14 - b.a14;
    r.a21 = a.a21 - b.a21;  r.a22 = a.a22 - b.a22;  r.a23 = a.a23 - b.a23;  r.a24 = a.a24 - b.a24;
    r.a31 = a.a31 - b.a31;  r.a32 = a.a32 - b.a32;  r.a33 = a.a33 - b.a33;  r.a34 = a.a34 - b.a34;
    r.a41 = a.a41 - b.a41;  r.a42 = a.a42 - b.a42;  r.a43 = a.a43 - b.a43;  r.a44 = a.a44 - b.a44;

    return r;
}

Matrix4x4 operator -(const Matrix4x4& a, float s)
{
    Matrix4x4 r;

    r.a11 = a.a11 - s;  r.a12 = a.a12 - s;  r.a13 = a.a13 - s;  r.a14 = a.a14 - s;
    r.a21 = a.a21 - s;  r.a22 = a.a22 - s;  r.a23 = a.a23 - s;  r.a24 = a.a24 - s;
    r.a31 = a.a31 - s;  r.a32 = a.a32 - s;  r.a33 = a.a33 - s;  r.a34 = a.a34 - s;
    r.a41 = a.a41 - s;  r.a42 = a.a42 - s;  r.a43 = a.a43 - s;  r.a44 = a.a44 - s;

    return r;
}

Matrix4x4 operator -(const Matrix4x4& a)
{
    Matrix4x4 r;

    r.a11 = -a.a11;  r.a12 = -a.a12;  r.a13 = -a.a13;  r.a14 = -a.a14;
    r.a21 = -a.a21;  r.a22 = -a.a22;  r.a23 = -a.a23;  r.a24 = -a.a24;
    r.a31 = -a.a31;  r.a32 = -a.a32;  r.a33 = -a.a33;  r.a34 = -a.a34;
    r.a41 = -a.a41;  r.a42 = -a.a42;  r.a43 = -a.a43;  r.a44 = -a.a44;

    return r;
}

Matrix4x4 operator *(const Matrix4x4& a, float s)
{
    Matrix4x4 r;

    r.a11 = a.a11 * s;  r.a12 = a.a12 * s;  r.a13 = a.a13 * s;  r.a14 = a.a14 * s;
    r.a21 = a.a21 * s;  r.a22 = a.a22 * s;  r.a23 = a.a23 * s;  r.a24 = a.a24 * s;
    r.a31 = a.a31 * s;  r.a32 = a.a32 * s;  r.a33 = a.a33 * s;  r.a34 = a.a34 * s;
    r.a41 = a.a41 * s;  r.a42 = a.a42 * s;  r.a43 = a.a43 * s;  r.a44 = a.a44 * s;

    return r;
}

Matrix4x4 operator *(const Matrix4x4& a, const Matrix4x4& b)
{
    Matrix4x4 c;

    c.a11 = a.a11 * b.a11 + a.a12 * b.a21 + a.a13 * b.a31 + a.a14 * b.a41;
    c.a12 = a.a11 * b.a12 + a.a12 * b.a22 + a.a13 * b.a32 + a.a14 * b.a42;
    c.a13 = a.a11 * b.a13 + a.a12 * b.a23 + a.a13 * b.a33 + a.a14 * b.a43;
    c.a14 = a.a11 * b.a14 + a.a12 * b.a24 + a.a13 * b.a34 + a.a14 * b.a44;

    c.a21 = a.a21 * b.a11 + a.a22 * b.a21 + a.a23 * b.a31 + a.a24 * b.a41;
    c.a22 = a.a21 * b.a12 + a.a22 * b.a22 + a.a23 * b.a32 + a.a24 * b.a42;
    c.a23 = a.a21 * b.a13 + a.a22 * b.a23 + a.a23 * b.a33 + a.a24 * b.a43;
    c.a24 = a.a21 * b.a14 + a.a22 * b.a24 + a.a23 * b.a34 + a.a24 * b.a44;

    c.a31 = a.a31 * b.a11 + a.a32 * b.a21 + a.a33 * b.a31 + a.a34 * b.a41;
    c.a32 = a.a31 * b.a12 + a.a32 * b.a22 + a.a33 * b.a32 + a.a34 * b.a42;
    c.a33 = a.a31 * b.a13 + a.a32 * b.a23 + a.a33 * b.a33 + a.a34 * b.a43;
    c.a34 = a.a31 * b.a14 + a.a32 * b.a24 + a.a33 * b.a34 + a.a34 * b.a44;

    c.a41 = a.a41 * b.a11 + a.a42 * b.a21 + a.a43 * b.a31 + a.a44 * b.a41;
    c.a42 = a.a41 * b.a12 + a.a42 * b.a22 + a.a43 * b.a32 + a.a44 * b.a42;
    c.a43 = a.a41 * b.a13 + a.a42 * b.a23 + a.a43 * b.a33 + a.a44 * b.a43;
    c.a44 = a.a41 * b.a14 + a.a42 * b.a24 + a.a43 * b.a34 + a.a44 * b.a44;

    return c;

    //return mat4(a.a11 * b.a11 + a.a12 * b.a21 + a.a13 * b.a31 + a.a14 * b.a41,
    //            a.a11 * b.a12 + a.a12 * b.a22 + a.a13 * b.a32 + a.a14 * b.a42,
    //            a.a11 * b.a13 + a.a12 * b.a23 + a.a13 * b.a33 + a.a14 * b.a43,
    //            a.a11 * b.a14 + a.a12 * b.a24 + a.a13 * b.a34 + a.a14 * b.a44,
    //            a.a21 * b.a11 + a.a22 * b.a21 + a.a23 * b.a31 + a.a24 * b.a41,
    //            a.a21 * b.a12 + a.a22 * b.a22 + a.a23 * b.a32 + a.a24 * b.a42,
    //            a.a21 * b.a13 + a.a22 * b.a23 + a.a23 * b.a33 + a.a24 * b.a43,
    //            a.a21 * b.a14 + a.a22 * b.a24 + a.a23 * b.a34 + a.a24 * b.a44,
    //            a.a31 * b.a11 + a.a32 * b.a21 + a.a33 * b.a31 + a.a34 * b.a41,
    //            a.a31 * b.a12 + a.a32 * b.a22 + a.a33 * b.a32 + a.a34 * b.a42,
    //            a.a31 * b.a13 + a.a32 * b.a23 + a.a33 * b.a33 + a.a34 * b.a43,
    //            a.a31 * b.a14 + a.a32 * b.a24 + a.a33 * b.a34 + a.a34 * b.a44,
    //            a.a41 * b.a11 + a.a42 * b.a21 + a.a43 * b.a31 + a.a44 * b.a41,
    //            a.a41 * b.a12 + a.a42 * b.a22 + a.a43 * b.a32 + a.a44 * b.a42,
    //            a.a41 * b.a13 + a.a42 * b.a23 + a.a43 * b.a33 + a.a44 * b.a43,
    //            a.a41 * b.a14 + a.a42 * b.a24 + a.a43 * b.a34 + a.a44 * b.a44);

}

Matrix4x4 operator /(const Matrix4x4& a, float s)
{
    Matrix4x4 r;

    r.a11 = a.a11 / s;  r.a12 = a.a12 / s;  r.a13 = a.a13 / s;  r.a14 = a.a14 / s;
    r.a21 = a.a21 / s;  r.a22 = a.a22 / s;  r.a23 = a.a23 / s;  r.a24 = a.a24 / s;
    r.a31 = a.a31 / s;  r.a32 = a.a32 / s;  r.a33 = a.a33 / s;  r.a34 = a.a34 / s;
    r.a41 = a.a41 / s;  r.a42 = a.a42 / s;  r.a43 = a.a43 / s;  r.a44 = a.a44 / s;

    return r;
}

bool operator == (const Matrix4x4& a, const Matrix4x4& b)
{
    return(a.a11 == b.a11 && a.a12 == b.a12 && a.a13 == b.a13 && a.a14 == b.a14 &&
        a.a21 == b.a21 && a.a22 == b.a22 && a.a23 == b.a23 && a.a24 == b.a24 &&
        a.a31 == b.a31 && a.a32 == b.a32 && a.a33 == b.a33 && a.a34 == b.a34 &&
        a.a41 == b.a41 && a.a42 == b.a42 && a.a43 == b.a43 && a.a44 == b.a44);
}

bool operator != (const Matrix4x4& a, const Matrix4x4& b)
{
    return(a.a11 != b.a11 || a.a12 != b.a12 || a.a13 != b.a13 || a.a14 != b.a14 ||
        a.a21 != b.a21 || a.a22 != b.a22 || a.a23 != b.a23 || a.a24 != b.a24 ||
        a.a31 != b.a31 || a.a32 != b.a32 || a.a33 != b.a33 || a.a34 != b.a34 ||
        a.a31 != b.a41 || a.a42 != b.a42 || a.a43 != b.a43 || a.a44 != b.a44);
}

Matrix4x4 operator *(float s, const Matrix4x4& m)
{
    return (m * s);
}

float4 MatVecHost (const Matrix4x4& m, const float4& v)
{
    float4 r;

    r.x = m.a11 * v.x + m.a12 * v.y + m.a13 * v.z + m.a14 * v.w;
    r.y = m.a21 * v.x + m.a22 * v.y + m.a23 * v.z + m.a24 * v.w;
    r.z = m.a31 * v.x + m.a32 * v.y + m.a33 * v.z + m.a34 * v.w;
    r.w = m.a41 * v.x + m.a42 * v.y + m.a43 * v.z + m.a44 * v.w;

    return r;
}

float3 MatVecHost(const Matrix4x4& m, const float3& v)
{
    float4 res = MatVecHost(m, make_float4(v.x, v.y, v.z, 1.f));
    return make_float3(res.x, res.y, res.z);
}

Vector4 operator *(const Vector4& v, const Matrix4x4& m)
{
    Vector4 r;

    r.x = m.a11 * v.x + m.a21 * v.y + m.a31 * v.z + m.a41 * v.w;
    r.y = m.a12 * v.x + m.a22 * v.y + m.a32 * v.z + m.a42 * v.w;
    r.z = m.a13 * v.x + m.a23 * v.y + m.a33 * v.z + m.a43 * v.w;
    r.w = m.a14 * v.x + m.a24 * v.y + m.a34 * v.z + m.a44 * v.w;

    return r;
}

Vector3 operator *(const Matrix4x4& m, const Vector3& v)
{
    Vector4 r;

    r.x = m.a11 * v.x + m.a12 * v.y + m.a13 * v.z + m.a14 * 1.0f;
    r.y = m.a21 * v.x + m.a22 * v.y + m.a23 * v.z + m.a24 * 1.0f;
    r.z = m.a31 * v.x + m.a32 * v.y + m.a33 * v.z + m.a34 * 1.0f;
    r.w = m.a41 * v.x + m.a42 * v.y + m.a43 * v.z + m.a44 * 1.0f;

    return Vector3(r.x, r.y, r.z);
}

Matrix4x4 transpose(const Matrix4x4& m)
{
    Matrix4x4 mat(m);
    return mat.transpose();
}

Matrix4x4 inverse(const Matrix4x4& m)
{
    Matrix4x4 mat(m);
    return mat.inverse();
}

float anglefloat3(const float3& u, const float3& v)
{
    float d = u.x * v.x + u.y * v.y + u.z * v.z;

    float a = sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
    float b = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

    float s = d / (a * b);
    float angle = (float)acos((double)s);

    return angle;
}

const Matrix4x4 Matrix4x4::zero()
{
    return Matrix4x4(0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0);
}

const Matrix4x4 Matrix4x4::identitiy()
{
    return Matrix4x4(1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);
}

const Matrix4x4 Matrix4x4::rotateX(float angle)
{
    const float c = cosf(angle * math_radians);
    const float s = sinf(angle * math_radians);

    return Matrix4x4(1, 0, 0, 0,
        0, c, -s, 0,
        0, s, c, 0,
        0, 0, 0, 1);
}

const Matrix4x4 Matrix4x4::rotateY(float angle)
{
    const float c = cosf(angle * math_radians);
    const float s = sinf(angle * math_radians);

    return Matrix4x4(c, 0, s, 0,
        0, 1, 0, 0,
        -s, 0, c, 0,
        0, 0, 0, 1);
}

const Matrix4x4 Matrix4x4::rotateZ(float angle)
{
    const float c = cosf(angle * math_radians);
    const float s = sinf(angle * math_radians);

    return Matrix4x4(c, -s, 0, 0,
        s, c, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);
}

const Matrix4x4 Matrix4x4::rotate(float angle, float x, float y, float z)
{
    Vector3 n(x, y, z);
    return rotate(angle, n);
}

const Matrix4x4 Matrix4x4::rotate(float angle, const vec3& n)
{
    vec3 t = normalize(n);

    const float c = cosf(angle * math_radians);
    const float ac = 1.0f - c;
    const float s = sinf(angle * math_radians);

    float m11 = t.x * t.x * ac + c;
    float m12 = t.x * t.y * ac + t.z * s;
    float m13 = t.x * t.z * ac - t.y * s;

    float m21 = t.y * t.x * ac - t.z * s;
    float m22 = t.y * t.y * ac + c;
    float m23 = t.y * t.z * ac + t.x * s;

    float m31 = t.z * t.x * ac + t.y * s;
    float m32 = t.z * t.y * ac - t.x * s;
    float m33 = t.z * t.z * ac + c;

    return Matrix4x4(m11, m12, m13, 0.0f,
        m21, m22, m23, 0.0f,
        m31, m32, m33, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
}

const Matrix4x4 Matrix4x4::scale(float x, float y, float z)
{
    return Matrix4x4(x, 0, 0, 0,
        0, y, 0, 0,
        0, 0, z, 0,
        0, 0, 0, 1);
}

const Matrix4x4 Matrix4x4::scale(const Vector3& s)
{
    return scale(s.x, s.y, s.z);
}

const Matrix4x4 Matrix4x4::translate(float x, float y, float z)
{
    return Matrix4x4(1, 0, 0, x,
        0, 1, 0, y,
        0, 0, 1, z,
        0, 0, 0, 1);
}

const Matrix4x4 Matrix4x4::translate(const Vector3& t)
{
    return translate(t.x, t.y, t.z);
}

const Matrix4x4 Matrix4x4::orthographic(float left, float right, float bottom, float top, float zNear, float zFar)
{
    const float tx = -(right + left) / (right - left);
    const float ty = -(top + bottom) / (top - bottom);
    const float tz = -(zFar + zNear) / (zFar - zNear);

    return Matrix4x4(2 / (right - left), 0, 0, tx,
        0, 2 / (top - bottom), 0, ty,
        0, 0, -2 / (zFar - zNear), tz,
        0, 0, 0, 1);
}

const Matrix4x4 Matrix4x4::perspective(float fov, float ratio, float zNear, float zFar)
{
    const float aspect = ratio;

    const float f = 1.0f / tanf(fov * math_radians / 2.0f);
    const float A = (zFar + zNear) / (zNear - zFar);
    const float B = (2.0f * zFar * zNear) / (zNear - zFar);

    return Matrix4x4(f / aspect, 0.0f, 0.0f, 0.0f,
        0.0f, f, 0.0f, 0.0f,
        0.0f, 0.0f, A, B,
        0.0f, 0.0f, -1.0f, 0.0f);
}

const Matrix4x4 Matrix4x4::lookAt(const Vector3& position, const Vector3& center, const Vector3& up)
{
    const vec3 f = normalize(position - center);
    const vec3 s = normalize(cross(up, f));
    const vec3 u = normalize(cross(f, s));

    return Matrix4x4(s.x, s.y, s.z, -dot(s, position),
        u.x, u.y, u.z, -dot(u, position),
        f.x, f.y, f.z, -dot(f, position),
        0, 0, 0, 1);
}

const Matrix4x4 Matrix4x4::lookAt(float px, float py, float pz, float cx, float cy, float cz, float ux, float uy, float uz)
{
    return lookAt(Vector3(px, py, pz), Vector3(cx, cy, cz), Vector3(ux, uy, uz));
}

const Matrix3x3 Matrix4x4::normalMatrix(const Matrix4x4& model)
{
    Matrix4x4 m(model);
    Matrix3x3 inv = m.inverse();

    return inv.transpose();
}



Matrix3x3::Matrix3x3()
    : a11(0.0),
    a21(0.0),
    a31(0.0),
    a12(0.0),
    a22(0.0),
    a32(0.0),
    a13(0.0),
    a23(0.0),
    a33(0.0)
{
}

Matrix3x3::Matrix3x3(float u)
{
    a11 = u; a12 = u; a13 = u;
    a21 = u; a22 = u; a23 = u;
    a31 = u; a32 = u; a33 = u;
}

Matrix3x3::Matrix3x3(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33)
{
    this->a11 = a11; this->a12 = a12; this->a13 = a13;
    this->a21 = a21; this->a22 = a22; this->a23 = a23;
    this->a31 = a31; this->a32 = a32; this->a33 = a33;
}

Matrix3x3::Matrix3x3(const Matrix3x3& m)
{
    this->a11 = m.a11; this->a12 = m.a12; this->a13 = m.a13;
    this->a21 = m.a21; this->a22 = m.a22; this->a23 = m.a23;
    this->a31 = m.a31; this->a32 = m.a32; this->a33 = m.a33;
}

Matrix3x3::Matrix3x3(const Matrix4x4& m)
{
    this->a11 = m.a11; this->a12 = m.a12; this->a13 = m.a13;
    this->a21 = m.a21; this->a22 = m.a22; this->a23 = m.a23;
    this->a31 = m.a31; this->a32 = m.a32; this->a33 = m.a33;
}

Matrix3x3::Matrix3x3(const float* vec)
{
    this->a11 = vec[0]; this->a12 = vec[1]; this->a13 = vec[2];
    this->a21 = vec[3]; this->a22 = vec[4]; this->a23 = vec[5];
    this->a31 = vec[6]; this->a32 = vec[7]; this->a33 = vec[8];
}

Matrix3x3::~Matrix3x3()
{
}

void Matrix3x3::set(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33)
{
    this->a11 = a11; this->a12 = a12; this->a13 = a13;
    this->a21 = a21; this->a22 = a22; this->a23 = a23;
    this->a31 = a31; this->a32 = a32; this->a33 = a33;
}

void Matrix3x3::set(float* vec)
{
    this->a11 = vec[0]; this->a12 = vec[1]; this->a13 = vec[2];
    this->a21 = vec[3]; this->a22 = vec[4]; this->a23 = vec[5];
    this->a31 = vec[6]; this->a32 = vec[7]; this->a33 = vec[8];
}

Matrix3x3& Matrix3x3::get()
{
    return *this;
}

void Matrix3x3::data(float* vec) const
{
    vec[0] = a11; vec[1] = a12; vec[2] = a13;
    vec[3] = a21; vec[4] = a22; vec[5] = a23;
    vec[6] = a31; vec[7] = a32; vec[8] = a33;
}

Matrix3x3 Matrix3x3::inverse()
{
    float D = determinant();

    D = (D == 0) ? 1 : D;

    return Matrix3x3(
        (a22 * a33 - a23 * a32) / D,
        -(a12 * a33 - a13 * a32) / D,
        (a12 * a23 - a13 * a22) / D,
        -(a21 * a33 - a23 * a31) / D,
        (a11 * a33 - a13 * a31) / D,
        -(a11 * a23 - a13 * a21) / D,
        (a21 * a32 - a22 * a31) / D,
        -(a11 * a32 - a12 * a31) / D,
        (a11 * a22 - a12 * a21) / D);
}

Matrix3x3 Matrix3x3::transpose()
{
    return Matrix3x3(a11, a21, a31, a12, a22, a32, a13, a23, a33);
}

void Matrix3x3::setToIdentity()
{
    this->a11 = 1.0f; this->a12 = 0.0f; this->a13 = 0.0f;
    this->a21 = 0.0f; this->a22 = 1.0f; this->a23 = 0.0f;
    this->a31 = 0.0f; this->a31 = 0.0f; this->a33 = 1.0f;
}

void Matrix3x3::SetRow(const int& row, const float3& u)
{
    if (row == 0)
    {
        a11 = u.x;
        a12 = u.y;
        a13 = u.z;
    }
    else if (row == 1)
    {
        a21 = u.x;
        a22 = u.y;
        a23 = u.z;
    }
    else if (row == 2)
    {
        a31 = u.x;
        a32 = u.y;
        a33 = u.z;
    }
}

void Matrix3x3::setToZero()
{
    this->a11 = 0.0f; this->a12 = 0.0f; this->a13 = 0.0f;
    this->a21 = 0.0f; this->a22 = 0.0f; this->a23 = 0.0f;
    this->a31 = 0.0f; this->a32 = 0.0f; this->a33 = 0.0f;
}

float Matrix3x3::determinant()
{
    float D = (a11 * a22 * a33) + (a12 * a23 * a31) + (a13 * a21 * a32)
        - ((a13 * a22 * a31) + (a11 * a23 * a32) + (a12 * a21 * a33));

    return D;
}

void Matrix3x3::print()
{
    printf("a11: %.5f  a12: %.5f  a13: %.5f\na21: %.5f  a22: %.5f  a23: %.5f\na31: %.5f  a32: %.5f  a33: %.5f\n",
        a11, a12, a13, a21, a22, a23, a31, a32, a33);
}

Matrix3x3& Matrix3x3::operator =(const Matrix3x3& a)
{
    this->a11 = a.a11; this->a12 = a.a12; this->a13 = a.a13;
    this->a21 = a.a21; this->a22 = a.a22; this->a23 = a.a23;
    this->a31 = a.a31; this->a32 = a.a32; this->a33 = a.a33;

    return *this;
}

Matrix3x3& Matrix3x3::operator +=(const Matrix3x3& a)
{
    this->a11 += a.a11; this->a12 += a.a12; this->a13 += a.a13;
    this->a21 += a.a21; this->a22 += a.a22; this->a23 += a.a23;
    this->a31 += a.a31; this->a32 += a.a32; this->a33 += a.a33;

    return *this;
}

Matrix3x3& Matrix3x3::operator +=(float s)
{
    this->a11 += s; this->a12 += s; this->a13 += s;
    this->a21 += s; this->a22 += s; this->a23 += s;
    this->a31 += s; this->a32 += s; this->a33 += s;

    return *this;
}

Matrix3x3& Matrix3x3::operator -=(const Matrix3x3& a)
{
    this->a11 -= a.a11; this->a12 -= a.a12; this->a13 -= a.a13;
    this->a21 -= a.a21; this->a22 -= a.a22; this->a23 -= a.a23;
    this->a31 -= a.a31; this->a32 -= a.a32; this->a33 -= a.a33;

    return *this;
}

Matrix3x3& Matrix3x3::operator -=(float s)
{
    this->a11 -= s; this->a12 -= s; this->a13 -= s;
    this->a21 -= s; this->a22 -= s; this->a23 -= s;
    this->a31 -= s; this->a32 -= s; this->a33 -= s;

    return *this;
}

Matrix3x3& Matrix3x3::operator *=(const Matrix3x3& m)
{
    Matrix3x3 c;

    c.a11 = this->a11 * m.a11 + this->a12 * m.a21 + this->a13 * m.a31;
    c.a21 = this->a21 * m.a11 + this->a22 * m.a21 + this->a23 * m.a31;
    c.a31 = this->a31 * m.a11 + this->a32 * m.a21 + this->a33 * m.a31;

    c.a12 = this->a11 * m.a12 + this->a12 * m.a22 + this->a13 * m.a32;
    c.a22 = this->a21 * m.a12 + this->a22 * m.a22 + this->a23 * m.a32;
    c.a32 = this->a31 * m.a12 + this->a32 * m.a22 + this->a33 * m.a32;

    c.a13 = this->a11 * m.a13 + this->a12 * m.a23 + this->a13 * m.a33;
    c.a23 = this->a21 * m.a13 + this->a22 * m.a23 + this->a23 * m.a33;
    c.a33 = this->a31 * m.a13 + this->a32 * m.a23 + this->a33 * m.a33;

    this->a11 = c.a11;
    this->a21 = c.a21;
    this->a31 = c.a31;

    this->a12 = c.a12;
    this->a22 = c.a22;
    this->a32 = c.a32;

    this->a13 = c.a13;
    this->a23 = c.a23;
    this->a33 = c.a33;

    return *this;
}

Matrix3x3& Matrix3x3::operator *=(float s)
{
    this->a11 *= s; this->a12 *= s; this->a13 *= s;
    this->a21 *= s; this->a22 *= s; this->a23 *= s;
    this->a31 *= s; this->a32 *= s; this->a33 *= s;

    return *this;
}

Matrix3x3& Matrix3x3::operator /=(float s)
{
    this->a11 /= s; this->a12 /= s; this->a13 /= s;
    this->a21 /= s; this->a22 /= s; this->a23 /= s;
    this->a31 /= s; this->a32 /= s; this->a33 /= s;

    return *this;
}

Matrix3x3 operator +(const Matrix3x3& a, const Matrix3x3& b)
{
    Matrix3x3 r;

    r.a11 = a.a11 + b.a11;  r.a12 = a.a12 + b.a12;  r.a13 = a.a13 + b.a13;
    r.a21 = a.a21 + b.a21;  r.a22 = a.a22 + b.a22;  r.a23 = a.a23 + b.a23;
    r.a31 = a.a31 + b.a31;  r.a32 = a.a32 + b.a32;  r.a33 = a.a33 + b.a33;

    return r;
}

Matrix3x3 operator +(const Matrix3x3& a, float s)
{
    Matrix3x3 r;

    r.a11 = a.a11 + s;  r.a12 = a.a12 + s;  r.a13 = a.a13 + s;
    r.a21 = a.a21 + s;  r.a22 = a.a22 + s;  r.a23 = a.a23 + s;
    r.a31 = a.a31 + s;  r.a32 = a.a32 + s;  r.a33 = a.a33 + s;

    return r;
}

Matrix3x3 operator +(float s, const Matrix3x3& a)
{
    Matrix3x3 r;

    r.a11 = a.a11 + s;  r.a12 = a.a12 + s;  r.a13 = a.a13 + s;
    r.a21 = a.a21 + s;  r.a22 = a.a22 + s;  r.a23 = a.a23 + s;
    r.a31 = a.a31 + s;  r.a32 = a.a32 + s;  r.a33 = a.a33 + s;

    return r;
}

Matrix3x3 operator -(const Matrix3x3& a, const Matrix3x3& b)
{
    Matrix3x3 r;

    r.a11 = a.a11 - b.a11;  r.a12 = a.a12 - b.a12;  r.a13 = a.a13 - b.a13;
    r.a21 = a.a21 - b.a21;  r.a22 = a.a22 - b.a22;  r.a23 = a.a23 - b.a23;
    r.a31 = a.a31 - b.a31;  r.a32 = a.a32 - b.a32;  r.a33 = a.a33 - b.a33;

    return r;
}

Matrix3x3 operator -(const Matrix3x3& a, float s)
{
    Matrix3x3 r;

    r.a11 = a.a11 - s;  r.a12 = a.a12 - s;  r.a13 = a.a13 - s;
    r.a21 = a.a21 - s;  r.a22 = a.a22 - s;  r.a23 = a.a23 - s;
    r.a31 = a.a31 - s;  r.a32 = a.a32 - s;  r.a33 = a.a33 - s;

    return r;
}


Matrix3x3 operator -(const Matrix3x3& a)
{
    Matrix3x3 r;

    r.a11 = -a.a11;  r.a12 = -a.a12;  r.a13 = -a.a13;
    r.a21 = -a.a21;  r.a22 = -a.a22;  r.a23 = -a.a23;
    r.a31 = -a.a31;  r.a32 = -a.a32;  r.a33 = -a.a33;

    return r;
}

Matrix3x3 operator *(const Matrix3x3& a, float s)
{
    Matrix3x3 r;

    r.a11 = a.a11 * s;  r.a12 = a.a12 * s;  r.a13 = a.a13 * s;
    r.a21 = a.a21 * s;  r.a22 = a.a22 * s;  r.a23 = a.a23 * s;
    r.a31 = a.a31 * s;  r.a32 = a.a32 * s;  r.a33 = a.a33 * s;

    return r;
}

Matrix3x3 operator *(const Matrix3x3& a, const Matrix3x3& b)
{
    Matrix3x3 c;

    c.a11 = a.a11 * b.a11 + a.a12 * b.a21 + a.a13 * b.a31;
    c.a21 = a.a21 * b.a11 + a.a22 * b.a21 + a.a23 * b.a31;
    c.a31 = a.a31 * b.a11 + a.a32 * b.a21 + a.a33 * b.a31;

    c.a12 = a.a11 * b.a12 + a.a12 * b.a22 + a.a13 * b.a32;
    c.a22 = a.a21 * b.a12 + a.a22 * b.a22 + a.a23 * b.a32;
    c.a32 = a.a31 * b.a12 + a.a32 * b.a22 + a.a33 * b.a32;

    c.a13 = a.a11 * b.a13 + a.a12 * b.a23 + a.a13 * b.a33;
    c.a23 = a.a21 * b.a13 + a.a22 * b.a23 + a.a23 * b.a33;
    c.a33 = a.a31 * b.a13 + a.a32 * b.a23 + a.a33 * b.a33;

    return c;
}

Matrix3x3 operator /(const Matrix3x3& a, float s)
{
    Matrix3x3 r;

    r.a11 = a.a11 / s;  r.a12 = a.a12 / s;  r.a13 = a.a13 / s;
    r.a21 = a.a21 / s;  r.a22 = a.a22 / s;  r.a23 = a.a23 / s;
    r.a31 = a.a31 / s;  r.a32 = a.a32 / s;  r.a33 = a.a33 / s;

    return r;
}

bool operator == (const Matrix3x3& a, const Matrix3x3& b)
{
    return(a.a11 == b.a11 && a.a12 == b.a12 && a.a13 == b.a13 &&
        a.a21 == b.a21 && a.a22 == b.a22 && a.a23 == b.a23 &&
        a.a31 == b.a31 && a.a32 == b.a32 && a.a33 == b.a33);
}

bool operator != (const Matrix3x3& a, const Matrix3x3& b)
{
    return(a.a11 != b.a11 || a.a12 != b.a12 || a.a13 != b.a13 ||
        a.a21 != b.a21 || a.a22 != b.a22 || a.a23 != b.a23 ||
        a.a31 != b.a31 || a.a32 != b.a32 || a.a33 != b.a33);
}

Matrix3x3 operator *(float s, const Matrix3x3& m)
{
    return (m * s);
}

Vector3 operator *(const Matrix3x3& m, const Vector3& v)
{
    Vector3 r;

    r.x = m.a11 * v.x + m.a12 * v.y + m.a13 * v.z;
    r.y = m.a21 * v.x + m.a22 * v.y + m.a23 * v.z;
    r.z = m.a31 * v.x + m.a32 * v.y + m.a33 * v.z;

    return r;
}

Vector3 operator *(const Vector3& v, const Matrix3x3& m)
{
    Vector3 r;

    r.x = m.a11 * v.x + m.a21 * v.y + m.a31 * v.z;
    r.y = m.a12 * v.x + m.a22 * v.y + m.a32 * v.z;
    r.z = m.a13 * v.x + m.a23 * v.y + m.a33 * v.z;

    return r;
}

Matrix3x3 transpose(const Matrix3x3& m)
{
    Matrix3x3 mat(m);
    return mat.transpose();
}

Matrix3x3 inverse(const Matrix3x3& m)
{
    Matrix3x3 mat(m);
    return mat.inverse();
}


Quaternion::Quaternion()
    : x(0.0f),
    y(0.0f),
    z(0.0f),
    w(1.0f)
{
}

Quaternion::Quaternion(const float x, const float y, const float z, const float w)
{
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}

Quaternion::Quaternion(const Vector3& v, const float s)
{
    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
    this->w = s;
}

Quaternion::Quaternion(const float* f)
{
    this->x = f[0];
    this->y = f[1];
    this->z = f[2];
    this->w = f[3];
}

Quaternion::Quaternion(const Quaternion& q)
{
    this->x = q.x;
    this->y = q.y;
    this->z = q.z;
    this->w = q.w;
}

Quaternion::~Quaternion()
{
}

float Quaternion::length()
{
    float l = sqrtf(this->w * this->w + this->x * this->x + this->y * this->y + this->z * this->z);
    return l;
}

float Quaternion::length2()
{
    //To-Do!
    return 0;
}

Quaternion Quaternion::normalized()
{
    Quaternion r(*this);
    float l = this->length();

    r.x /= l;
    r.y /= l;
    r.z /= l;
    r.w /= l;

    return r;
}

void Quaternion::normalize()
{
    float l = this->length();

    this->x /= l;
    this->y /= l;
    this->z /= l;
    this->w /= l;
}

void Quaternion::set(const float x, const float y, const float z, const float w)
{
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}

void Quaternion::set(const Vector3& v, const float s)
{
    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
    this->w = s;
}

void Quaternion::set(const float* f)
{
    this->x = f[0];
    this->y = f[1];
    this->z = f[2];
    this->w = f[3];
}

void Quaternion::set(const Quaternion& q)
{
    this->x = q.x;
    this->y = q.y;
    this->z = q.z;
    this->w = q.w;
}

Quaternion& Quaternion::setEuler(const float Yaw, const float Pitch, const float Roll)
{
    float cosY = cosf(Yaw / 2.0f);
    float sinY = sinf(Yaw / 2.0f);
    float cosP = cosf(Pitch / 2.0f);
    float sinP = sinf(Pitch / 2.0f);
    float cosR = cosf(Roll / 2.0f);
    float sinR = sinf(Roll / 2.0f);

    set(
        cosR * sinP * cosY + sinR * cosP * sinY,
        cosR * cosP * sinY - sinR * sinP * cosY,
        sinR * cosP * cosY - cosR * sinP * sinY,
        cosR * cosP * cosY + sinR * sinP * sinY
    );

    return *this;
}

Quaternion Quaternion::slerp(const Quaternion& from, const Quaternion& to, const float t)
{
    Quaternion to1;
    float omega, cosom, sinom, scale0, scale1;

    //calculate cosine
    cosom = (from.x * to.x + from.y * to.y + from.z * to.z) + from.w + to.w;

    //Adjust signs (if necessary) 
    if (cosom < 0.0)
    {
        cosom = -cosom;
        to1 = -to;
    }
    else
    {
        to1 = to;
    }

    //Calculate coefficients
    if ((1.0 - cosom) > QUATERNION_ERROR)
    {
        //standard case (slerp)
        omega = (float)acos(cosom);
        sinom = (float)sin(omega);
        scale0 = (float)sin((1.0 - t) * omega) / sinom;
        scale1 = (float)sin(t * omega) / sinom;
    }
    else
    {
        //'from' and 'to' are very close - just do linear interpolation
        scale0 = 1.0f - t;
        scale1 = t;
    }

    return scale0 * from + scale1 * to1;
    return Quaternion();
}

void Quaternion::toMatrix(float* m)
{
    if (!m) return;

    // First row
    m[0] = 1.0f - 2.0f * (y * y + z * z);
    m[1] = 2.0f * (x * y + z * w);
    m[2] = 2.0f * (x * z - y * w);
    m[3] = 0.0f;

    // Second row
    m[4] = 2.0f * (x * y - z * w);
    m[5] = 1.0f - 2.0f * (x * x + z * z);
    m[6] = 2.0f * (z * y + x * w);
    m[7] = 0.0f;

    // Third row
    m[8] = 2.0f * (x * z + y * w);
    m[9] = 2.0f * (y * z - x * w);
    m[10] = 1.0f - 2.0f * (x * x + y * y);
    m[11] = 0.0f;

    // Fourth row
    m[12] = 0;
    m[13] = 0;
    m[14] = 0;
    m[15] = 1.0f;
}

void Quaternion::fromAxisAngle(float x, float y, float z, float degrees)
{
    // First we want to convert the degrees to radians 
    // since the angle is assumed to be in radians
    float angle = float((degrees / 180.0f) * QUATERNION_PI);

    // Here we calculate the sin( theta / 2) once for optimization
    float result = (float)sin(angle / 2.0f);

    // Calcualte the w value by cos( theta / 2 )
    this->w = (float)cos(angle / 2.0f);

    // Calculate the x, y and z of the quaternion
    this->x = float(x * result);
    this->y = float(y * result);
    this->z = float(z * result);
}

Quaternion& Quaternion::get()
{
    return *this;
}

void Quaternion::print()
{
    printf("v: %3.2f, %3.2f, %3.2f  s: %3.2f", this->x, this->y, this->z, this->w);
}

Quaternion& Quaternion::operator =(const Quaternion& q)
{
    this->x = q.x;
    this->y = q.y;
    this->z = q.z;
    this->w = q.w;

    return *this;
}

Quaternion& Quaternion::operator +=(const Quaternion& q)
{
    this->x += q.x;
    this->y += q.y;
    this->z += q.z;
    this->w += q.w;

    return *this;
}

Quaternion& Quaternion::operator -=(const Quaternion& q)
{
    this->x -= q.x;
    this->y -= q.y;
    this->z -= q.z;
    this->w -= q.w;

    return *this;
}

Quaternion& Quaternion::operator *=(const Quaternion& q)
{
    w = w * q.w - x * q.x - y * q.y - z * q.z;

    x = w * q.x + x * q.w + y * q.z - z * q.y;
    y = w * q.y + y * q.w + z * q.x - x * q.z;
    z = w * q.z + z * q.w + x * q.y - y * q.x;

    return *this;
}

Quaternion& Quaternion::operator *=(float s)
{
    this->x *= s;
    this->x *= s;
    this->z *= s;
    this->w *= s;

    return *this;
}

Quaternion& Quaternion::operator /=(float s)
{
    this->x /= s;
    this->y /= s;
    this->z /= s;
    this->w /= s;

    return *this;
}

Quaternion Quaternion::operator *(const Quaternion& q)
{
    Quaternion r;

    r.w = w * q.w - x * q.x - y * q.y - z * q.z;
    r.x = w * q.x + x * q.w + y * q.z - z * q.y;
    r.y = w * q.y + y * q.w + z * q.x - x * q.z;
    r.z = w * q.z + z * q.w + x * q.y - y * q.x;

    return(r);
}

Quaternion operator -(const Quaternion& a)
{
    return Quaternion(-a.x, -a.y, -a.z, -a.w);
}

Quaternion operator +(const Quaternion& a, const Quaternion& b)
{
    return Quaternion(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

Quaternion operator -(const Quaternion& a, const Quaternion& b)
{
    return Quaternion(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

Quaternion operator *(const Quaternion& a, const float f)
{
    return Quaternion(a.x * f, a.y * f, a.z * f, a.w * f);
}

Quaternion operator *(const float f, const Quaternion& a)
{
    return Quaternion(a.x * f, a.y * f, a.z * f, a.w * f);
}

Quaternion operator *(const Quaternion& a, const Quaternion& b)
{
    Quaternion r;

    r.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    r.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    r.y = a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z;
    r.z = a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x;

    return r;
}

bool operator == (const Quaternion& a, const Quaternion& b)
{
    return ((a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w));
}

bool operator != (const Quaternion& a, const Quaternion& b)
{
    return ((a.x != b.x) || (a.y != b.y) || (a.z != b.z) || (a.w != b.w));
}