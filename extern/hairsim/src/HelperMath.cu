#include <HelperMath.cuh>

__host__ __device__ Mat3::Mat3()
{
	M11 = 0.f, M12 = 0.f, M13 = 0.f;
	M21 = 0.f, M22 = 0.f, M23 = 0.f;
	M31 = 0.f, M32 = 0.f, M33 = 0.f;
}

__host__ __device__ Mat3::Mat3(float m11, float m12, float m13, float m21, float m22, float m23, float m31, float m32, float m33)
{
	M11 = m11, M12 = m12, M13 = m13;
	M21 = m21, M22 = m22, M23 = m23;
	M31 = m31, M32 = m32, M33 = m33;
}

__host__ __device__ Mat3::Mat3(const float3& a, const float3& b)
{
	// matrix given by a*b^{T}
	M11 = a.x * b.x, M12 = a.x * b.y, M13 = a.x * b.z;
	M21 = a.y * b.x, M22 = a.y * b.y, M23 = a.y * b.z;
	M31 = a.z * b.x, M32 = a.z * b.y, M33 = a.z * b.z;
}

__host__ __device__ Mat3::Mat3(const float3& a)
{
	// matrix given by a*a^{T}
	M11 = a.x * a.x, M12 = a.x * a.y, M13 = a.x * a.z;
	M21 = a.y * a.x, M22 = a.y * a.y, M23 = a.y * a.z;
	M31 = a.z * a.x, M32 = a.z * a.y, M33 = a.z * a.z;
}

__host__ __device__ Mat3 Mat3::Identity()
{
	return Mat3(1.f, 0.f, 0.f,
				0.f, 1.f, 0.f,
				0.f, 0.f, 1.f);
}

__host__ __device__ Mat3 Mat3::Zero()
{
	return Mat3(0.f, 0.f, 0.f,
				0.f, 0.f, 0.f,
				0.f, 0.f, 0.f);
}

__host__ __device__ void Mat3::Data(float* vec) const
{
	vec[0] = M11; vec[1] = M12; vec[2] = M13;
	vec[3] = M21; vec[4] = M22; vec[5] = M23;
	vec[6] = M31; vec[7] = M32; vec[8] = M33; 
}

__host__ __device__ void Mat3::Print()
{
	printf("(%f, %f, %f)\n", M11, M12, M13);
	printf("(%f, %f, %f)\n", M21, M22, M23);
	printf("(%f, %f, %f)\n", M31, M32, M33);
}

__host__ __device__ Mat4::Mat4()
{
	M11 = 0.f, M12 = 0.f, M13 = 0.f, M14 = 0.f;
	M21 = 0.f, M22 = 0.f, M23 = 0.f, M24 = 0.f;
	M31 = 0.f, M32 = 0.f, M33 = 0.f, M34 = 0.f;
	M41 = 0.f, M42 = 0.f, M43 = 0.f, M44 = 0.f;
}

__host__ __device__ float Angle(const float3& a, const float3& b)
{
	float d = dot(a, b);

	float al = length(a);
	float bl = length(b);

	float s = d / (al * bl);
	float angle = (float)acos((double)s);

	return angle;
}

__host__ __device__ Mat3 operator*(const float& r, const Mat3& a)
{
	Mat3 c;

	c.M11 = r * a.M11;
	c.M12 = r * a.M12;
	c.M13 = r * a.M13;

	c.M21 = r * a.M21;
	c.M22 = r * a.M22;
	c.M23 = r * a.M23;

	c.M31 = r * a.M31;
	c.M32 = r * a.M32;
	c.M33 = r * a.M33;


	return c;
}

__host__ __device__ Mat3 operator*(const Mat3& a, const Mat3& b)
{
	Mat3 c;

	c.M11 = a.M11 * b.M11 + a.M12 * b.M21 + a.M13 * b.M31;
	c.M12 = a.M11 * b.M12 + a.M12 * b.M22 + a.M13 * b.M32;
	c.M13 = a.M11 * b.M13 + a.M12 * b.M23 + a.M13 * b.M33;

	c.M21 = a.M21 * b.M11 + a.M22 * b.M21 + a.M23 * b.M31;
	c.M22 = a.M21 * b.M12 + a.M22 * b.M22 + a.M23 * b.M32;
	c.M23 = a.M21 * b.M13 + a.M22 * b.M23 + a.M23 * b.M33;

	c.M31 = a.M31 * b.M11 + a.M32 * b.M21 + a.M33 * b.M31;
	c.M32 = a.M31 * b.M12 + a.M32 * b.M22 + a.M33 * b.M32;
	c.M33 = a.M31 * b.M13 + a.M32 * b.M23 + a.M33 * b.M33;

	return c;
}

__host__ __device__ Mat3 operator+(const Mat3& a, const Mat3& b)
{
	Mat3 c;

	c.M11 = a.M11 + b.M11;
	c.M12 = a.M12 + b.M12;
	c.M13 = a.M13 + b.M13;

	c.M21 = a.M21 + b.M21;
	c.M22 = a.M22 + b.M22;
	c.M23 = a.M23 + b.M23;

	c.M31 = a.M31 + b.M31;
	c.M32 = a.M32 + b.M32;
	c.M33 = a.M33 + b.M33;

	return c;
}

__host__ __device__ Mat3 operator-(const Mat3& a, const Mat3& b)
{
	Mat3 c;

	c.M11 = a.M11 - b.M11;
	c.M12 = a.M12 - b.M12;
	c.M13 = a.M13 - b.M13;

	c.M21 = a.M21 - b.M21;
	c.M22 = a.M22 - b.M22;
	c.M23 = a.M23 - b.M23;

	c.M31 = a.M31 - b.M31;
	c.M32 = a.M32 - b.M32;
	c.M33 = a.M33 - b.M33;

	return c;
}

__host__ __device__ float3 operator*(const Mat3& a, const float3& b)
{
	float3 c;

	c.x = a.M11 * b.x + a.M12 * b.y + a.M13 * b.z;
	c.y = a.M21 * b.x + a.M22 * b.y + a.M23 * b.z;
	c.z = a.M31 * b.x + a.M32 * b.y + a.M33 * b.z;

	return c;
}

__host__ __device__ float Mat3::Det()
{
	float D = (M11 * M22 * M33) + (M12 * M23 * M31) + (M13 * M21 * M32)
		- ((M13 * M22 * M31) + (M11 * M23 * M32) + (M12 * M21 * M33));

	return D;
}

__host__ __device__ Mat3 Mat3::Inverse()
{
	float D = Det();

	D = (D == 0) ? 1 : D;

	return Mat3(
		(M22 * M33 - M23 * M32) / D,
		-(M12 * M33 - M13 * M32) / D,
		(M12 * M23 - M13 * M22) / D,
		-(M21 * M33 - M23 * M31) / D,
		(M11 * M33 - M13 * M31) / D,
		-(M11 * M23 - M13 * M21) / D,
		(M21 * M32 - M22 * M31) / D,
		-(M11 * M32 - M12 * M31) / D,
		(M11 * M22 - M12 * M21) / D);
}

__host__ __device__ float Mat3::NormInfty()
{
	float norm = 0.f;

	norm = fabs(M11) > norm ? fabs(M11) : norm;
	norm = fabs(M12) > norm ? fabs(M12) : norm;
	norm = fabs(M13) > norm ? fabs(M13) : norm;
	norm = fabs(M21) > norm ? fabs(M21) : norm;
	norm = fabs(M22) > norm ? fabs(M22) : norm;
	norm = fabs(M23) > norm ? fabs(M23) : norm;
	norm = fabs(M31) > norm ? fabs(M31) : norm;
	norm = fabs(M32) > norm ? fabs(M32) : norm;
	norm = fabs(M33) > norm ? fabs(M33) : norm;

	return norm;
}

__host__ __device__ float Mat3::GetEntry(int opt)
{
	switch (opt)
	{
	case(11):
	{
		return M11;
		break;
	}
	case(12):
	{
		return M12;
		break;
	}
	case(13):
	{
		return M13;
		break;
	}
	case(21):
	{
		return M21;
		break;
	}
	case(22):
	{
		return M22;
		break;
	}
	case(23):
	{
		return M23;
		break;
	}
	case(31):
	{
		return M31;
		break;
	}
	case(32):
	{
		return M32;
		break;
	}
	case(33):
	{
		return M33;
		break;
	}
	default:
		printf("! Invalid matrix entry\n");
		break;
	}
}

__host__ __device__ Mat4::Mat4(float m11, float m12, float m13, float m14, float m21, float m22, float m23, float m24, float m31, float m32, float m33, float m34, float m41, float m42, float m43, float m44)
{
	M11 = m11, M12 = m12, M13 = m13, M14 = m14;
	M21 = m21, M22 = m22, M23 = m23, M24 = m24;
	M31 = m31, M32 = m32, M33 = m33, M34 = m34;
	M41 = m41, M42 = m42, M43 = m43, M44 = m44;
}

__host__ __device__ Mat4 Mat4::Identity()
{
	return Mat4(1.f, 0.f, 0.f, 0.f,
				0.f, 1.f, 0.f, 0.f,
				0.f, 0.f, 1.f, 0.f,
				0.f, 0.f, 0.f, 1.f);
}

__host__ __device__ Mat4 Mat4::Zero()
{
	return Mat4(0.f, 0.f, 0.f, 0.f,
				0.f, 0.f, 0.f, 0.f,
				0.f, 0.f, 0.f, 0.f,
				0.f, 0.f, 0.f, 0.f);
}

__host__ __device__ Mat4 Mat4::Scale(const float3& s)
{
	return Mat4(s.x, 0.f, 0.f, 0.f,
				0.f, s.y, 0.f, 0.f,
				0.f, 0.f, s.z, 0.f,
				0.f, 0.f, 0.f, 1.f);
}

__host__ __device__ Mat4 Mat4::Translate(const float3& s)
{
	return Mat4(1.f, 0.f, 0.f, s.x,
				0.f, 1.f, 0.f, s.y,
				0.f, 0.f, 1.f, s.z,
				0.f, 0.f, 0.f, 1.f);
}

__host__ __device__ Mat4 Mat4::RotateX(const float& theta)
{
	MATH_CONSTANTS
	const float cos = cosf(theta * Math_radians);
	const float sin = sinf(theta * Math_radians);

	return Mat4(1.f, 0.f, 0.f, 0.f,
				0.f, cos, -sin, 0.f,
				0.f, sin, cos, 0.f,
				0.f, 0.f, 0.f, 1.f);
}

__host__ __device__ Mat4 Mat4::RotateY(const float& theta)
{
	MATH_CONSTANTS
	const float cos = cosf(theta * Math_radians);
	const float sin = sinf(theta * Math_radians);

	return Mat4(cos, 0.f, sin, 0.f,
				0.f, 1.f, 0.f, 0.f,
				-sin, 0.f, cos, 0.f,
				0.f, 0.f, 0.f, 1.f);
}

__host__ __device__ Mat4 Mat4::RotateZ(const float& theta)
{
	MATH_CONSTANTS
	const float cos = cosf(theta * Math_radians);
	const float sin = sinf(theta * Math_radians);

	return Mat4(cos, -sin, 0.f, 0.f,
				sin, cos, 0.f, 0.f,
				0.f, 0.f, 1.f, 0.f,
				0.f, 0.f, 0.f, 1.f);
}

__host__ __device__ Mat4 Mat4::Rotate(const float& theta, const float3& axis)
{
	MATH_CONSTANTS
	float3 t = normalize(axis);

	const float c = cosf(theta * Math_radians);
	const float ac = 1.0f - c;
	const float s = sinf(theta * Math_radians);

	float m11 = t.x * t.x * ac + c;
	float m12 = t.x * t.y * ac + t.z * s;
	float m13 = t.x * t.z * ac - t.y * s;

	float m21 = t.y * t.x * ac - t.z * s;
	float m22 = t.y * t.y * ac + c;
	float m23 = t.y * t.z * ac + t.x * s;

	float m31 = t.z * t.x * ac + t.y * s;
	float m32 = t.z * t.y * ac - t.x * s;
	float m33 = t.z * t.z * ac + c;

	return Mat4(m11, m12, m13, 0.0f,
		m21, m22, m23, 0.0f,
		m31, m32, m33, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
}

__host__ __device__ Mat4 Mat4::Transpose(const Mat4& A)
{
	return Mat4(A.M11, A.M21, A.M31, A.M41,
		A.M12, A.M22, A.M32, A.M42,
		A.M13, A.M23, A.M33, A.M43,
		A.M14, A.M24, A.M34, A.M44);
}

__host__ __device__ void Mat4::Data(float* vec) const
{
	vec[0] = M11; vec[1] = M12; vec[2] = M13; vec[3] = M14;
	vec[4] = M21; vec[5] = M22; vec[6] = M23; vec[7] = M24;
	vec[8] = M31; vec[9] = M32; vec[10] = M33; vec[11] = M34;
	vec[12] = M41; vec[13] = M42; vec[14] = M43; vec[15] = M44;
}

__host__ __device__ void Mat4::Print()
{
	printf("(%f, %f, %f, %f)\n", M11, M12, M13, M14);
	printf("(%f, %f, %f, %f)\n", M21, M22, M23, M24);
	printf("(%f, %f, %f, %f)\n", M31, M32, M33, M34);
	printf("(%f, %f, %f, %f)\n", M41, M42, M43, M44);
}

__host__ __device__ float Mat4::Det()
{
	float D1 = (M22 * M33 * M44) + (M23 * M34 * M42) + (M24 * M32 * M43)
		- ((M24 * M33 * M42) + (M22 * M34 * M43) + (M23 * M32 * M44));

	float D2 = (M21 * M33 * M44) + (M23 * M34 * M41) + (M24 * M31 * M43)
		- ((M24 * M33 * M41) + (M21 * M34 * M43) + (M23 * M31 * M44));

	float D3 = (M21 * M32 * M44) + (M22 * M34 * M41) + (M24 * M31 * M42)
		- ((M24 * M32 * M41) + (M21 * M34 * M42) + (M22 * M31 * M44));

	float D4 = (M21 * M32 * M43) + (M22 * M33 * M41) + (M23 * M31 * M42)
		- ((M23 * M32 * M41) + (M21 * M33 * M42) + (M22 * M31 * M43));


	float D = M11 * D1 - M12 * D2 + M13 * D3 - M14 * D4;

	return D;
}

__host__ __device__ Mat4 Mat4::Inverse()
{
	float D = Det();

	D = (D == 0) ? 1 : D;

	Mat3 m11(M22, M23, M24, M32, M33, M34, M42, M43, M44);
	Mat3 m12(M21, M23, M24, M31, M33, M34, M41, M43, M44);
	Mat3 m13(M21, M22, M24, M31, M32, M34, M41, M42, M44);
	Mat3 m14(M21, M22, M23, M31, M32, M33, M41, M42, M43);

	Mat3 m21(M12, M13, M14, M32, M33, M34, M42, M43, M44);
	Mat3 m22(M11, M13, M14, M31, M33, M34, M41, M43, M44);
	Mat3 m23(M11, M12, M14, M31, M32, M34, M41, M42, M44);
	Mat3 m24(M11, M12, M13, M31, M32, M33, M41, M42, M43);

	Mat3 m31(M12, M13, M14, M22, M23, M24, M42, M43, M44);
	Mat3 m32(M11, M13, M14, M21, M23, M24, M41, M43, M44);
	Mat3 m33(M11, M12, M14, M21, M22, M24, M41, M42, M44);
	Mat3 m34(M11, M12, M13, M21, M22, M23, M41, M42, M43);

	Mat3 m41(M12, M13, M14, M22, M23, M24, M32, M33, M34);
	Mat3 m42(M11, M13, M14, M21, M23, M24, M31, M33, M34);
	Mat3 m43(M11, M12, M14, M21, M22, M24, M31, M32, M34);
	Mat3 m44(M11, M12, M13, M21, M22, M23, M31, M32, M33);

	return Mat4(
		m11.Det() / D,
		-m21.Det() / D,
		m31.Det() / D,
		-m41.Det() / D,

		-m12.Det() / D,
		m22.Det() / D,
		-m32.Det() / D,
		m42.Det() / D,

		m13.Det() / D,
		-m23.Det() / D,
		m33.Det() / D,
		-m43.Det() / D,

		-m14.Det() / D,
		m24.Det() / D,
		-m34.Det() / D,
		m44.Det() / D
	);
}

__host__ __device__ float Mat4::GetEntry(int opt)
{
	switch (opt)
	{
	case(11):
	{
		return M11;
		break;
	}
	case(12):
	{
		return M12;
		break;
	}
	case(13):
	{
		return M13;
		break;
	}
	case(14):
	{
		return M14;
		break;
	}
	case(21):
	{
		return M21;
		break;
	}
	case(22):
	{
		return M22;
		break;
	}
	case(23):
	{
		return M23;
		break;
	}
	case(24):
	{
		return M24;
		break;
	}
	case(31):
	{
		return M31;
		break;
	}
	case(32):
	{
		return M32;
		break;
	}
	case(33):
	{
		return M33;
		break;
	}
	case(34):
	{
		return M34;
		break;
	}
	case(41):
	{
		return M41;
		break;
	}
	case(42):
	{
		return M42;
		break;
	}
	case(43):
	{
		return M43;
		break;
	}
	case(44):
	{
		return M44;
		break;
	}
	default:
		printf("! Invalid matrix entry\n");
		break;
	}
}

__host__ __device__ Mat4 operator *(const Mat4& a, const Mat4& b)
{
	Mat4 c;

	c.M11 = a.M11 * b.M11 + a.M12 * b.M21 + a.M13 * b.M31 + a.M14 * b.M41;
	c.M12 = a.M11 * b.M12 + a.M12 * b.M22 + a.M13 * b.M32 + a.M14 * b.M42;
	c.M13 = a.M11 * b.M13 + a.M12 * b.M23 + a.M13 * b.M33 + a.M14 * b.M43;
	c.M14 = a.M11 * b.M14 + a.M12 * b.M24 + a.M13 * b.M34 + a.M14 * b.M44;

	c.M21 = a.M21 * b.M11 + a.M22 * b.M21 + a.M23 * b.M31 + a.M24 * b.M41;
	c.M22 = a.M21 * b.M12 + a.M22 * b.M22 + a.M23 * b.M32 + a.M24 * b.M42;
	c.M23 = a.M21 * b.M13 + a.M22 * b.M23 + a.M23 * b.M33 + a.M24 * b.M43;
	c.M24 = a.M21 * b.M14 + a.M22 * b.M24 + a.M23 * b.M34 + a.M24 * b.M44;

	c.M31 = a.M31 * b.M11 + a.M32 * b.M21 + a.M33 * b.M31 + a.M34 * b.M41;
	c.M32 = a.M31 * b.M12 + a.M32 * b.M22 + a.M33 * b.M32 + a.M34 * b.M42;
	c.M33 = a.M31 * b.M13 + a.M32 * b.M23 + a.M33 * b.M33 + a.M34 * b.M43;
	c.M34 = a.M31 * b.M14 + a.M32 * b.M24 + a.M33 * b.M34 + a.M34 * b.M44;

	c.M41 = a.M41 * b.M11 + a.M42 * b.M21 + a.M43 * b.M31 + a.M44 * b.M41;
	c.M42 = a.M41 * b.M12 + a.M42 * b.M22 + a.M43 * b.M32 + a.M44 * b.M42;
	c.M43 = a.M41 * b.M13 + a.M42 * b.M23 + a.M43 * b.M33 + a.M44 * b.M43;
	c.M44 = a.M41 * b.M14 + a.M42 * b.M24 + a.M43 * b.M34 + a.M44 * b.M44;

	return c;
}

__host__ __device__ Mat4 operator*(const float& r, const Mat4& a)
{
	Mat4 c;

	c.M11 = r * a.M11;
	c.M12 = r * a.M12;
	c.M13 = r * a.M13;
	c.M14 = r * a.M14;

	c.M21 = r * a.M21;
	c.M22 = r * a.M22;
	c.M23 = r * a.M23;
	c.M24 = r * a.M24;

	c.M31 = r * a.M31;
	c.M32 = r * a.M32;
	c.M33 = r * a.M33;
	c.M34 = r * a.M34;

	c.M41 = r * a.M41;
	c.M42 = r * a.M42;
	c.M43 = r * a.M43;
	c.M44 = r * a.M44;

	return c;
}

__host__ __device__ float4 operator*(const Mat4& a, const float4& b)
{
	float4 c;
	c.x = a.M11 * b.x + a.M12 * b.y + a.M13 * b.z + a.M14 * b.w;
	c.y = a.M21 * b.x + a.M22 * b.y + a.M23 * b.z + a.M24 * b.w;
	c.z = a.M31 * b.x + a.M32 * b.y + a.M33 * b.z + a.M34 * b.w;
	c.w = a.M41 * b.x + a.M42 * b.y + a.M43 * b.z + a.M44 * b.w;

	return c;
}

__host__ __device__ float3 operator*(const Mat4& a, const float3& b)
{
	float4 bExt = make_float4(b.x, b.y, b.z, 1.f);
	float4 cExt = a * bExt;
	return make_float3(cExt);
}

