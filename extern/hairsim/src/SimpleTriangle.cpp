#include "SimpleTriangle.h"

std::ostream& operator<<(std::ostream& os, const SimpleTriangle& t)
{
	os << t.Idx;
	return os;
}

bool SimpleTriangle::Intersect(const float3& rayStart, const float3& rayDir, const float3& v0, const float3& v1, const float3& v2, float& t)
{
	// Intersection Algorithm
	float eps = 0.000000001f;
	float3 e1 = v1 - v0;
	float3 e2 = v2 - v0;

	float3 p = cross(rayDir, e2);
	float a = dot(e1, p);

	if (a > -eps && a < eps)
	{
		return false;
	}

	float f = 1.0f / a;
	float3 s = rayStart - v0;
	float u = f * dot(s, p);

	if (u < 0.0f || u > 1.0f)
		return false;

	float3 q = cross(s, e1);
	float v = f * dot(rayDir, q);

	if (v < 0.0f || u + v > 1.0f)
		return false;

	t = f * dot(e2, q);

	return true;
}
