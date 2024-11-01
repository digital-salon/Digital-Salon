#pragma once
#include "HelperMath.cuh"
#include <iostream>
#include <memory>
#include <limits>
#include <vector>

#define MAX_BONE_NR 4

struct SimpleVertex
{
	float3 Pos = make_float3(0.f);
	float3 Pos0 = make_float3(0.f);
	float3 Normal = make_float3(0.f);
	float3 Normal0 = make_float3(0.f);
	float2 Tex = make_float2(0.f);

	float3 PosPrevAnim = make_float3(0.f);
	float3 Vel = make_float3(0.f);

	int BoneIDs[MAX_BONE_NR];
	float BoneWeights[MAX_BONE_NR];
};

struct MiniVertex
{
	float3 Pos = make_float3(0.f);
	float3 Normal = make_float3(0.f);
};

class SimpleTriangle
{
public:
	SimpleTriangle() {}
	uint V[3] = { 0,0,0 };
	float3 Center = make_float3(0.f);
	float3 Normal = make_float3(0.f);
	int Idx = 0;
	int Selected = 0;

	friend std::ostream& operator<<(std::ostream& os, const SimpleTriangle& t);
	static bool Intersect(const float3& rayStart, const float3& rayDir, const float3& v0, const float3& v1, const float3& v2, float& t);
};