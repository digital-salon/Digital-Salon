#pragma once

#include <string>
#include <vector>
#include "SimpleTriangle.h"
#include <Rendering/Geometry.h>
#include <fstream>
#include "Rendering/VertexBufferObject.h"

struct Quad2D
{
public:
	Quad2D(float3 a, float3 b, float3 c, float3 d) : A(a), B(b), C(c), D(d) {}
	Quad2D() {}
	float3 A, B, C, D;
};

class Mesh2D
{
public:
	Mesh2D(const int2& nxy = make_int2(64, 64), const float2& origin = make_float2(-8.f), const float2& end = make_float2(8.f));
	~Mesh2D();
	void Render();
	std::vector<Quad2D>& GetSquares() { return Squares; }

private:
	std::vector<int> Indices;
	std::vector<float3> VertexPos;
	//std::vector<float3> VertexNormal;
	//std::vector<float2> VertexTex;
	std::vector<Quad2D> Squares;
	std::shared_ptr<VertexBufferObject> MeshVbo;
	std::shared_ptr<VertexBufferObject> MeshVboWF;
	void BuildMesh();
	void BuildVbo();
	//void NormalizeGeometry(const float3& pos, const float3& scale);
	//void ComputeVertexNormals();
	//void ComputeCenters();

	// Mesh Params
	int2 Nxy;
	float2 Origin;
	float2 End;
};
