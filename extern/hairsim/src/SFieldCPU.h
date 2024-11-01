#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <cmath>

#include "SimpleLoader.h"

using namespace std;

// Elements that form a triangle
enum TriElement { V0, V1, V2, E01, E12, E02, F };

// Container for SDF query
struct PointSDF
{
	float Distance = numeric_limits<float>::max();
	float3 NearestPoint;
	TriElement NearestElement;
	int TriIdx = -1;

	void Print()
	{
		printf("Distance %f, NP (%f,%f,%f), Element %i, Tri %i\n",
			Distance, NearestPoint.x, NearestPoint.y, NearestPoint.z,
			NearestElement, TriIdx);
	}
};

// Useful containers for bounding sphere optimization
struct BoundingSphere
{
	float3 Center;
	float Radius = 0.f;
};

struct BoundingNode
{
	BoundingSphere SphereLeft, SphereRight;
	int Left, Right = -1;
};

struct MiniTriangle
{
	float3 V[3];
	float3 Center;
	int Idx = -1;
};


// Main class handling static SDF computation
class SFieldCPU
{
public:

	// Constructors
	SFieldCPU() { NumVertices = 0, NumTriangles = 0; }
	SFieldCPU(shared_ptr<SimpleObject> mesh);

	// Update
	void UpdateVertices(SimpleVertex* vertices);

	// Getters/Methods
	void BuildSDF(int3 GridSize, float3 Origin, float3 Ds);
	vector<float>& GetSDF() { return ArraySDF; }
	PointSDF DistancePointField(const float3& u);
	PointSDF SignedDistancePointField(const float3& u);

private:

	// Static SDF
	vector<float> ArraySDF;
	void QuerySDF(PointSDF& res, const BoundingNode& node, const float3& u);
	void QuerySDFNoTree(PointSDF& res, const float3& u);

	// Geometric Info
	vector<shared_ptr<SimpleTriangle>> Triangles;
	vector<shared_ptr<SimpleVertex>> Vertices;
	vector<shared_ptr<int>> Indices;
	vector<float3> TriPseudoNorm;
	vector<float3[3]> EdgPseudoNorm;
	vector<float3> VertPseudoNorm;
	int NumVertices, NumTriangles;

	// Helper structures
	unordered_map<int, float3> EdgeNormals; // Quickly access normal of edge between two triangles
	unordered_map<int, int> EdgeTriCounter; // # of triangles that share each edge

	// Bounding Volumes
	BoundingSphere RootBV;
	vector<BoundingNode> BoundingNodes;

	// Helpers
	float DistancePointTriangle(TriElement& element, float3& nearPoint, const float3& u, const SimpleTriangle& tri);
	float3 GetEdgeNormal(const int& i, const int& j);
	void AddEdgeNormal(const int& i, const int& j, const float3& normal);
	int VertToKey(const int& i, const int& j);
	void BuildExtraData();
	void BuildTreeBV(int nodeIdx, BoundingSphere& sphere, vector<MiniTriangle>& triangles, int begin, int end);
	bool Built = false;
};