#include "SFieldCPU.h"

SFieldCPU::SFieldCPU(shared_ptr<SimpleObject> mesh)
{
	// Gets mesh data
	Triangles = mesh->GetTriangles();
	Vertices = mesh->GetVertices();
	Indices = mesh->GetIndices();

	NumTriangles = Triangles.size();
	NumVertices = mesh->GetVertices().size();

	// Builds additional data (pseudonormals and BV tree)
	BuildExtraData();
}

void SFieldCPU::UpdateVertices(SimpleVertex* vertices)
{

	// Cleans everything
	BoundingNodes.clear();
	TriPseudoNorm.clear();
	EdgPseudoNorm.clear();
	VertPseudoNorm.clear();
	ArraySDF.clear();

	// Helper structures
	EdgeNormals.clear(); 
	EdgeTriCounter.clear();
	Built = false;

	// Update vert positions and normals
    #pragma omp parallel for
	for (int i = 0; i < Vertices.size(); i++)
	{
		Vertices[i] = make_shared<SimpleVertex>(vertices[i]);
	}

	// Recompute normals
	#pragma omp parallel for
	for (int i = 0; i < Triangles.size(); i++)
	{
		const float3& v0 = Vertices[Triangles[i]->V[0]]->Pos;
		const float3& v1 = Vertices[Triangles[i]->V[1]]->Pos;
		const float3& v2 = Vertices[Triangles[i]->V[2]]->Pos;

		Triangles[i]->Normal = normalize(cross(v1 - v0, v2 - v0));
	}

	BuildExtraData();

}

void SFieldCPU::BuildSDF(int3 GridSize, float3 Origin, float3 Ds)
{
	// Initialize array for SDF values
	int totalSize = GridSize.x * GridSize.y * GridSize.z;
	ArraySDF = vector<float>(totalSize);

	// Fills in parallel-cpu
#pragma omp parallel for
	for (int i = 0; i < totalSize; i++)
	{
		// Idx to 3D idx
		int3 idx3D;
		idx3D.x = i % GridSize.x;
		idx3D.y = (i / GridSize.x) % GridSize.y;
		idx3D.z = i / (GridSize.x * GridSize.y);

		// 3D Idx to position (stored at box vertex)
		float3 pos = Origin;
		pos += Ds.x * (idx3D.x) * make_float3(1.f, 0.f, 0.f);
		pos += Ds.y * (idx3D.y) * make_float3(0.f, 1.f, 0.f);
		pos += Ds.z * (idx3D.z) * make_float3(0.f, 0.f, 1.f);

		// Gets SDF
		PointSDF res = SignedDistancePointField(pos);
		ArraySDF[i] = res.Distance;
	}
}

PointSDF SFieldCPU::DistancePointField(const float3& u)
{
	// Prepares container
	PointSDF res;
	res.Distance = numeric_limits<float>::max();

	// Fills using BV tree
	QuerySDF(res, BoundingNodes[0], u);

	return res;
}

PointSDF SFieldCPU::SignedDistancePointField(const float3& u)
{
	// First, gets unsigned distance
	PointSDF res = DistancePointField(u);

	// Uses pseudonormals to get the sign
	float3 pseudoNormal = make_float3(0.f);
	switch (res.NearestElement)
	{
	case TriElement::V0:
		pseudoNormal = VertPseudoNorm[*Indices[3 * res.TriIdx + 0]];
		break;
	case TriElement::V1:
		pseudoNormal = VertPseudoNorm[*Indices[3 * res.TriIdx + 1]];
		break;
	case TriElement::V2:
		pseudoNormal = VertPseudoNorm[*Indices[3 * res.TriIdx + 2]];
		break;
	case TriElement::E01:
		pseudoNormal = EdgPseudoNorm[res.TriIdx][0];
		break;
	case TriElement::E12:
		pseudoNormal = EdgPseudoNorm[res.TriIdx][1];
		break;
	case TriElement::E02:
		pseudoNormal = EdgPseudoNorm[res.TriIdx][2];
		break;
	case TriElement::F:
		pseudoNormal = TriPseudoNorm[res.TriIdx];
		break;

	default:
		printf("\nWarning: something off at SDF computation\n");
		break;
	}

	float3 dir = u - res.NearestPoint;
	res.Distance *= dot(dir, pseudoNormal) >= 0.0 ? 1.0 : -1.0;

	return res;
}

void SFieldCPU::QuerySDF(PointSDF& res, const BoundingNode& node, const float3& u)
{
	// Reached a leaf
	if (node.Left == -1)
	{
		float3 nearPoint;
		TriElement nearElement;
		float distance = DistancePointTriangle(nearElement, nearPoint, u, *Triangles[node.Right]);

		if (distance < res.Distance)
		{
			res.NearestPoint = nearPoint;
			res.NearestElement = nearElement;
			res.Distance = distance;
			res.TriIdx = node.Right;
		}
	}

	// Else, recursion over tree
	else
	{
		// Find closer child
		float distLeft = length(u - node.SphereLeft.Center) - node.SphereLeft.Radius;
		float distRight = length(u - node.SphereRight.Center) - node.SphereRight.Radius;

		if (distLeft < distRight)
		{
			// Both to avoid overlap buggs
			if (distLeft < res.Distance) QuerySDF(res, BoundingNodes[node.Left], u);
			if (distRight < res.Distance) QuerySDF(res, BoundingNodes[node.Right], u);
		}
		else
		{
			// Both to avoid overlap buggs
			if (distRight < res.Distance) QuerySDF(res, BoundingNodes[node.Right], u);
			if (distLeft < res.Distance) QuerySDF(res, BoundingNodes[node.Left], u);
		}
	}
}

void SFieldCPU::QuerySDFNoTree(PointSDF& res, const float3& u)
{
	// Used for debugging purposes, brute-force shortest distance computation
	for (int i = 0; i < NumTriangles; i++)
	{
		float3 nearPoint;
		TriElement nearElement;
		float distance = DistancePointTriangle(nearElement, nearPoint, u, *Triangles[i]);

		if (distance < res.Distance)
		{
			res.NearestPoint = nearPoint;
			res.NearestElement = nearElement;
			res.Distance = distance;
			res.TriIdx = i;
		}
	}
}

float SFieldCPU::DistancePointTriangle(TriElement& element, float3& nearPoint, const float3& u, const SimpleTriangle& tri)
{

	// Intersection algorithm from Andreas and Henrik's paper
	// It divides triangle into different regions
	const float3& v0 = Vertices[tri.V[0]]->Pos;
	const float3& v1 = Vertices[tri.V[1]]->Pos;
	const float3& v2 = Vertices[tri.V[2]]->Pos;

	float3 diff = v0 - u;
	float3 edge0 = v1 - v0;
	float3 edge1 = v2 - v0;
	float a00 = dot(edge0, edge0);
	float a01 = dot(edge0, edge1);
	float a11 = dot(edge1, edge1);
	float b0 = dot(diff, edge0);
	float b1 = dot(diff, edge1);
	float c = dot(diff, diff);
	float det = abs(a00 * a11 - a01 * a01);
	float s = a01 * b1 - a11 * b0;
	float t = a01 * b0 - a00 * b1;

	float d2 = -1.0;

	if (s + t <= det)
	{
		if (s < 0)
		{
			if (t < 0)  // region 4
			{
				if (b0 < 0)
				{
					t = 0;
					if (-b0 >= a00)
					{
						element = TriElement::V1;
						s = 1;
						d2 = a00 + (2) * b0 + c;
					}
					else
					{
						element = TriElement::E01;
						s = -b0 / a00;
						d2 = b0 * s + c;
					}
				}
				else
				{
					s = 0;
					if (b1 >= 0)
					{
						element = TriElement::V0;
						t = 0;
						d2 = c;
					}
					else if (-b1 >= a11)
					{
						element = TriElement::V2;
						t = 1;
						d2 = a11 + (2) * b1 + c;
					}
					else
					{
						element = TriElement::E02;
						t = -b1 / a11;
						d2 = b1 * t + c;
					}
				}
			}
			else  // region 3
			{
				s = 0;
				if (b1 >= 0)
				{
					element = TriElement::V0;
					t = 0;
					d2 = c;
				}
				else if (-b1 >= a11)
				{
					element = TriElement::V2;
					t = 1;
					d2 = a11 + (2) * b1 + c;
				}
				else
				{
					element = TriElement::E02;
					t = -b1 / a11;
					d2 = b1 * t + c;
				}
			}
		}
		else if (t < 0)  // region 5
		{
			t = 0;
			if (b0 >= 0)
			{
				element = TriElement::V0;
				s = 0;
				d2 = c;
			}
			else if (-b0 >= a00)
			{
				element = TriElement::V1;
				s = 1;
				d2 = a00 + (2) * b0 + c;
			}
			else
			{
				element = TriElement::E01;
				s = -b0 / a00;
				d2 = b0 * s + c;
			}
		}
		else  // region 0 
		{
			element = TriElement::F;
			// minimum at interior point
			float invDet = (1) / det;
			s *= invDet;
			t *= invDet;
			d2 = s * (a00 * s + a01 * t + (2) * b0) +
				t * (a01 * s + a11 * t + (2) * b1) + c;
		}
	}
	else
	{
		float tmp0, tmp1, numer, denom;

		if (s < 0)  // region 2
		{
			tmp0 = a01 + b0;
			tmp1 = a11 + b1;
			if (tmp1 > tmp0)
			{
				numer = tmp1 - tmp0;
				denom = a00 - (2) * a01 + a11;
				if (numer >= denom)
				{
					element = TriElement::V1;
					s = 1;
					t = 0;
					d2 = a00 + (2) * b0 + c;
				}
				else
				{
					element = TriElement::E12;
					s = numer / denom;
					t = 1 - s;
					d2 = s * (a00 * s + a01 * t + (2) * b0) +
						t * (a01 * s + a11 * t + (2) * b1) + c;
				}
			}
			else
			{
				s = 0;
				if (tmp1 <= 0)
				{
					element = TriElement::V2;
					t = 1;
					d2 = a11 + (2) * b1 + c;
				}
				else if (b1 >= 0)
				{
					element = TriElement::V0;
					t = 0;
					d2 = c;
				}
				else
				{
					element = TriElement::E02;
					t = -b1 / a11;
					d2 = b1 * t + c;
				}
			}
		}
		else if (t < 0)  // region 6
		{
			tmp0 = a01 + b1;
			tmp1 = a00 + b0;
			if (tmp1 > tmp0)
			{
				numer = tmp1 - tmp0;
				denom = a00 - (2) * a01 + a11;
				if (numer >= denom)
				{
					element = TriElement::V2;
					t = 1;
					s = 0;
					d2 = a11 + (2) * b1 + c;
				}
				else
				{
					element = TriElement::E12;
					t = numer / denom;
					s = 1 - t;
					d2 = s * (a00 * s + a01 * t + (2) * b0) +
						t * (a01 * s + a11 * t + (2) * b1) + c;
				}
			}
			else
			{
				t = 0;
				if (tmp1 <= 0)
				{
					element = TriElement::V1;
					s = 1;
					d2 = a00 + (2) * b0 + c;
				}
				else if (b0 >= 0)
				{
					element = TriElement::V0;
					s = 0;
					d2 = c;
				}
				else
				{
					element = TriElement::E01;
					s = -b0 / a00;
					d2 = b0 * s + c;
				}
			}
		}
		else  // region 1
		{
			numer = a11 + b1 - a01 - b0;
			if (numer <= 0)
			{
				element = TriElement::V2;
				s = 0;
				t = 1;
				d2 = a11 + (2) * b1 + c;
			}
			else
			{
				denom = a00 - (2) * a01 + a11;
				if (numer >= denom)
				{
					element = TriElement::V1;
					s = 1;
					t = 0;
					d2 = a00 + (2) * b0 + c;
				}
				else
				{
					element = TriElement::E12;
					s = numer / denom;
					t = 1 - s;
					d2 = s * (a00 * s + a01 * t + (2) * b0) +
						t * (a01 * s + a11 * t + (2) * b1) + c;
				}
			}
		}
	}

	// To avoid numerical error.
	if (d2 < 0)
	{
		d2 = 0;
	}

	nearPoint = v0 + s * edge0 + t * edge1;
	return sqrt(d2);
}

float3 SFieldCPU::GetEdgeNormal(const int& i, const int& j)
{
	return EdgeNormals.find(VertToKey(i, j))->second;
}

void SFieldCPU::AddEdgeNormal(const int& i, const int& j, const float3& normal)
{
	int key = VertToKey(i, j);

	// First time edge is computed
	if (EdgeNormals.find(key) == EdgeNormals.end())
	{
		EdgeNormals[key] = normal;
		EdgeTriCounter[key] = 1;
	}
	// Otherwise, adds normal and counter
	else
	{
		EdgeNormals[key] += normal;
		EdgeTriCounter[key] += 1;
	}
}

int SFieldCPU::VertToKey(const int& i, const int& j)
{
	return min(i, j) * NumVertices + max(i, j);
}

void SFieldCPU::BuildExtraData()
{
	// Build tree containing the triangles
	vector<MiniTriangle> minTriangles(NumTriangles);

#pragma omp parallel for
	for (int i = 0; i < NumTriangles; i++)
	{
		minTriangles[i].Idx = Triangles[i]->Idx;
		minTriangles[i].Center = Triangles[i]->Center;
		minTriangles[i].V[0] = Vertices[Triangles[i]->V[0]]->Pos;
		minTriangles[i].V[1] = Vertices[Triangles[i]->V[1]]->Pos;
		minTriangles[i].V[2] = Vertices[Triangles[i]->V[2]]->Pos;
	}

	BoundingNodes.push_back(BoundingNode());
	BuildTreeBV(0, RootBV, minTriangles, 0, NumTriangles);

	// Compute pseudonormals
	// First, generate containers
	TriPseudoNorm = vector<float3>(NumTriangles);
	EdgPseudoNorm = vector<float3[3]>(NumTriangles);
	VertPseudoNorm = vector<float3>(NumVertices, make_float3(0.f));

	// Iterate over all geometry
	for (int i = 0; i < NumTriangles; i++)
	{

		// Data
		const float3& a = Vertices[Triangles[i]->V[0]]->Pos;
		const float3& b = Vertices[Triangles[i]->V[1]]->Pos;
		const float3& c = Vertices[Triangles[i]->V[2]]->Pos;
		const float3& tNormal = Triangles[i]->Normal;

		// Pseudonorm triangle
		TriPseudoNorm[i] = tNormal;

		// Pseudonorm vertex
		float alpha_0 = acos(dot(normalize(b - a), normalize(c - a)));
		float alpha_1 = acos(dot(normalize(a - b), normalize(c - b)));
		float alpha_2 = acos(dot(normalize(b - c), normalize(a - c)));

		VertPseudoNorm[*Indices[3 * i + 0]] += alpha_0 * tNormal;
		VertPseudoNorm[*Indices[3 * i + 1]] += alpha_1 * tNormal;
		VertPseudoNorm[*Indices[3 * i + 2]] += alpha_2 * tNormal;

		// Pseudonorm edges
		AddEdgeNormal(*Indices[3 * i + 0], *Indices[3 * i + 1], tNormal);
		AddEdgeNormal(*Indices[3 * i + 1], *Indices[3 * i + 2], tNormal);
		AddEdgeNormal(*Indices[3 * i + 2], *Indices[3 * i + 0], tNormal);
	}

	// Normalize vectors
	for (float3& n : VertPseudoNorm)
	{
		n = normalize(n);
	}

	for (int i = 0; i < NumTriangles; i++)
	{
		EdgPseudoNorm[i][0] = normalize(GetEdgeNormal(*Indices[3 * i + 0], *Indices[3 * i + 1]));
		EdgPseudoNorm[i][1] = normalize(GetEdgeNormal(*Indices[3 * i + 1], *Indices[3 * i + 2]));
		EdgPseudoNorm[i][2] = normalize(GetEdgeNormal(*Indices[3 * i + 2], *Indices[3 * i + 0]));
	}

	// DEBUG: Watertight mesh.
	bool SingleEdge = false;
	bool TripleEdge = false;
	for (const auto counter : EdgeTriCounter)
	{
		if (counter.second == 1) SingleEdge = true;
		if (counter.second > 2) TripleEdge = true;
	}
	//if (SingleEdge) printf("\n There is a single edge case, this may cause leaking!\n.");
	//if (TripleEdge) printf("\n There is a triple edge case, this may cause leaking!\n.");

	// Flag pseudonormal info has been built
	Built = true;
}

void SFieldCPU::BuildTreeBV(int nodeIdx, BoundingSphere& sphere, vector<MiniTriangle>& triangles, int begin, int end)
{
	int numTriangles = end - begin;

	// We are at a leaf
	if (numTriangles == 1)
	{
		// Build node leaf
		BoundingNodes[nodeIdx].Left = -1;
		BoundingNodes[nodeIdx].Right = triangles[begin].Idx;

		// Bounding sphere
		MiniTriangle& t = triangles[begin];
		float3 center = (t.V[0] + t.V[1] + t.V[2]) / 3.f;
		float radius = max(max(length(t.V[0] - center), length(t.V[1] - center)), length(t.V[2] - center));
		sphere.Center = center;
		sphere.Radius = radius;
	}
	// Otherwise, generate sub-tree
	else
	{
		// Compute center of AABB, and largest sub-mesh length dimension
		float3 top = make_float3(numeric_limits<float>::min());
		float3 bottom = make_float3(numeric_limits<float>::max());
		float3 center = make_float3(0.f);

		// Iterate over triangles
		for (int i = begin; i < end; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				float3& pos = triangles[i].V[j];
				center += pos;

				// Greater bounds
				top.x = max(top.x, pos.x);
				top.y = max(top.y, pos.y);
				top.z = max(top.z, pos.z);

				// Lower bounds
				bottom.x = min(bottom.x, pos.x);
				bottom.y = min(bottom.y, pos.y);
				bottom.z = min(bottom.z, pos.z);
			}
		}

		center /= (3.f * numTriangles);
		float3 distances = top - bottom;
		int split_axis;
		if (distances.x >= distances.y && distances.x >= distances.z) split_axis = 0;
		else if (distances.y >= distances.x && distances.y >= distances.z) split_axis = 1;
		else split_axis = 2;

		// Sphere bounding sub-mesh
		float radius = 0.f;
		for (int i = begin; i < end; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				radius = max(radius, length(center - triangles[i].V[j]));
			}
		}

		sphere.Center = center;
		sphere.Radius = radius;

		// Lambda to sort triangles along the chosen axis
		sort(triangles.begin() + begin, triangles.begin() + end,
			[split_axis](const MiniTriangle& a, const MiniTriangle& b)
			{
				if (split_axis == 0) return a.Center.x < b.Center.x;
				else if (split_axis == 1) return a.Center.y < b.Center.y;
				else return a.Center.z < b.Center.z;
			}
		);

		// Build children
		int midIdx = int(0.5 * (begin + end));

		BoundingNodes[nodeIdx].Left = BoundingNodes.size();
		BoundingNodes.push_back(BoundingNode());
		BuildTreeBV(BoundingNodes[nodeIdx].Left, BoundingNodes[nodeIdx].SphereLeft, triangles, begin, midIdx);

		BoundingNodes[nodeIdx].Right = BoundingNodes.size();
		BoundingNodes.push_back(BoundingNode());
		BuildTreeBV(BoundingNodes[nodeIdx].Right, BoundingNodes[nodeIdx].SphereRight, triangles, midIdx, end);
	}
}