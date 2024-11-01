#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>
#include <map>
#include <assert.h>

#include <SimpleTriangle.h>

using namespace std;

struct Armature
{
	Mat4 Transformation;
	string Name;
	int BoneIdx;
	vector<Armature> Children;

	void print(const int depth = 0)
	{
		for (int i = 0; i < depth; ++i)
		{
			if (i != depth - 1) std::cout << "    ";
			else std::cout << "|-- ";
		}
		std::cout << this->Name << std::endl;
		this->Transformation.Print();
		for (uint i = 0; i < this->Children.size(); ++i)
		{
			this->Children[i].print(depth + 1);
		}
	}
};

class Bone
{
public:
	Bone();
	Bone(const std::string& name, int id) : Name(name), ID(id) {}

	// Animation
	void UpdateTrans();
	float3 Angles = make_float3(0.f);
	float3 MoveVec = make_float3(0.f);
	float ScaleNum = 1.f;

	// Setters and Getters
	string GetName() { return Name; }
	int GetID() { return ID; }
	Mat4 GetLocalTrans();
	Mat4& GetLocalRot() { return Rot; }
	Mat4& GetGlobalTrans() { return GlobalTrans; }
	Mat4& GetOffsetTrans() { return OffsetTrans; }

	void SetRot(float3 r);
	void SetMove(float3 t);
	void SetScale(float3 s);
	void SetGlobalTrans(const Mat4& t) { GlobalTrans = t; }
	void SetOffsetTrans(const Mat4& t) { OffsetTrans = t; }
	void SetName(std::string name) { Name = name; }
	void SetID(int id) { ID = id; }
private:
	string Name;
	int ID;
	Mat4 Move, Rot, Scale;
	Mat4 GlobalTrans;
	Mat4 OffsetTrans;
};

