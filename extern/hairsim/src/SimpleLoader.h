#pragma once
#include <SimpleTriangle.h>
#include <Bone.h>
#include <string>       
#include <iostream>     
#include <sstream>   

#define ASSIMP_LOAD_FLAGS (aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices | aiProcess_PopulateArmatureData )
using namespace std;


class SimpleObject
{
public:
	SimpleObject() {};
	SimpleObject(const string& fname, const float3& pos = make_float3(0.f), const float3& scale = make_float3(1.f), const bool& anim = false, const int& numFrames = 1);
	SimpleObject(const string& fname, const float4& pos, const bool& anim = false, const int& numFrames = 1);
	~SimpleObject();

	// Animation
	void UpdateFrameSeq(const int& frame);
	void ExportObj(const string& fname);
	void MoveBone(const int& boneID);
	void UpdateVertices();
	void UpdateVerticesFromSeq(SimpleVertex* vert);
	void UpdateBoneTrans();

	// Getters/Setters
	Mat4& GetMasterTrans() { return MasterTrans; }
	Mat4& GetMasterTransInv() { return MasterTransInv; }
	vector<Mat4>& GetFinalTrans() { return FinalTrans; }
	vector<shared_ptr<SimpleTriangle>>& GetTriangles() { return TrianglesObj; }
	vector<shared_ptr<int>>& GetIndices() { return Indices; }
	vector<shared_ptr<SimpleVertex>>& GetVertices() { return Vertices; }
	vector<SimpleTriangle> GetTrianglesRaw();
	vector<SimpleVertex> GetVerticesRaw();
	vector<MiniVertex> GetAnimVerticesRaw();
	int GetNumTriangles() { return NumTriangles; }
	int GetNumVertices() { return NumVertices; }
	int GetNumFrames() { return NumFrames; }
	int GetNumBones() { return NumBones; }
	vector<Bone>& GetBones() { return Bones; }
	vector<float3>& GetAABB() { return AABB; }
	vector<vector<float3>>& GetAnimAABB() { return AnimAABB; }
	vector<float3>& GetAABB0() { return AABB0; }

private:

	// Animation
	vector<vector<shared_ptr<MiniVertex>>> AnimVertex;
	vector<vector<float3>> AnimAABB;
	vector<bool> InsideGrid;
	bool Animated = false;
	int NumFrames = 1;

	// Mesh Geometry
	vector<shared_ptr<SimpleTriangle>> TrianglesObj;
	vector<shared_ptr<SimpleVertex>> Vertices;
	int NumVertices, NumTriangles, NumIndices;
	vector<shared_ptr<int>> Indices;
	vector<float3> AABB, AABB0;

	// Bones
	vector<Mat4> FinalTrans;
	vector<Bone> Bones;
	Armature Skeleton;
	Mat4 MasterTrans = Mat4::Identity();
	Mat4 MasterTransInv = Mat4::Identity();
	int NumBones = 0;

	// Animation Helpers
	void ComputeAABBSequence();
	void LoadMeshSequence(const string& fname, const float3& pos, const float3& scale);
	void FillGridInfo(const string& fname);
	void ComputeFinalTrans();
	void ComputeBoneOffset(const Armature& node, Mat4 parentTrans);
	void ComputeBoneGlobal(const Armature& node, Mat4 parentTrans);

	// General Helpers
	void ComputeAABB();

	// Bone Helpers
	void ReadArmature(Armature& dest, const aiNode* src);
	int FindBoneName(const std::string& name);
	void GenerateDummyBone();

	// Assimp Helpers
	void LoaderAssimp(const string& fname, const float3& pos, const float3& scale);
	void SetVertexBoneData(SimpleVertex& v, int boneID, float weight);
	void ExtractBoneWeight(aiMesh* mesh, const aiScene* scene);
	Mat4 AssimpToMat4(const aiMatrix4x4& mat);
	void ProcessNode(aiNode* node, const aiScene* scene);
	void ProcessMesh(aiMesh* mesh, const aiScene* scene);
	void ProcessNodeSequence(aiNode* node, const aiScene* scene, const int& frame);
	void ProcessMeshSequence(aiMesh* mesh, const aiScene* scene, const int& frame);
	void InitBoneData(SimpleVertex& v);
	void BuildTriangles();
};