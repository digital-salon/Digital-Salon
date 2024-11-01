#include "SimpleLoader.h"

SimpleObject::SimpleObject(const string& fname, const float3& pos, const float3& scale, const bool& anim, const int& numFrames) : Animated(anim), NumFrames(numFrames)
{
	// Important! Assumes input file contains only one mesh
	LoaderAssimp(fname, pos, scale);

	// Fills information for embedding grid
	FillGridInfo(fname);

	// Load pre-computed sequence
	if (Animated)
	{
		LoadMeshSequence(fname, pos, scale);
		ComputeAABBSequence();
	}

	// Axis Aligned Bounding Box
	ComputeAABB();
}

SimpleObject::SimpleObject(const string& fname, const float4& pos, const bool& anim, const int& numFrames) : Animated(anim), NumFrames(numFrames)
{
	// Important! Assumes input file contains only one mesh
	LoaderAssimp(fname, make_float3(pos.x, pos.y, pos.z), make_float3(pos.w));

	// Fills information for embedding grid
	FillGridInfo(fname);

	// Load pre-computed sequence
	if (Animated)
	{
		LoadMeshSequence(fname, make_float3(pos.x, pos.y, pos.z), make_float3(pos.w));
		ComputeAABBSequence();
	}

	// Axis Aligned Bounding Box
	ComputeAABB();
}

SimpleObject::~SimpleObject()
{
}

void SimpleObject::UpdateFrameSeq(const int& frame)
{
	#pragma omp parallel for
	for (int i = 0; i < NumVertices; i++)
	{
		SimpleVertex& v = *Vertices[i];

		v.Pos = AnimVertex[frame][i]->Pos;
		v.Normal = AnimVertex[frame][i]->Normal;
	}
}

void SimpleObject::ExportObj(const string& fname)
{
	// Creates file
	ofstream file(fname);

	// Header
	int a = fname.find_last_of('/');
	int b = fname.find_last_of('.');
	string name = fname.substr(a + 1, b - a - 1);
	file << "o " << name << "\n";

	// Vertices
	for (int i = 0; i < NumVertices; i++)
	{
		float3& pos = Vertices[i]->Pos;
		file << "v " << pos.x << " " << pos.y << " " << pos.z << "\n";
	}

	// Indices
	for (int i = 0; i < NumTriangles - 1; i++)
	{
		SimpleTriangle& tri = *TrianglesObj[i];
		file << "f " << tri.V[0] + 1 << " " << tri.V[1] + 1 << " " << tri.V[2] + 1 << "\n";
	}
	SimpleTriangle& tri = *TrianglesObj[NumTriangles - 1];
	file << "f " << tri.V[0] + 1 << " " << tri.V[1] + 1 << " " << tri.V[2] + 1 << "\n";

	// Closes
	file.close();
}

void SimpleObject::MoveBone(const int& boneID)
{
	Bones[boneID].UpdateTrans();
}

void SimpleObject::UpdateVertices()
{
	#pragma omp parallel for
	for (int i = 0; i < NumVertices; i++)
	{
		SimpleVertex& v = *Vertices[i];
		float4 pos0 = make_float4(v.Pos0, 1.f);
		float4 posF = make_float4(0.f);
		for (int k = 0; k < MAX_BONE_NR; k++)
		{
			const int& idx = v.BoneIDs[k];
			if (idx != -1)
			{
				posF += v.BoneWeights[k] * (FinalTrans[idx] * pos0);
			}
			else break;
		}

		v.Pos = make_float3(posF);
	}
}

void SimpleObject::UpdateVerticesFromSeq(SimpleVertex* vert)
{
	#pragma omp parallel for
	for (int i = 0; i < NumVertices; i++)
	{
		Vertices[i] = make_shared<SimpleVertex>(vert[i]);
	}
}

void SimpleObject::UpdateBoneTrans()
{
	// Updates armature
	ComputeBoneGlobal(Skeleton, MasterTrans);

	// Updates final trans
	ComputeFinalTrans();
}

void SimpleObject::ComputeAABBSequence()
{
	#pragma omp parallel for
	for (int frame = 0; frame < NumFrames; frame++)
	{
		float3 minBound = make_float3(numeric_limits<float>::max());
		float3 maxBound = make_float3(numeric_limits<float>::min());

		for (int i = 0; i < TrianglesObj.size(); i++)
		{
			SimpleTriangle& t = *TrianglesObj[i];
			if (InsideGrid[i])
			{
				for (int j = 0; j < 3; j++)
				{
					float3& pos = AnimVertex[frame][t.V[j]]->Pos;

					// Lower bounds
					minBound.x = min(minBound.x, pos.x);
					minBound.y = min(minBound.y, pos.y);
					minBound.z = min(minBound.z, pos.z);

					// Greater bounds
					maxBound.x = max(maxBound.x, pos.x);
					maxBound.y = max(maxBound.y, pos.y);
					maxBound.z = max(maxBound.z, pos.z);
				}
			}
		}

		AnimAABB[frame] = vector<float3>{ minBound, maxBound };
	}
}

void SimpleObject::LoadMeshSequence(const string& fname, const float3& pos, const float3& scale)
{
	// Allocate memory for containers
	AnimVertex = vector<vector<shared_ptr<MiniVertex>>>(NumFrames);
	AnimAABB = vector<vector<float3>>(NumFrames);

	// Load meshes
	#pragma omp parallel for
	for (int i = 0; i < NumFrames; i++)
	{
		// Get this frame's file name
		string name = fname.substr(0, fname.find_last_of('_') + 1) + to_string(i + 1) + ".obj";

		// Load with Assimp
		Assimp::Importer import;
		const aiScene * scene = import.ReadFile(name.c_str(), ASSIMP_LOAD_FLAGS);

		// Checks errors
		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		{
			cout << "ERROR::ASSIMP::" << import.GetErrorString() << endl;
		}

		// Process assimp information recursively
		ProcessNodeSequence(scene->mRootNode, scene, i);
	}

	// Apply the same transform to all the sequence
	#pragma omp parallel for
	for (int frame = 0; frame < NumFrames; frame++)
	{
		#pragma omp parallel for
		for (int i = 0; i < NumVertices; i++)
		{
			SimpleVertex& v = *Vertices[i];
			MiniVertex& animV = *AnimVertex[frame][i];
			float4 pos0 = make_float4(animV.Pos, 1.f);
			float4 posF = make_float4(0.f);
			for (int k = 0; k < MAX_BONE_NR; k++)
			{
				const int& idx = v.BoneIDs[k];
				if (idx != -1)
				{
					posF += v.BoneWeights[k] * (FinalTrans[idx] * pos0);
				}
				else break;
			}

			animV.Pos = make_float3(posF);
		}
	}
}

void SimpleObject::FillGridInfo(const string& fname)
{
	// Get actual name
	string name = fname.substr(0, fname.find_last_of('/')) + "/grid.txt";

	// Load file
	std::fstream in(name);
	std::string line;

	// Init indices
	InsideGrid = vector<bool>(TrianglesObj.size(), true);

	// Read (optional) file containing selection for
	// limited grid
	while (std::getline(in, line))
	{

		int idx;
		bool opt;
		std::stringstream ss(line);

		ss >> idx;
		ss >> opt;


		InsideGrid[idx] = opt;
	}
}

void SimpleObject::ComputeFinalTrans()
{
	#pragma omp parallel for
	for (int i = 0; i < NumBones; i++)
	{
		Bone& b = Bones[i];
		FinalTrans[i] = b.GetGlobalTrans() * b.GetLocalTrans() * b.GetOffsetTrans();
	}
}

void SimpleObject::ComputeBoneOffset(const Armature& node, Mat4 parentTrans)
{
	Mat4 globalTrans = parentTrans * node.Transformation;
	if (node.BoneIdx != -1)
	{
		Bones[node.BoneIdx].SetOffsetTrans(globalTrans.Inverse());
	}

	for (int i = 0; i < node.Children.size(); i++)
	{
		ComputeBoneOffset(node.Children[i], globalTrans);
	}
}

void SimpleObject::ComputeBoneGlobal(const Armature& node, Mat4 parentTrans)
{
	Mat4 globalTrans = parentTrans * node.Transformation;
	if (node.BoneIdx != -1)
	{

		Bones[node.BoneIdx].SetGlobalTrans(globalTrans);
		globalTrans = globalTrans * Bones[node.BoneIdx].GetLocalTrans();
	}

	for (int i = 0; i < node.Children.size(); i++)
	{
		ComputeBoneGlobal(node.Children[i], globalTrans);
	}
}

void SimpleObject::LoaderAssimp(const string& fname, const float3& pos, const float3& scale)
{
	// Load with Assimp
	Assimp::Importer import;
	const aiScene * scene = import.ReadFile(fname.c_str(), ASSIMP_LOAD_FLAGS);

	// Checks errors
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		cout << "ERROR::ASSIMP::" << import.GetErrorString() << endl;
		return;
	}

	// Process assimp information recursively
	ProcessNode(scene->mRootNode, scene);

	// Fills armature tree
	ReadArmature(Skeleton, scene->mRootNode);

	// Builds master matrix
	MasterTrans = Mat4::Translate(pos) * Mat4::Scale(scale);
	MasterTransInv = MasterTrans.Inverse();

	// Assimp transformations:
	// global -> from bone to mesh space
	// offset -> from mesh space to bone
	
	// Bone offset
	ComputeBoneOffset(Skeleton, Mat4::Identity());

	// Additional bone matrices
	ComputeBoneGlobal(Skeleton, MasterTrans);
	ComputeFinalTrans();

	// Makes sure we have at least one bone
	if (NumBones == 0) GenerateDummyBone();

	// First vertex update
	UpdateVertices();

	// Builds geometry
	BuildTriangles();

}

void SimpleObject::ComputeAABB()
{

	float3 minBound = make_float3(numeric_limits<float>::max());
	float3 maxBound = make_float3(numeric_limits<float>::min());

	float3 minBound0 = make_float3(numeric_limits<float>::max());
	float3 maxBound0 = make_float3(numeric_limits<float>::min());

	for (int i = 0; i < TrianglesObj.size(); i++)
	{
		SimpleTriangle& t = *TrianglesObj[i];
		if (InsideGrid[i])
		{
			for (int j = 0; j < 3; j++)
			{
				float3& pos0 = Vertices[t.V[j]]->Pos0;
				float3& pos = Vertices[t.V[j]]->Pos;

				// Lower bounds
				minBound.x = min(minBound.x, pos.x);
				minBound.y = min(minBound.y, pos.y);
				minBound.z = min(minBound.z, pos.z);

				// Greater bounds
				maxBound.x = max(maxBound.x, pos.x);
				maxBound.y = max(maxBound.y, pos.y);
				maxBound.z = max(maxBound.z, pos.z);

				// Initial BB0 (before bone trans)
				// Lower bounds
				minBound0.x = min(minBound0.x, pos0.x);
				minBound0.y = min(minBound0.y, pos0.y);
				minBound0.z = min(minBound0.z, pos0.z);

				// Greater bounds
				maxBound0.x = max(maxBound0.x, pos0.x);
				maxBound0.y = max(maxBound0.y, pos0.y);
				maxBound0.z = max(maxBound0.z, pos0.z);
			}
		}
	}

	AABB0 = vector<float3>{ minBound0, maxBound0 };
	AABB = vector<float3>{ minBound, maxBound };
}

void SimpleObject::ReadArmature(Armature& dest, const aiNode* src)
{
	assert(src);

	// Finds index of bone's name
	dest.BoneIdx = FindBoneName(src->mName.data);
	dest.Name = src->mName.data;
	dest.Transformation = AssimpToMat4(src->mTransformation);
	for (int i = 0; i < src->mNumChildren; i++)
	{
		Armature newBone;
		ReadArmature(newBone, src->mChildren[i]);
		dest.Children.push_back(newBone);
	}
}

int SimpleObject::FindBoneName(const string& name)
{
	int idx = -1;
	for (int i = 0; i < NumBones; i++)
	{
		if (Bones[i].GetName() == name)
		{
			idx = i;
			break;
		}
	}
	return idx;
}

void SimpleObject::GenerateDummyBone()
{
	// Generates Dummy Data
	NumBones = 1;
	Bones = vector<Bone>(NumBones);
	FinalTrans = vector<Mat4>(NumBones, MasterTrans);

	// Identity Bones
	Bone dummyBone;
	dummyBone.SetID(0);
	dummyBone.SetName("dummy");
	dummyBone.SetGlobalTrans(Mat4::Identity());
	dummyBone.SetOffsetTrans(Mat4::Identity());
	Bones[0] = dummyBone;

	// Dummy Weight
	for (int i = 0; i < Vertices.size(); i++)
	{
		SimpleVertex& v = *Vertices[i];
		v.BoneIDs[0] = 0;
		v.BoneWeights[0] = 1.f;
	}

}

void SimpleObject::SetVertexBoneData(SimpleVertex& v, int boneID, float weight)
{
	for (int i = 0; i < MAX_BONE_NR; i++)
	{
		if (v.BoneIDs[i] < 0)
		{
			v.BoneWeights[i] = weight;
			v.BoneIDs[i] = boneID;
			break;
		}
	}
}

void SimpleObject::ExtractBoneWeight(aiMesh* mesh, const aiScene* scene)
{
	// Allocate Memory
	NumBones = mesh->mNumBones;
	Bones = vector<Bone>(NumBones);
	FinalTrans = vector<Mat4>(NumBones, Mat4::Identity());
	int boneCtr = 0;

	// Set Bone-Vertex Relations
	for (int boneIdx = 0; boneIdx < mesh->mNumBones; boneIdx++)
	{
		int boneID = -1;
		string boneName = mesh->mBones[boneIdx]->mName.C_Str();
		int nameIdx = FindBoneName(boneName);
		if (nameIdx == -1)
		{
			Bone newBone;
			newBone.SetID(boneCtr);
			newBone.SetName(boneName);
			Bones[boneCtr] = newBone;
			boneID = boneCtr;
			boneCtr++;
		}
		else
		{
			boneID = Bones[nameIdx].GetID();
		}

		assert(boneID != -1);
		auto weights = mesh->mBones[boneIdx]->mWeights;
		int numWeights = mesh->mBones[boneIdx]->mNumWeights;

		for (int weightIdx = 0; weightIdx < numWeights; weightIdx++)
		{
			int vertexID = weights[weightIdx].mVertexId;
			float weight = weights[weightIdx].mWeight;
			assert(vertexID <= Vertices.size());
			SetVertexBoneData(*Vertices[vertexID], boneID, weight);
		}
	}
}

Mat4 SimpleObject::AssimpToMat4(const aiMatrix4x4& mat)
{
	return Mat4(mat.a1, mat.a2, mat.a3, mat.a4,
				mat.b1, mat.b2, mat.b3, mat.b4,
				mat.c1, mat.c2, mat.c3, mat.c4,
				mat.d1, mat.d2, mat.d3, mat.d4);
}

void SimpleObject::ProcessNode(aiNode* node, const aiScene* scene)
{
	// Process all the node's meshes
	for (unsigned int i = 0; i < node->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		ProcessMesh(mesh, scene);
	}

	// then do the same for each of its children
	for (unsigned int i = 0; i < node->mNumChildren; i++)
	{
		ProcessNode(node->mChildren[i], scene);
	}
}

void SimpleObject::ProcessMesh(aiMesh* mesh, const aiScene* scene)
{
	// Process Vertices
	NumVertices = mesh->mNumVertices;
	Vertices = vector<shared_ptr<SimpleVertex>>(NumVertices);

	#pragma omp parallel for
	for (int i = 0; i < NumVertices; i++)
	{
		SimpleVertex v;

		v.Pos.x = mesh->mVertices[i].x;
		v.Pos.y = mesh->mVertices[i].y;
		v.Pos.z = mesh->mVertices[i].z;

		v.Pos0 = v.Pos;

		v.Normal.x = mesh->mNormals[i].x;
		v.Normal.y = mesh->mNormals[i].y;
		v.Normal.z = mesh->mNormals[i].z;

		v.Normal0 = v.Normal;

		if (mesh->mTextureCoords[0])
		{
			v.Tex.x = mesh->mTextureCoords[0][i].x;
			v.Tex.y = mesh->mTextureCoords[0][i].y;
		}

		InitBoneData(v);

		Vertices[i] = make_shared<SimpleVertex>(v);
	}

	// Process Indices
	NumIndices = 0;
	for (int i = 0; i < mesh->mNumFaces; i++)
	{
		NumIndices += mesh->mFaces[i].mNumIndices;
	}
	Indices = vector<shared_ptr<int>>(NumIndices);

	int idx = 0;
	for (int i = 0; i < mesh->mNumFaces; i++)
	{
		aiFace face = mesh->mFaces[i];
		for (int j = 0; j < face.mNumIndices; j++)
		{
			Indices[idx] = make_shared<int>(face.mIndices[j]);
			idx++;
		}
	}

	// Process Bones
	ExtractBoneWeight(mesh, scene);
}

void SimpleObject::ProcessNodeSequence(aiNode* node, const aiScene* scene, const int& frame)
{
	// Process all the node's meshes
	for (unsigned int i = 0; i < node->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		ProcessMeshSequence(mesh, scene, frame);
	}

	// then do the same for each of its children
	for (unsigned int i = 0; i < node->mNumChildren; i++)
	{
		ProcessNodeSequence(node->mChildren[i], scene, frame);
	}
}

void SimpleObject::ProcessMeshSequence(aiMesh* mesh, const aiScene* scene, const int& frame)
{
	// Process Vertices
	AnimVertex[frame] = vector<shared_ptr<MiniVertex>>(mesh->mNumVertices);

	#pragma omp parallel for
	for (int i = 0; i < mesh->mNumVertices; i++)
	{
		MiniVertex v;

		v.Pos.x = mesh->mVertices[i].x;
		v.Pos.y = mesh->mVertices[i].y;
		v.Pos.z = mesh->mVertices[i].z;

		v.Normal.x = mesh->mNormals[i].x;
		v.Normal.y = mesh->mNormals[i].y;
		v.Normal.z = mesh->mNormals[i].z;


		AnimVertex[frame][i] = make_shared<MiniVertex>(v);
	}
}

void SimpleObject::BuildTriangles()
{

	// Allocate memory
	NumTriangles = NumIndices / 3;
	TrianglesObj = vector<shared_ptr<SimpleTriangle>>(NumTriangles);

	// Fills array
	#pragma omp parallel for
	for (int i = 0; i < NumTriangles; i++)
	{
		// Fills Index Data
		SimpleTriangle t;
		t.V[0] = *Indices[3 * i + 0];
		t.V[1] = *Indices[3 * i + 1];
		t.V[2] = *Indices[3 * i + 2];

		// Node's Positions 
		const float3& v0 = Vertices[t.V[0]]->Pos;
		const float3& v1 = Vertices[t.V[1]]->Pos;
		const float3& v2 = Vertices[t.V[2]]->Pos;

		// Computes Attributes
		t.Center = (v0 + v1 + v2) / 3.f;
		t.Normal = normalize(cross(v1 - v0, v2 - v0));
		t.Idx = i;

		TrianglesObj[i] = make_shared<SimpleTriangle>(t);
	}
}

vector<SimpleTriangle> SimpleObject::GetTrianglesRaw()
{
	vector<SimpleTriangle> triangles(NumTriangles);

	// Fills Array
	#pragma omp parallel for
	for (int i = 0; i < NumTriangles; i++)
	{
		triangles[i] = *TrianglesObj[i];
	}

	return triangles;
}

vector<SimpleVertex> SimpleObject::GetVerticesRaw()
{
	vector<SimpleVertex> vertices(NumVertices);

	// Fills Array
	#pragma omp parallel for
	for (int i = 0; i < NumVertices; i++)
	{
		vertices[i] = *Vertices[i];
	}

	return vertices;
}

vector<MiniVertex> SimpleObject::GetAnimVerticesRaw()
{
	vector<MiniVertex> vertices(NumVertices * NumFrames);

	// Fills Array
	#pragma omp parallel for
	for (int frame = 0; frame < NumFrames; frame++)
	{
		#pragma omp parallel for
		for (int i = 0; i < NumVertices; i++)
		{
			vertices[NumVertices * frame + i] = *AnimVertex[frame][i];
		}
	}

	return vertices;
}

void SimpleObject::InitBoneData(SimpleVertex& v)
{
	for (int i = 0; i < MAX_BONE_NR; i++)
	{
		v.BoneIDs[i] = -1;
		v.BoneWeights[i] = 0.f;
	}
}