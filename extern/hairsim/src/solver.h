#pragma once

#include <fstream>
// #include <ppl.h>
#include <omp.h>
#include <chrono>

#include "Rendering/VertexBufferObject.h"
#include "cubuffer.h"
#include "cudavbo.h"
#include "cudatextures.h"
#include "SFieldCPU.h"
#include "HairSim.cuh"
#include "HairGen.h"
#include "VdbExp.h"


class Solver
{
public:

	// Constructor
	Solver(const HairInfo& hair, const HeadInfo& head);
	~Solver();

	void Clear();
	void InitParams();

	// Init methods
	void InitBuffers(vector<shared_ptr<SimpleObject>>& objs, vector<GLuint>& vboIdx, shared_ptr<HairGenerator> hairGen);
	void InitParticles();
	
	// Updaters
	//void ComputeSDF();
	void ComputeSdfCPU();
	void ComputeBary();
	void UpdateSimulation();
	void UpdateProcessor();
	void Reset();

	// User Interaction
	void InitCut(float3 u, float3 dir);
	//void TriggerHairCut(float3& u, float3& dir);
	//void TriggerHairGrab(float3& u);
	//void UpdateGrabPos(float3& u);
	//void KillGrab();

	// Getters/setters
	SimulationParams& GetParamsSim() { return SimParams; }
	HeadInfo HeadInformation;


private:

	// Parameters
	SimulationParams SimParams;

	// Import/Export
	//vector<int> StrandsLength;
	//vector<int> CumulativeLength;
	void WriteVDB(const string& fileName);
	void WriteHair(const string& fileName);
	void WriteObj(const string& fileName);
	void WriteData();
	void ExportData();

	// Step Update
	void UpdateAnimation();
	void UpdateDynamics();
	void UpdateRoots();
	//void UpdateSimFrame();
	void UpdateMesh();
	//void UpdateRigidSDF();
	//void UpdateSDFManual();

	// Preprocessing
	void FixRootPositions();

	// Custom Animations (for experiments)
	//void RigidHeadMotion(vector<Bone>& bones);
	//void modern(vector<Bone>& bones);
	//void WindBlowingOneSide();
	void WindBlowing();
	//void Curly();
	//void chengan1(vector<Bone>& bones);
	//void chengan2(vector<Bone>& bones);
	//void BarbaCorta(vector<Bone>& bones);
	//void BarbaLarga(vector<Bone>& bones);
	//void RollerCoaster(vector<Bone>& bones);

	// Rigid head-motion
	shared_ptr<CuBuffer<Mat4>> RigidMotion;

	// Interpolation
	shared_ptr<CuBuffer<InterParticle>> InterParticles;

	// Hair Data
	shared_ptr<CuBuffer<Particle>> Particles;
	shared_ptr<CuBuffer<float>> RestLengths;
	shared_ptr<CuBuffer<int>> RootIdx;

	// Heptadiagional sparse LU solver
	shared_ptr<CuBuffer<Mat3>> StrandA;
	shared_ptr<CuBuffer<Mat3>> StrandL;
	shared_ptr<CuBuffer<Mat3>> StrandU;
	shared_ptr<CuBuffer<float3>> StrandV;
	shared_ptr<CuBuffer<float3>> StrandB;

	// Hair-hair interactions
	shared_ptr<CuBuffer<float>> HairWeightU; // hair-"fluid" weight U
	shared_ptr<CuBuffer<float>> HairWeightV; // hair-"fluid" weight V
	shared_ptr<CuBuffer<float>> HairWeightW; // hair-"fluid" weight W

	shared_ptr<CuBuffer<float>> HairVelU; // hair-"fluid" velocity U
	shared_ptr<CuBuffer<float>> HairVelV; // hair-"fluid" velocity V
	shared_ptr<CuBuffer<float>> HairVelW; // hair-"fluid" velocity W

	shared_ptr<CuBuffer<float2>> HairPressure; // hair-"fluid" pressure
	shared_ptr<CuBuffer<CellType>> VoxelType; // to categorize all voxels
	shared_ptr<CuBuffer<float>> HairPicU; // incompressible velocity fields
	shared_ptr<CuBuffer<float>> HairPicV;
	shared_ptr<CuBuffer<float>> HairPicW;
	shared_ptr<CuBuffer<float>> HairDiv; // divergence hair fluid field

	// Hair-solid interactions
	shared_ptr<CuBuffer<float3>> RootBary; // barycentric coordinates of roots
	shared_ptr<CuBuffer<int>> RootTri; // triangles to wich roots are attached
	shared_ptr<CuBuffer<float>> HeadVelX; // head velocity field
	shared_ptr<CuBuffer<float>> HeadVelY;
	shared_ptr<CuBuffer<float>> HeadVelZ;
	shared_ptr<CuBuffer<float>> HeadVelWeight;
	shared_ptr<CuBuffer<float>> Sdf; // head sdf
	shared_ptr<CuBuffer<float3>> NablaSdf;

	// Solid Object Data
	shared_ptr<CuBuffer<SimpleTriangle>> HeadTriangles; // head triangle GPU
	shared_ptr<CuBuffer<SimpleVertex>> HeadVertices; // head nodes GPU
	shared_ptr<CuBuffer<MiniVertex>> AnimVertices; // sequence of vertices GPU
	vector<shared_ptr<SimpleObject>> Meshes; // head & others
	shared_ptr<CuBuffer<Mat4>> FinalBoneTrans; // bone trans head
	shared_ptr<SFieldCPU> SdfCPU; // cpu sdf computation

	// Misc
	DeviceBuffers DBuffers;
	curandState* RandStateGrid;

	// CUDA to VBO
	vector<shared_ptr<VboResource<float>>> MainMappers; // head, hair

	// CUDA Kernel Dims
	int2 BlockThreadGridU = make_int2(0, 256); // Number of blocks and threads for U-grid in CUDA
	int2 BlockThreadGridV = make_int2(0, 256); // %% for V-grid in CUDA
	int2 BlockThreadGridW = make_int2(0, 256); // %% for W-grid in CUDA
	int2 BlockThreadHair = make_int2(0, 256);  // %% for hair in CUDA
	int2 BlockThreadRoot = make_int2(0, 256);  // %% for roots in CUDA
	int2 BlockThreadMesh = make_int2(0, 256);  // % for triangles in CUDA
	int2 BlockThreadVert = make_int2(0, 256);  // % for vertices in CUDA
	int2 BlockThreadGrid = make_int2(0, 256);  // % for grid in CUDA
	int2 BlockThreadSdf = make_int2(0, 256);   // % for sdf grid in CUDA
	int2 BlockThreadInter = make_int2(0, 256); // % for hair interpolation in CUDA 

	// Helpers
	//float DynamicQuant(vector<float> times, vector<float> values, float t);
	void UpdateAnimGridBox(const int& a, const int& b, const float& lambda);
	void AnimTimeToFrame(int& a, int& b, float& lambda);
	int DivUp(const int& a, const int& b); // Computes necesary number of blocks given number of threads
	inline Eigen::Vector3f Float2eigen(const float3& u) { return Eigen::Vector3f(u.x, u.y, u.z); }
	inline float3 Eigen2float(const Eigen::Vector3f& u) { return make_float3(u[0], u[1], u[2]); }
	void FreeBuffers();
	int SelectCudaDevice();
	void MapOpenGL();
	vector<int> StrandLength;
	int ExportCounter = 0;
	int ExportIdx = 0;

};