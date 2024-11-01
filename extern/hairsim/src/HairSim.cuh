#pragma once

#include "SimpleLoader.h"
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <surface_indirect_functions.h>
#include <surface_functions.h>


// Definitions for calling CUDA kernels

// Mesh triangles
#define INIT_T int tIdx = blockIdx.x*blockDim.x + threadIdx.x;
#define IDX_OUTSIDE_TRIANGLE (tIdx < 0 || tIdx > Params.NumTriangles - 1)

// Mesh vertices
#define INIT_V int vIdx = blockIdx.x*blockDim.x + threadIdx.x;
#define IDX_OUTSIDE_VERT (vIdx < 0 || vIdx > Params.NumVertices - 1)

// Eulerian grids
#define INIT_G int cellIdx = blockIdx.x*blockDim.x + threadIdx.x;
#define IDX_OUTSIDE_SDF (cellIdx < 0 || cellIdx > Params.NumSdfCells-1)
#define IDX_OUTSIDE_G (cellIdx < 0 || cellIdx > Params.NumGridCells - 1)
#define IDX_BOUNDARY_G (cellIdx == 0 || cellIdx == Params.NumGridCells - 1)
#define IDX_OUTSIDE_GU (cellIdx < 0 || cellIdx > Params.NumGridCellsU - 1)
#define IDX_BOUNDARY_GU (cellIdx == 0 || cellIdx == Params.NumGridCellsU - 1)
#define IDX_OUTSIDE_GV (cellIdx < 0 || cellIdx > Params.NumGridCellsV - 1)
#define IDX_BOUNDARY_GV (cellIdx == 0 || cellIdx == Params.NumGridCellsV - 1)
#define IDX_OUTSIDE_GW (cellIdx < 0 || cellIdx > Params.NumGridCellsW - 1)
#define IDX_BOUNDARY_GW (cellIdx == 0 || cellIdx == Params.NumGridCellsW - 1)

// Hair particles
#define INIT_P int pIdx = blockIdx.x*blockDim.x + threadIdx.x;
#define IDX_OUTSIDE_P (pIdx < 0 || pIdx > Params.NumParticles - 1)
#define IDX_BOUNDARY_P (pIdx == 0 || pIdx == Params.NumParticles - 1)
#define IDX_OUTSIDE_INTER (pIdx < 0 || pIdx > Params.NumInter -1)

// Hair roots
#define INIT_R int rIdx = blockIdx.x*blockDim.x + threadIdx.x;
#define IDX_OUTSIDE_R (rIdx < 0 || rIdx > Params.NumRoots - 1)
#define IDX_BOUNDARY_R (rIdx == 0 || rIdx == Params.NumRoots - 1)

// Experiment Option (select between experiments)
enum ExpOpt {
    CANTI_E, CANTI_L, SAG_WIND, SAG_HEAD, SANDBOX, WIG,
    ROLLER, SALON, KATE, CURLY, SPACE, FACIAL,
    MODERN, CHENGAN, SICTION, ANIMATION, NONE
};

enum AnimType
{
    SEQUENCE, BONE, WIND
};

// Type of cell for fluid solver
enum CellType
{
    FLUID, AIR, SOLID
};

// Wrapers for loading assets
struct HairInfo
{
    // Basic information
    string FileName;
    float4 Move;
    int3 EulerDim;
    int ID = -1;
};

struct HeadInfo
{
    // Basic information
    string FileName;
    float4 Move;
    int3 SdfDim;
    int ID = -1;

    // Animation data
    int NumFrames = 0;
    bool Animate = false;
};

struct InterParticle
{
    int2 VboIdx;
    int2 InterIdx;
    float Lambda;
};

struct Particle
{
    Particle(float3 position, int strandLength, int localIdx) : Position(position), StrandLength(strandLength), LocalIdx(localIdx) {}
    Particle() {}

    // Dynamic info
    float3 Position = make_float3(0.f); // params at t+dt
    float3 Velocity = make_float3(0.f);

    float3 InitialPosition = make_float3(0.f); // position before dynamics
    float3 Position0 = make_float3(0.f); // position at t

    // Geometry info
    int StrandLength = 0;
    int LocalIdx = 0;
    int GlobalIdx = 0;

    // Indices for finding rest length of one-phase springs 
    // (left and right, negative value means not connected)
    int2 EdgeRestIdx = { -1,-1 };
    int2 BendRestIdx = { -1,-1 };
    int2 TorsRestIdx = { -1,-1 };

    // Biphasic Interaction
    float3 GravityPos = make_float3(0.f);
    float3 GravityPos0 = make_float3(0.f);
    float3 Angular = make_float3(0.f);

    // Head Interaction
    //float3 PositionPreBone = make_float3(0.f);
    //float3 GravityPreBone = make_float3(0.f);

    // Cut Interaction
    bool Cut = false;
    int CutParent = -1;

    // Grab Interaction
    //bool BeingPulled = false;
    //float GrabL0 = 0.f;

    // Indices for mapping into OpenGL's VBO
    int2 VboIdx = { -1,-1 };
};

struct SimulationParams
{

    // Rigid head
    float3 RigidPos;
    float3 RigidAngle;

    int CutMin;


    // System dimensions
    int NumVertices; // number of vertices in main mesh
    int NumTriangles; // number of triangles in main mesh
    int NumParticles; // number of particles in hair discretization
    int NumInter;
    int NumSprings; // number of springs
    int NumRoots; // number of hair strands/roots
    int NumGridCells; // number of grid voxels
    int NumSdfCells; // number of grid voxels for sdf
    int NumGridCellsU; // number of grid voxels for staggered grids
    int NumGridCellsV;
    int NumGridCellsW;
    int MaxStrandLength; // used for uniform exporting

    // Sdf Grid
    float LengthIncreaseGrid; // percentage of grid increase w.r.t. loaded obj
    float ThreshSdf; // small threshold for head collision
    int3 SdfDim; // grid dimensions (nx, ny, nz)
    float3 SdfDs; // grid step sizes (dx, dy, dz)
    float3 SdfInvDs; // inverse step sizes (1/dx, 1/dy, 1/dz)
    float3 SdfInvSqDs; // inverse squared step sizes (1/dx^2,1/dy^2,1/dz^2)

    // Eulerian hair
    int3 HairDim; // grid dimensions (nx,ny,nz)
    float3 HairDs; // grid step sizes (dx,dy,dz)
    float3 HairInvDs; // inverse step sizes (1/dx, 1/dy, 1/dz)
    float3 HairInvSqDs; // inverse squared step sizes (1/dx^2,1/dy^2,1/dz^2)
    float3 HairAxis[3], HairAxis0[3]; // grid orientation (e_x, e_y, e_z)
    float3 HairOrig, HairOrig0; // grid local origin (0,0,0)
    float3 HairCenter, HairCenter0; // grid center

    // PIC/FLIP parameters
    int NumGridNeighbors; // number of neighbors to look for in particle2grid routine
    float MaxWeight; // maximum node weight
    int NumStepsJacobi; // number of iterations for Jacobi pressure solver
    float JacobiWeight; // (0,1] weight in Jacobi solver
    float FlipWeight; // (0,1) control of FLIP-PIC grid2particle weight
    int Parity; // Sdf parity (to avoid buggs if mesh was inverted)

    // Physical Parameters
    float Dt; // time step (s)
    float DtN; // nested time step (s)
    float InvDt; // inverse time step (s^{-1})
    float3 WindSpeed; // wind intensity (m/s)
    float Gravity; // gravity constant (m/s^2)
    float EdgeK; // spring constants (N/m)
    float BendK;
    float TorsionK;
    float AngularK; // (N/rad)
    float GravityK; // (N/m)
    float Damping; // particle damping
    float HairMass; // mass (kg)
    float Mu; // inverse hair mass (kg^{-1})
    float Friction; // friction coef. of head

    // Solver Parameters
    float StrainError;
    int StrainSteps;
    int NestedSteps;

    // Export Options
    bool SaveSelection;
    bool LoadSelection;

    // Preprocessing options
    bool FixRootPositions;
    bool TooglePreProcess;
    bool RecomputeSdf;

    // Experiments manuscript
    ExpOpt Experiment;

    // Import/Export
    int ExportStep; // to export at every i-th frame

    // Animation
    bool MeshSequence; // tracks if there is a loaded mesh sequence
    float SimEndTime; // stop animation here
    float SimTime; // 'inside' simulation time
    float AnimDuration; // time duration of animation
    int NumFrames; // number of frames for mesh sequence
    float3 RootMove;

    // Mesh/Bones
    int NumBonesInfluence; // max number of bones influencing a vertex
    int NumBones; // total number of bones for main mesh
    int SelectedBone; // for interacting via GUI

    // Toogle Options
    bool Animate; // start prescribed animatiomn
    bool Wind;
    AnimType AnimationType;
    bool PauseSim; // stop iterating simulation

    // Toogle export
    bool FixedHairLength; // export strands of equal size
    bool ExportSequence; // self-explained
    bool ExportDataFormat; // export hair in .data format
    bool ExportVDB; // export SDF in volumetric format
    bool ExportHair; // export hair in obj/data format
    bool ExportMesh; // export mesh in obj format
    bool Export; // general export toogle

    // Toogle Debug
    bool SolidCollision;
    bool HairCollision;
    bool DrawBoneWeight;
    bool DrawBones;
    bool DrawBonesAxis;
    bool DrawHead;
    bool DrawHeadWF;
    bool DrawGrid;

    // User Interaction
    //bool ToogleCutHair;
    //bool ToogleGrabHair;
    float CutRadius; // radious of sphere for cutting hair
    //float GrabRadius; // radius of cylinder for grabbing hair
    //float GrabK; // spring constant for grabbing
    //float3 GrabPos;
};

struct GenerationParams
{

    // Interpolation
    int NumInterpolated;
    bool Interpolate;

    // Procedural Parameters
    float StepLength;
    int StepsMin;
    int StepsMax;
    int HairsPerTri;
    float HairThickness;
    float GravityInfluence;
    float DirNoise;
    float GravNoise;
    float GravityDotInfluence;
    float SpiralRad;
    float FreqMult;
    float SpiralAmount;
    float SpiralY;
    float SpiralImpact;
    float PartingImpact;
    float PartingStrengthX;
    float PartingStrengthY;

    // Interaction params
    float SelectionRadius;
    float HairColor[3] = {(1.f/255) * 93, (1.f/255) * 60, (1.f/255) * 23};
    int HairDebbug = 0;

    // Loader options
    int MaxLoadStrandSize;
    int MaxLoadNumStrands;
};

struct DeviceBuffers
{

    // Interpolation
    InterParticle* InterParticles;

    // Other
    Mat4* RigidMotion;

    // Hair
    Particle* Particles;
    float* RestLenghts;
    int* RootIdx;

    // Solver
    Mat3* StrandA;
    Mat3* StrandL;
    Mat3* StrandU;
    float3* StrandV;
    float3* StrandB;

    // Eulerian
    CellType* VoxelType;
    float2* HairPressure;
    float* HairVelU;
    float* HairVelV;
    float* HairVelW;
    float* HairPicU;
    float* HairPicV;
    float* HairPicW;
    float* HairDiv;
    float* HairWeightU;
    float* HairWeightV;
    float* HairWeightW;

    // Head
    Mat4* FinalBoneTrans;
    SimpleTriangle* HeadTriangles;
    SimpleVertex* HeadVertices;
    float* Sdf;
    float3* NablaSdf;
    float* HeadVelX;
    float* HeadVelY;
    float* HeadVelZ;
    float* HeadVelWeight;

    // Animation
    MiniVertex* AnimVertices;
    float3* RootBary;
    int* RootTri;
};

// Init/set variables
void launchKernelResetParticles(const int2& blockThread, DeviceBuffers buffers);
void launchKernelInitParticles(const int2& blockThread, DeviceBuffers buffers);

// Integration
void launchKernelFillMatrices(const int2& blockThread, DeviceBuffers buffers);
void launchKernelSolveVelocity(const int2& blockThread, DeviceBuffers buffers);
void launchKernelPositionUpdate(const int2& blockThread, DeviceBuffers buffers);

// Eulerian solver
void launchKernelSegmentToGrid(const int2& blockThreadParticles, const int2& blockThreadGrid, const int2& blockThreadU, const int2& blockThreadV, const int2& blockThreadW, DeviceBuffers buffers);
void launchKernelGridToParticle(const int2& blockThread, DeviceBuffers buffers);
void launchKernelProjectVelocity(const int2& blockThreadGrid, const int2& blockThreadU, const int2& blockThreadV, const int2& blockThreadW, const int& numIter, DeviceBuffers buffers);

// Hair-solir solver
void launchKernelHeadCollision(const int2& blockThread, DeviceBuffers buffers);
void launchKernelNablaSdf(const int2& blockThread, DeviceBuffers buffers);
void launchKernelInitHeadVel(const int2& blockThread, DeviceBuffers buffers);
void launchKernelInitHeadVerticesVel(const int2& blockThread, DeviceBuffers buffers);
void launchKernelUpdateVelocitySdf(const int2& blockThreadSdf, const int2& blockThreadVertices, DeviceBuffers buffers);

// Additional dynamics
void launchKernelStrainLimiting(const int2& blockThread, DeviceBuffers buffers);
void launchKernelSwapPositions(const int2& blockThread, DeviceBuffers buffers);
void launchKernelMoveRoots(const int2& blockThread, DeviceBuffers buffers);
void launchKernelUpdateMesh(const int2& blockThread, DeviceBuffers buffers);

// Grooming
void launchKernelCutHair(const int2& blockThread, DeviceBuffers buffers);
void launchKernelCutSelect(const int2 &blockthread, const int2 &blockthreadRoot, float3 pos, float3 dir, DeviceBuffers buffers);

// Animation
void launchKernelUpdateRootsSeq(const int2& blockthread, DeviceBuffers buffers);
void launchKernelUpdateAnimSeq(const int2& blockthread, const int& a, const int& b, const float& lambda, DeviceBuffers buffers);

// Host helpers
void launchKernelCudaToGLInter(const int2& blockThread, float* hairVbo, DeviceBuffers buffers);
void launchKernelCudaToGLHair(const int2& blockThread, float* hairVbo, DeviceBuffers buffers);
void launchKernelCudaToGLMesh(const int2& blockThread, float* meshVbo, DeviceBuffers buffers);
void launchKernelSetRandom(const int2& blockThread, curandState* state);
void copySimParamsToDevice(SimulationParams* hostParams);

// Device helpers

// Computations
__host__ __device__ float3 BaryCoordinates(const float3& p, const float3& a, const float3& b, const float3& c);
__device__ float3 ExtForce(const int& globalI, const DeviceBuffers& buffers);
__device__ float3 SprForce(const int& globalI, const DeviceBuffers& buffers);
__device__ float3 SprForce(const int& globalI, const int& globalJ, const int& restIdx, const DeviceBuffers& buffers);
__device__ Mat3 DirMat(const int& globalI, const int& globalJ, const DeviceBuffers& buffers);
__device__ int3 PosToCellSdf(const float3& p);
__device__ int3 PosToCell(const float3& p);
__device__ float3 NormalPos(const float3& p, const int3& cellIdx);
__device__ bool CellInsideGrid(const int3& idx);
__device__ bool CellOutsideInfluence(const int3& idx);
__device__ float DistancePointSegment(const float3& A, const float3& B, const float3& u, float& t);

// Samplers
__inline__ __device__ Mat3 SampleA(const int& globalI, const int& globalJ, const DeviceBuffers& buffers);
__inline__ __device__ Mat3 SampleL(const int& globalI, const int& globalJ, const DeviceBuffers& buffers);
__inline__ __device__ Mat3 SampleU(const int& globalI, const int& globalJ, const DeviceBuffers& buffers);
__inline__ __device__ float3 SampleV(const int& globalI, const DeviceBuffers& buffers);
__device__ float3 SampleHairPIC(const int3& idx, const DeviceBuffers& buffers);
__device__ float3 SampleHairVel(const int3& idx, const DeviceBuffers& buffers);
__device__ float3 SampleHairFLIP(const int3& idx, const DeviceBuffers& buffers);
__device__ float SamplePressure(const int3& idx, const DeviceBuffers& buffers);
__device__ float4 TriInterpSdf(const int3& idx, const float3& pos, const DeviceBuffers& buffers);
__device__ float3 TriInterSdfVel(const int3& idx, const float3& pos, const DeviceBuffers& buffers);
__device__ float3 SampleNablaSdf(const int3& idx, const DeviceBuffers& buffers);
__device__ float SampleSdf(const int3& idx, const DeviceBuffers& buffers);
__device__ float4 SampleTotalSdf(const int3& idx, const DeviceBuffers& buffers);
__device__ float3 SampleSdfVel(const int3& idx, const DeviceBuffers& buffers);

// Indices
__inline__ __device__ int IdxA(const int& globalI, const int& globalJ);
__inline__ __device__ int IdxL(const int& globalI, const int& globalJ);
__inline__ __device__ int IdxU(const int& globalI, const int& globalJ);
__inline__ __device__ int3 Idx3DU(const int& idx);
__inline__ __device__ int3 Idx3DV(const int& idx);
__inline__ __device__ int3 Idx3DW(const int& idx);
__inline__ __device__ int3 Idx3D(const int& idx);
__inline__ __device__ int3 Idx3DSdf(const int& idx);
__inline__ __device__ int Idx1DU(const int3& idx);
__inline__ __device__ int Idx1DV(const int3& idx);
__inline__ __device__ int Idx1DW(const int3& idx);
__inline__ __device__ int Idx1D(const int3& idx);
__inline__ __device__ int Idx1DSdf(const int3& idx);

// // Helper functions for reading/writting over 3D textures
// template <class T> __inline__ __device__ void
// surfWrite(const T& value, const cudaSurfaceObject_t& surfObject, const int& x, const int& y, const int& z)
// {
//     surf3Dwrite<T>(value, surfObject, sizeof(T) * x, y, z);
    
// }

// template <class T> __inline__ __device__ T
// surfRead(const cudaSurfaceObject_t& surfObject, const int& x, const int& y, const int& z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
// {
//     T value;

//     surf3Dread<T>(&value, surfObject, x * sizeof(T), y, z, boundaryMode);
//     return value;
// }

// Trilinear interpolation
// https://handwiki.org/wiki/Trilinear_interpolation
template<typename T> __inline__ __device__ T
TriInterpolate(const float3& ud, const T& c000, const T& c001, const T& c010, const T& c011, const T& c100, const T& c101, const T& c110, const T& c111)
{
    T c00 = c000 * (1.f - ud.x) + c100 * ud.x;
    T c01 = c001 * (1.f - ud.x) + c101 * ud.x;
    T c10 = c010 * (1.f - ud.x) + c110 * ud.x;
    T c11 = c011 * (1.f - ud.x) + c111 * ud.x;

    T c0 = c00 * (1.f - ud.y) + c10 * ud.y;
    T c1 = c01 * (1.f - ud.y) + c11 * ud.y;

    T c = c0 * (1.f - ud.z) + c1 * ud.z;

    return c;
}
