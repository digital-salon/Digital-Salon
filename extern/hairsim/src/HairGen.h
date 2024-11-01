#pragma once


#include "HairLoader.h"
#include "HairSim.cuh"

#include <random>

// Useful Structs
struct HairRoot
{
    float3 Pos = make_float3(0.f);
    float3 Normal = make_float3(0.f);
    int Steps = 0;
    int TriangleIdx = -1;
};

struct HairNode
{
    HairNode& operator = (const HairNode& n)
    {
        Pos = n.Pos;
        Dir = n.Dir;
        Ortho = n.Ortho;
        Par = n.Par;
        Thick = n.Thick;
        VboIdx = n.VboIdx;
        InterpolIdx = n.InterpolIdx;
        Interpolated = n.Interpolated;
        Lambda = n.Lambda;
        return *this;
    }

    HairNode(const HairNode& n)
        : Pos(n.Pos),
        Dir(n.Dir),
        Ortho(n.Ortho),
        Par(n.Par),
        Thick(n.Thick),
        VboIdx(n.VboIdx),
        InterpolIdx(n.InterpolIdx),
        Interpolated(n.Interpolated),
        Lambda(n.Lambda)
    {}

    HairNode(float3 pos, float3 dir, float thick) : Pos(pos), Dir(dir), Thick(thick) {}

    HairNode() {}

    // Data for procedural generation
    float3 Pos;
    float3 Par;
    float3 Ortho;
    float3 Dir;
    float Thick;

    // Data for rendering
    // each point appears at most twice on VBO since we draw lines
    // per segments: (a,b), (b,c), (c,d)...
    int2 VboIdx = { -1,-1 }; 


    // Interpolation
    int2 InterpolIdx = {-1, -1};
    bool Interpolated = false;
    float Lambda = -1.f;
};


class SFieldCPU;

class HairGenerator
{
public:
    // Constructors
    HairGenerator();
    HairGenerator(vector<shared_ptr<SimpleObject>>& meshes);

    // Interpolation from loaded hair
    vector<vector<HairNode>> InterpolateHair(const vector<vector<HairNode>>& hair);
    vector<vector<HairNode>> InterpolateHairSecond(const vector<vector<HairNode>>& hair);
    vector<vector<HairNode>> InterpolateHairThird(const vector<vector<HairNode>>& hair);
    vector<int> ClosestRoots(const vector<HairNode>& target, const vector<vector<HairNode>>& hair, int n);
    int theClosestRoot(const vector<HairNode>& target, const vector<vector<HairNode>>& hair, int n);
    vector<int> ClosestRootsTriangle(const vector<HairNode>& target, const vector<vector<HairNode>>& hair, const vector<int>& hairTri, int targetIdx, int n);
    double RootDistance(const vector<HairNode>& a, const vector<HairNode>& b);
    int SampleRoot(const vector<int>& roots);
    vector<HairNode> InterpolateStrands(const vector<HairNode>& a, const vector<HairNode>& b, const int& na, const int& nb);

    // Methods
    vector<vector<HairNode>> LoadHair(const string& fname, const float4& move);
    vector<vector<HairNode>> CreateTestSiction(const int& numStrands, const float4& limits, const float& height);
    vector<vector<HairNode>> GenerateStrands();
    void GenerateRoots();
    void GenerateParticles();
    void Clear();

    //void GenerateRod(const float3& x0, const float3& xn, const int& steps);

    // Setters/Getters
    vector<InterParticle>& InterParticles() {return mInterParticles; }
    vector<Particle>& GetParticles() { return Particles; }
    vector<int>& GetRootIdx() { return RootIdx; }
    vector<float>& GetRestLengths() { return RestLengths; }
    vector<int>& GetStrandLengths() { return StrandsLength; }
    vector<vector<HairNode>>& GetStrands() { return Strands; }
    GenerationParams& GetParamsGen() { return GenParams; }
    int GetNumRoots() { return Roots.size(); }
    int GetMaxStrandLength() { return MaxStrandLength; }

private:

    // Interpolation
    vector<InterParticle> mInterParticles;
    shared_ptr<SFieldCPU> SdfCPU; 

    // Parameters
    GenerationParams GenParams;

    // Data for simulator
    vector<Particle> Particles;
    vector<float> RestLengths;
    vector<int> RootIdx;

    // Misc data
    vector<int> StrandsLength;
    int MaxStrandLength = 0;

    // Geometry
    shared_ptr<SimpleObject> MainMesh; // head
    vector<HairRoot> Roots; // per-generation roots (restart each time)
    vector<vector<HairNode>> Strands; // all generated strands

    // Load Hair
    vector<vector<HairNode>> External2Strand(const float4& move);
    Loader HairLoader;

    // Misc Helpers
    void InitParams();
    void ComputePFTs(vector<HairNode>& strand);
    void LoadComputePFTs(vector<HairNode>& strand);
    inline float3 Eigen2float(const Eigen::Vector3f& u) { return make_float3(u[0], u[1], u[2]); }
};