#include "solver.h"

Solver::Solver(const HairInfo& hair, const HeadInfo& head)
{
    SelectCudaDevice();
    InitParams();
    SimParams.HairDim = hair.EulerDim;
    SimParams.SdfDim = head.SdfDim;
}

Solver::~Solver()
{
    FreeBuffers();
}

void Solver::Clear()
{
    
}

void Solver::InitBuffers(vector<shared_ptr<SimpleObject>>& objs, vector<GLuint>& vboIdx, shared_ptr<HairGenerator> hairGen)
{
    // Unwrap generator
    vector<float>& restLengths = hairGen->GetRestLengths();
    vector<Particle>& particles = hairGen->GetParticles();
    vector<InterParticle>& interParticles = hairGen->InterParticles();
    vector<int>& rootIdx = hairGen->GetRootIdx();
    Meshes = objs;

    // System Dimensions
    int numGridCellsU = (SimParams.HairDim.x + 1) * SimParams.HairDim.y * SimParams.HairDim.z;
    int numGridCellsV = SimParams.HairDim.x * (SimParams.HairDim.y + 1) * SimParams.HairDim.z;
    int numGridCellsW = SimParams.HairDim.x * SimParams.HairDim.y * (SimParams.HairDim.z + 1);
    int numGridCells = SimParams.HairDim.x * SimParams.HairDim.y * SimParams.HairDim.z;
    int numSdfCells = SimParams.SdfDim.x * SimParams.SdfDim.y * SimParams.SdfDim.z;
    int numTriangles = objs[0]->GetNumTriangles();
    int numVertices = objs[0]->GetNumVertices();
    int numFrames = objs[0]->GetNumFrames();
    int numBones = objs[0]->GetNumBones();
    int numParticles = particles.size();
    int numRoots = rootIdx.size();

    SimParams.MaxStrandLength = hairGen->GetMaxStrandLength();
    StrandLength = hairGen->GetStrandLengths();
    SimParams.NumGridCellsU = numGridCellsU;
    SimParams.NumGridCellsV = numGridCellsV;
    SimParams.NumGridCellsW = numGridCellsW;
    SimParams.NumParticles = numParticles;
    SimParams.NumInter = interParticles.size();
    SimParams.NumTriangles = numTriangles;
    SimParams.NumGridCells = numGridCells;
    SimParams.NumSdfCells = numSdfCells;
    SimParams.NumVertices = numVertices;
    SimParams.NumFrames = numFrames;
    SimParams.NumRoots = numRoots;
    SimParams.MeshSequence = numFrames > 1;

    // Cuda-OpenGL Mappers (0-head, 1-hair)
    MainMappers = vector<shared_ptr<VboResource<float>>>(2);
    MainMappers[0] = make_shared<VboResource<float>>(vboIdx[0]);
    MainMappers[1] = make_shared<VboResource<float>>(vboIdx[1]);

    //Allocate GPU Memory
    
    // Eulerian hair
    HairWeightU = make_shared<CuBuffer<float>>(numGridCellsU);
    HairWeightV = make_shared<CuBuffer<float>>(numGridCellsV);
    HairWeightW = make_shared<CuBuffer<float>>(numGridCellsW);
    HairVelU = make_shared<CuBuffer<float>>(numGridCellsU);
    HairVelV = make_shared<CuBuffer<float>>(numGridCellsV);
    HairVelW = make_shared<CuBuffer<float>>(numGridCellsW);
    HairPicU = make_shared<CuBuffer<float>>(numGridCellsU);
    HairPicV = make_shared<CuBuffer<float>>(numGridCellsV);
    HairPicW = make_shared<CuBuffer<float>>(numGridCellsW);
    HairPressure = make_shared<CuBuffer<float2>>(numGridCells);
    VoxelType = make_shared<CuBuffer<CellType>>(numGridCells);
    HairDiv = make_shared<CuBuffer<float>>(numGridCells);

    DBuffers.HairPressure = HairPressure->DevPtr();
    DBuffers.HairWeightU = HairWeightU->DevPtr();
    DBuffers.HairWeightV = HairWeightV->DevPtr();
    DBuffers.HairWeightW = HairWeightW->DevPtr();
    DBuffers.VoxelType = VoxelType->DevPtr();
    DBuffers.HairVelU = HairVelU->DevPtr();
    DBuffers.HairVelV = HairVelV->DevPtr();
    DBuffers.HairVelW = HairVelW->DevPtr();
    DBuffers.HairPicU = HairPicU->DevPtr();
    DBuffers.HairPicV = HairPicV->DevPtr();
    DBuffers.HairPicW = HairPicW->DevPtr();
    DBuffers.HairDiv = HairDiv->DevPtr();

    // Lagrangian Hair
    InterParticles = make_shared<CuBuffer<InterParticle>>(interParticles.size(), true);
    Particles = make_shared<CuBuffer<Particle>>(numParticles, true);
    RestLengths = make_shared<CuBuffer<float>>(restLengths.size());
    RootIdx = make_shared<CuBuffer<int>>(numRoots, true);

    DBuffers.RestLenghts = RestLengths->DevPtr();
    DBuffers.Particles = Particles->DevPtr();
    DBuffers.InterParticles = InterParticles->DevPtr();
    DBuffers.RootIdx = RootIdx->DevPtr();

    RestLengths->CopyHostToDevice(restLengths.data());
    Particles->CopyHostToDevice(particles.data());
    InterParticles->CopyHostToDevice(interParticles.data());
    RootIdx->CopyHostToDevice(rootIdx.data());

    Particles->CopyDeviceToHost();
    RootIdx->CopyDeviceToHost();

    // Heptadiagional solver
    StrandA = make_shared<CuBuffer<Mat3>>(7 * numParticles);
    StrandL = make_shared<CuBuffer<Mat3>>(4 * numParticles);
    StrandU = make_shared<CuBuffer<Mat3>>(4 * numParticles);
    StrandV = make_shared<CuBuffer<float3>>(numParticles);
    StrandB = make_shared<CuBuffer<float3>>(numParticles);

    DBuffers.StrandL = StrandL->DevPtr();
    DBuffers.StrandU = StrandU->DevPtr();
    DBuffers.StrandV = StrandV->DevPtr();
    DBuffers.StrandA = StrandA->DevPtr();
    DBuffers.StrandB = StrandB->DevPtr();

    // Head
    HeadVertices = make_shared<CuBuffer<SimpleVertex>>(numVertices, true);
    HeadTriangles = make_shared<CuBuffer<SimpleTriangle>>(numTriangles);
    RigidMotion = make_shared<CuBuffer<Mat4>>(3);
    FinalBoneTrans = make_shared<CuBuffer<Mat4>>(numBones);
    Sdf = make_shared<CuBuffer<float>>(numSdfCells, true);
    HeadVelX = make_shared<CuBuffer<float>>(numSdfCells);
    HeadVelY = make_shared<CuBuffer<float>>(numSdfCells);
    HeadVelZ = make_shared<CuBuffer<float>>(numSdfCells);
    HeadVelWeight = make_shared<CuBuffer<float>>(numSdfCells);
    NablaSdf = make_shared<CuBuffer<float3>>(numSdfCells);
    SdfCPU = make_shared<SFieldCPU>(objs[0]);
    RootBary = make_shared<CuBuffer<float3>>(numRoots, true);
    RootTri = make_shared<CuBuffer<int>>(numRoots, true);

    DBuffers.HeadTriangles = HeadTriangles->DevPtr();
    DBuffers.HeadVertices = HeadVertices->DevPtr();
    DBuffers.FinalBoneTrans = FinalBoneTrans->DevPtr();
    DBuffers.RigidMotion = RigidMotion->DevPtr();
    DBuffers.Sdf = Sdf->DevPtr();
    DBuffers.NablaSdf = NablaSdf->DevPtr();
    DBuffers.RootBary = RootBary->DevPtr();
    DBuffers.RootTri = RootTri->DevPtr();
    DBuffers.HeadVelX = HeadVelX->DevPtr();
    DBuffers.HeadVelY = HeadVelY->DevPtr();
    DBuffers.HeadVelZ = HeadVelZ->DevPtr();
    DBuffers.HeadVelWeight = HeadVelWeight->DevPtr();

    HeadTriangles->CopyHostToDevice(objs[0]->GetTrianglesRaw().data());
    HeadVertices->CopyHostToDevice(objs[0]->GetVerticesRaw().data());
    FinalBoneTrans->CopyHostToDevice(objs[0]->GetFinalTrans().data());

    vector<Mat4> NoTrans = {Mat4::Identity(), Mat4::Identity(), Mat4::Identity()};
    RigidMotion->CopyHostToDevice(NoTrans.data());

    // Animation (mesh sequence)
    if (SimParams.MeshSequence)
    {
        AnimVertices = make_shared<CuBuffer<MiniVertex>>(numFrames * numVertices);
        DBuffers.AnimVertices = AnimVertices->DevPtr();
        AnimVertices->CopyHostToDevice(objs[0]->GetAnimVerticesRaw().data());
    }

    // Compute block sizes for CUDA
    BlockThreadHair.x = DivUp(numParticles, BlockThreadHair.y);
    BlockThreadMesh.x = DivUp(numTriangles, BlockThreadMesh.y);
    BlockThreadVert.x = DivUp(numVertices, BlockThreadVert.y);
    BlockThreadGridU.x = DivUp(numGridCellsU, BlockThreadGridU.y);
    BlockThreadGridV.x = DivUp(numGridCellsV, BlockThreadGridV.y);
    BlockThreadGridW.x = DivUp(numGridCellsW, BlockThreadGridW.y);
    BlockThreadGrid.x = DivUp(numGridCells, BlockThreadGrid.y);
    BlockThreadSdf.x = DivUp(numSdfCells, BlockThreadSdf.y);
    BlockThreadRoot.x = DivUp(numRoots, BlockThreadRoot.y);
    BlockThreadInter.x = DivUp(interParticles.size(), BlockThreadInter.y);

    // Params to GPU
    copySimParamsToDevice(&SimParams);

    //Random state CUDA (for grid)
    cudaMalloc(&RandStateGrid, numGridCells * sizeof(curandState));
    launchKernelSetRandom(BlockThreadGrid, RandStateGrid);
    ///////////

    if (VERBOSE) printf("there are %i particles on GPU. \n", numParticles);
    if (VERBOSE) printf("CUDA will use %i blocks, %i threads for the hair", BlockThreadHair.x, BlockThreadHair.y);
    if (VERBOSE) printf("; also %i blocks, %i threads for the grid\n", BlockThreadGrid.x, BlockThreadGrid.y);
}

void Solver::InitParticles()
{
    launchKernelInitParticles(BlockThreadHair, DBuffers);

    // Init head velocity
    launchKernelInitHeadVerticesVel(BlockThreadVert, DBuffers);
    launchKernelInitHeadVel(BlockThreadSdf, DBuffers);
}

void Solver::ComputeSdfCPU()
{
    // Build on CPU
    int gridSize = SimParams.SdfDim.x * SimParams.SdfDim.y * SimParams.SdfDim.z;
    SdfCPU->BuildSDF(SimParams.SdfDim, SimParams.HairOrig, SimParams.SdfDs);

    #pragma omp parallel for
    for (int i = 0; i < gridSize; i++)
    {
        Sdf->HostPtr()[i] = SimParams.Parity * SdfCPU->GetSDF()[i];
    }

    // Copy to GPU
    Sdf->CopyHostToDevice();

    // Compute the gradient
    launchKernelNablaSdf(BlockThreadSdf, DBuffers);

    //cout << "Sanity Check" << endl;
    //printf("Sdf Dim (%i,%i,%i)\n", SimParams.SdfDim.x, SimParams.SdfDim.y, SimParams.SdfDim.z);
    //printf("Ds (%f,%f,%f)\n", SimParams.SdfDs.x, SimParams.SdfDs.y, SimParams.SdfDs.z);
    //printf("Orig (%f, %f, %f)\n", SimParams.HairOrig.x, SimParams.HairOrig.y, SimParams.HairOrig.z);

}

void Solver::ComputeBary()
{
    // We need the SDf, so we do these computations on CPU
    Particles->CopyDeviceToHost();
    RootIdx->CopyDeviceToHost();

    // Unwrap
    vector<shared_ptr<SimpleTriangle>>& triangles = Meshes[0]->GetTriangles();
    vector<shared_ptr<SimpleVertex>>& vertices = Meshes[0]->GetVertices();
    Particle* particlesCPU = Particles->HostPtr();
    float3* baryCPU = RootBary->HostPtr();
    int* idxCPU = RootIdx->HostPtr();
    int* triCPU = RootTri->HostPtr();

    // Fills in parallel-cpu
    #pragma omp parallel for
    for (int i = 0; i < SimParams.NumRoots; i++)
    {
        // Get closest triangle to this root
        PointSDF sample = SdfCPU->DistancePointField(particlesCPU[idxCPU[i]].Position);
        triCPU[i] = sample.TriIdx;

        // Get barycentric coordinates
        shared_ptr<SimpleTriangle>& triangle = triangles[sample.TriIdx];
        float3 a = vertices[triangle->V[0]]->Pos;
        float3 b = vertices[triangle->V[1]]->Pos;
        float3 c = vertices[triangle->V[2]]->Pos;
        baryCPU[i] = BaryCoordinates(sample.NearestPoint, a, b, c);
    }

    // Copy everything to GPU
    RootBary->CopyHostToDevice();
    RootTri->CopyHostToDevice();
}

void Solver::UpdateSimulation()
{

    // Update parameters on GPU
    // also updates inverse mass and nested dt as it may be
    // that some parameter was changed
    SimParams.Mu = 1.f / SimParams.HairMass;
    SimParams.DtN = SimParams.Dt / (1.f * SimParams.NestedSteps);
    SimParams.InvDt = 1.f / SimParams.Dt;

    Mat4 rot = Mat4::RotateZ(SimParams.RigidAngle.z) *  Mat4::RotateY(SimParams.RigidAngle.y) * Mat4::RotateX(SimParams.RigidAngle.x);
    Mat4 rigid = Mat4::Translate(SimParams.RigidPos) *  rot;
    Mat4 inv = Mat4::Transpose(rigid.Inverse());
    vector<Mat4> mats = {rigid, inv, rigid.Inverse()};
    RigidMotion->CopyHostToDevice(mats.data());
    SimParams.HairOrig = rigid * SimParams.HairOrig0;
    SimParams.HairCenter = rigid * SimParams.HairCenter0;
    SimParams.HairAxis[0] = rot * SimParams.HairAxis0[0];
    SimParams.HairAxis[1] = rot * SimParams.HairAxis0[1];
    SimParams.HairAxis[2] = rot * SimParams.HairAxis0[2];
    copySimParamsToDevice(&SimParams);

    // Custom Animation
    if(SimParams.Animate) UpdateAnimation();

    // Update Solid Geometry
    //UpdateMesh();
    //UpdateRigidSDF();
    UpdateRoots();

    // Update Hair
    //UpdateSDFManual();
    UpdateDynamics();

    // Maps to OpenGl
    MapOpenGL();

    // Export
    ExportData();

    // Elapsed time
    if (SimParams.Animate) SimParams.SimTime += SimParams.Dt;

    // Pause for debugging
    //SimParams.PauseSim = true;
}

void Solver::UpdateProcessor()
{
    // Update parameters on GPU
    // also updates inverse mass and nested dt as it may be
    // that some parameter was changed
    SimParams.Mu = 1.f / SimParams.HairMass;
    SimParams.DtN = SimParams.Dt / (1.f * SimParams.NestedSteps);
    copySimParamsToDevice(&SimParams);

    // Move roots to head
    FixRootPositions();

    // Maps to OpenGl
    MapOpenGL();

    // Export
    ExportData();
}

void Solver::Reset()
{
    // Restart particles
    launchKernelResetParticles(BlockThreadHair, DBuffers);

    // Restar export indices
    ExportCounter = 0;
    ExportIdx = 0;
    SimParams.SimTime = 0.f;
}

void Solver::InitCut(float3 u, float3 dir)
{
    launchKernelCutSelect(BlockThreadHair, BlockThreadRoot, u, dir, DBuffers);
}

void Solver::WriteVDB(const string& fileName)
{
#ifdef WITH_OPENVDB
    VdbExp vdbExport;
    vdbExport.AddGridData("SDF", Sdf->HostPtr(), SimParams.SdfDim, 1.f, true);
    vdbExport.Save(fileName, false);
#endif
}

void Solver::WriteHair(const string& fileName)
{

    // Temporal object for saving
    Loader hair;

    // Copy from GPU to CPU
    Particles->CopyDeviceToHost();
    hair.strands = vector<vector<Eigen::Vector3f>>(SimParams.NumRoots);

    #pragma omp parallel for
    for (int i = 0; i < SimParams.NumRoots; i++)
    {
        int rootIdx = RootIdx->HostPtr()[i];
        hair.strands[i] = vector<Eigen::Vector3f>(StrandLength[i]);
        #pragma omp parallel for
        for (int j = 0; j < StrandLength[i]; j++)
        {
            hair.strands[i][j] = Float2eigen(Particles->HostPtr()[rootIdx + j].Position);
        }
    }

    // Makes sure all strands have the same size
    if (SimParams.FixedHairLength)
    {
        #pragma omp parallel for
        for (int i = 0; i < SimParams.NumRoots; i++)
        {
            // Continue in the last strand direction
            int lastIdx = RootIdx->HostPtr()[i] + StrandLength[i] - 1;
            float3 dir = Particles->HostPtr()[lastIdx].Position - Particles->HostPtr()[lastIdx - 1].Position;
            dir = 0.001 * normalize(dir);

            // Check if condition already met
            vector<Eigen::Vector3f>& strand = hair.strands[i];
            while (strand.size() < SimParams.MaxStrandLength)
            {
                strand.push_back(Float2eigen(Eigen2float(strand.back()) + dir));
            }
        }
    }

    // Export
    if (SimParams.ExportDataFormat) hair.save_data(fileName + ".data");
    else hair.save_obj(fileName + ".obj");
}

void Solver::WriteObj(const string& fileName)
{
    // Update data on CPU
    if (SimParams.AnimationType == SEQUENCE && SimParams.Animate == true)
    {
        Meshes[0]->UpdateVerticesFromSeq(HeadVertices->HostPtr());
    }

    // Writes
    Meshes[0]->ExportObj(fileName + ".obj");
}

void Solver::WriteData()
{
    string timeStamp = SimParams.ExportSequence ? "_" + to_string(ExportIdx) : "";
    string folder = "./Export/";

    if (SimParams.ExportHair) WriteHair(folder + "hair" + timeStamp);
    if (SimParams.ExportMesh) WriteObj(folder + "mesh" + timeStamp);
    if (SimParams.ExportVDB) WriteVDB(folder + "sdf" + timeStamp);
}

void Solver::ExportData()
{
    if (SimParams.Export)
    {

        // export whole data sequence
        if (SimParams.ExportSequence)
        {
            if (ExportCounter % SimParams.ExportStep == 0)
            {
                WriteData();
                ExportCounter++;
                ExportIdx++;
            }
            else
            {
                ExportCounter++;
            }
        }
        // export single file
        else
        {
            WriteData();
            SimParams.ExportHair = false;
            SimParams.ExportMesh = false;
            SimParams.ExportVDB = false;
            SimParams.Export = false;
        }
    }
}

void Solver::UpdateAnimation()
{
    switch (SimParams.AnimationType)
    {
    case(SEQUENCE):
    {
        // transforms time to equivalent frame
        float lambda;
        int a, b;
        AnimTimeToFrame(a, b, lambda);

        // update vertices from sequence
        launchKernelUpdateAnimSeq(BlockThreadVert, a, b, lambda, DBuffers);

        // update velocity head (particle to grid)
        launchKernelUpdateVelocitySdf(BlockThreadSdf, BlockThreadVert, DBuffers);

        // update hair roots
        launchKernelUpdateRootsSeq(BlockThreadRoot, DBuffers);

        // update corresponding box
        UpdateAnimGridBox(a, b, lambda);

        // update Sdf
        HeadVertices->CopyDeviceToHost();
        SdfCPU->UpdateVertices(HeadVertices->HostPtr());
        ComputeSdfCPU();

        break;
    }
    case (WIND):
    {
        WindBlowing();
    }
    default:
        break;
    }
}

void Solver::UpdateDynamics()
{
    // Nested Integration
    for (int i = 0; i < SimParams.NestedSteps; i++)
    {
        // Prepare linear system
        launchKernelFillMatrices(BlockThreadHair, DBuffers);

        // Solve implicit Velocity
        launchKernelSolveVelocity(BlockThreadRoot, DBuffers);

        // Update positions
        launchKernelPositionUpdate(BlockThreadHair, DBuffers);
    }

    // Head collision
    if (SimParams.SolidCollision) launchKernelHeadCollision(BlockThreadHair, DBuffers);

    //Strain limiting
    launchKernelStrainLimiting(BlockThreadRoot, DBuffers);

    // Hair-hair interaction
    if (SimParams.HairCollision)
    {
        // Eulerian part (coarse collision detection)
        
        // Rasterize particles into grid
        launchKernelSegmentToGrid(BlockThreadHair, BlockThreadGrid, BlockThreadGridU, BlockThreadGridV, BlockThreadGridW, DBuffers);

        // Enforce incompressible condition
        launchKernelProjectVelocity(BlockThreadGrid, BlockThreadGridU, BlockThreadGridV, BlockThreadGridW, SimParams.NumStepsJacobi, DBuffers);

        // Transfer velocity back to particles
        launchKernelGridToParticle(BlockThreadHair, DBuffers);

        // Lagrangian part (detailed collision detection)
    }

    // Swaps positions (x and x0)
    launchKernelSwapPositions(BlockThreadHair, DBuffers);

    // Update Cut Particles
    launchKernelCutHair(BlockThreadHair, DBuffers);
}

void Solver::UpdateRoots()
{
    launchKernelMoveRoots(BlockThreadHair, DBuffers);
}

void Solver::UpdateMesh()
{
    launchKernelUpdateMesh(BlockThreadMesh, DBuffers);
}

void Solver::FixRootPositions()
{
    if (SimParams.FixRootPositions)
    {
        // Gets particles on CPU
        Particles->CopyDeviceToHost();
        Particle* particlesCPU = Particles->HostPtr();

        // Fills in parallel-cpu
        #pragma omp parallel for
        for (int i = 0; i < SimParams.NumParticles; i++)
        {
            // Only moves roots
            if (particlesCPU[i].LocalIdx == 0)
            {
                PointSDF sample = SdfCPU->DistancePointField(particlesCPU[i].Position);
                float3 dirVector = sample.NearestPoint - particlesCPU[i].Position;
                for (int j = 0; j < particlesCPU[i].StrandLength; j++)
                {
                    particlesCPU[i + j].Position += dirVector;
                }
            }
        }

        // Copy everything to GPU
        Particles->CopyHostToDevice();
        SimParams.FixRootPositions = false;
    }
}

void Solver::WindBlowing()
{
    float t = SimParams.SimTime;
    float PI = 3.14159f;
    //if (t < 5.f * PI / 2.f || t > 20.f * PI / 3.f) SimParams.WindSpeed.x = 0.f;
    //else SimParams.WindSpeed.x = 4.8f * sinf(1.2 * t);
    //SimParams.WindSpeed.x = t;
    //SimParams.WindSpeed.x *= 4.3;

    if (t < 1.f)
    {
        SimParams.WindSpeed.x = 30.f + 4.f*sinf(cosf(t));
    }
    else if (t < 3.f)
    {
        SimParams.WindSpeed.x = -30.f + 4.f * sinf(cosf(t));
    }
    else if (t < 4.f)
    {
        SimParams.WindSpeed.x = 30.f + 4.f * sinf(cosf(t));
    }
    else if (t < 6.f)
    {
        SimParams.WindSpeed.x = -33.f + 4.f * sinf(cosf(t));
    }
    else if (t < 8.f)
    {
        SimParams.WindSpeed.x = 30.f + 4.f * sinf(cosf(t));
    }
    else if (t < 12.f) SimParams.WindSpeed.x = -15.f;
    //else SimParams.WindSpeed.x = 30.f * sinf(0.7f * t) * cosf(0.002f * t);
    //SimParams.WindSpeed.x = 20.f * sinf(5.f * cosf(0.08f * t));
   else SimParams.WindSpeed.x = 0.f;
}

void Solver::FreeBuffers()
{
    //TextSDF->destroy();
}

void Solver::InitParams()
{

    SimParams.RigidAngle = make_float3(0.f);
    SimParams.RigidPos = make_float3(0.f);

    SimParams.CutMin = 4;

    SimParams.AnimationType = SEQUENCE;
    SimParams.Wind = false;
    SimParams.CutRadius = 0.4f;

    // Basic parameters
    SimParams.NumVertices = 0;
    SimParams.NumTriangles = 0; 
    SimParams.NumParticles = 0; 
    SimParams.NumRoots = 0;
    SimParams.NumSprings = 0;

    // Physical parameters
    SimParams.Dt = 0.02f;//0.0023;
    SimParams.WindSpeed = make_float3(0.f);
    SimParams.Gravity = 26.18f;
    SimParams.AngularK = 100.f;
    SimParams.GravityK = 1000.f;//0.23;
    SimParams.EdgeK = 100000.f;
    SimParams.BendK = 100000.f;
    SimParams.TorsionK = 100000.f;//50.f;
    SimParams.Damping = 1.5;
    SimParams.HairMass = 1.f;
    SimParams.Friction = 5.f;//3.5f;

    // Eulerian simulation
    SimParams.LengthIncreaseGrid = 0.3f; // percentage of grid increase w.r.t. loaded obj
    SimParams.ThreshSdf = 0.15f;
    SimParams.NumGridNeighbors = 1;
    SimParams.NumStepsJacobi = 50;
    SimParams.JacobiWeight = 0.6f;
    SimParams.FlipWeight = 0.8f;//0.95f;

    // Solver paremters
    SimParams.StrainError = 0.1;  //0.2
    SimParams.StrainSteps = 4;  //10;
    SimParams.NestedSteps = 1;//10;

    // Animation
    SimParams.MeshSequence = false;
    SimParams.SimTime = 0.f;
    SimParams.AnimDuration = 27.f;
    SimParams.SimEndTime = -1.f;
    SimParams.NumFrames = 1;
    SimParams.RootMove = make_float3(0.f);

    // Geometry
    SimParams.NumBonesInfluence = MAX_BONE_NR;
    SimParams.NumBones = 0;
    SimParams.SelectedBone = 0;

    // Import/Export
    SimParams.SaveSelection = false;
    SimParams.LoadSelection = false;
    SimParams.ExportStep = 3;

    // Preprocessing
    SimParams.FixRootPositions = false;
    SimParams.TooglePreProcess = false;
    SimParams.RecomputeSdf = true;

    // Experiments manuscript
    SimParams.Experiment = NONE;
    SimParams.AnimationType = SEQUENCE;

    // Toogle Options
    SimParams.FixedHairLength = false;
    SimParams.Animate = false;
    SimParams.PauseSim = false;
    SimParams.ExportSequence = false;
    SimParams.ExportDataFormat = false;
    SimParams.ExportVDB = false;
    SimParams.ExportHair = false;
    SimParams.ExportMesh = false;
    SimParams.Export = false;

    // Toogle Debug
    SimParams.SolidCollision = true;
    SimParams.HairCollision = true;
    SimParams.DrawBoneWeight = false;
    SimParams.DrawBones = false;
    SimParams.DrawBonesAxis = false;
    SimParams.DrawHead = true;
    SimParams.DrawHeadWF = false;
    SimParams.DrawGrid = false;
}

int Solver::SelectCudaDevice()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device = deviceCount - 1;
    checkCudaErrors(cudaSetDevice(device));
    return device;
}

void Solver::UpdateAnimGridBox(const int& a, const int& b, const float& lambda)
{
    // Number of cells per dimension
    int nx = SimParams.HairDim.x;
    int ny = SimParams.HairDim.y;
    int nz = SimParams.HairDim.z;

    // Total size of grid embedding obj
    vector<vector<float3>>& boxes = Meshes[0]->GetAnimAABB();
    float3 minLambda = (1 - lambda) * boxes[a][0] + lambda * boxes[b][0];
    float3 maxLambda = (1 - lambda) * boxes[a][1] + lambda * boxes[b][1];
    
    vector<float3> minMax = { minLambda, maxLambda };
    float3 distances = minMax[1] - minMax[0];

    // Expands grid around obj
    float scale = SimParams.LengthIncreaseGrid;
    float3 dimIncrease = 0.5f * scale * distances;
    vector<float3> newMinMax = { minMax[0] - dimIncrease, minMax[1] + dimIncrease };

    // Stream info into solver
    float3 newDist = (newMinMax[1] - newMinMax[0]);
    float dx = newDist.x / (1.f * nx);
    float dy = newDist.y / (1.f * ny);
    float dz = newDist.z / (1.f * nz);

    // After bone transform (identity at t0)
    SimParams.HairCenter = 0.5 * (newMinMax[1] + newMinMax[0]);
    SimParams.HairAxis[0] = make_float3(1.f, 0.f, 0.f);
    SimParams.HairAxis[1] = make_float3(0.f, 1.f, 0.f);
    SimParams.HairAxis[2] = make_float3(0.f, 0.f, 1.f);
    SimParams.HairOrig = newMinMax[0];

    // Static params
    SimParams.MaxWeight = 1.5 * max(dx, max(dy, dz)) * (SimParams.NumGridNeighbors + 1.f);
    SimParams.HairInvSqDs = make_float3(1.f / (dx * dx), 1.f / (dy * dy), 1.f / (dz * dz));
    SimParams.HairInvDs = make_float3(1.f / dx, 1.f / dy, 1.f / dz);
    SimParams.HairDs = make_float3(dx, dy, dz);
}

void Solver::AnimTimeToFrame(int& a, int& b, float& lambda)
{
    float normalTime = fmin(SimParams.SimTime, SimParams.AnimDuration) / SimParams.AnimDuration;
    a = floor(normalTime * (SimParams.NumFrames - 1));
    b = min(a + 1, SimParams.NumFrames - 1);
    lambda = normalTime * (SimParams.NumFrames - 1) - a;
}

int Solver::DivUp(const int& a, const int& b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void Solver::MapOpenGL()
{
    // CUDA to OpenGL 

    // head
    launchKernelCudaToGLMesh(BlockThreadMesh, MainMappers[0]->Map(), DBuffers);
    MainMappers[0]->UnMap();

    // hair
    launchKernelCudaToGLHair(BlockThreadHair, MainMappers[1]->Map(), DBuffers);
    MainMappers[1]->UnMap();

    // interpolation
    launchKernelCudaToGLInter(BlockThreadInter, MainMappers[1]->Map(), DBuffers);
    MainMappers[1]->UnMap();
}
