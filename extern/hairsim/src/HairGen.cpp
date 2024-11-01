#include "HairGen.h"

#include "SFieldCPU.h"

HairGenerator::HairGenerator()
{
    InitParams();
}

HairGenerator::HairGenerator(vector<shared_ptr<SimpleObject>>& meshes)
{
    // Gets geometric info and starts params
    MainMesh = meshes[0];
    SdfCPU = make_shared<SFieldCPU>(meshes[0]);
    InitParams();
}

vector<vector<HairNode>> HairGenerator::InterpolateHairSecond(const vector<vector<HairNode>> &hair)
{

    // Find closest triangle to each strand
    vector<int> hairTri = vector<int>(hair.size());
    #pragma omp parallel for
    for(int i=0; i<hairTri.size(); i++)
    {
        PointSDF tempRes = SdfCPU->DistancePointField(hair[i][0].Pos);
        hairTri[i] = tempRes.TriIdx;
    }


    // Max number of added strands
    int numberInterp = GenParams.NumInterpolated;

    // New container
    vector<vector<HairNode>> newHair = vector<vector<HairNode>>(hair);

    // Temporal container to be able to parallelize over strands
    vector<vector<vector<HairNode>>> addedHair = vector<vector<vector<HairNode>>>(hair.size());

    #pragma omp parallel for
    for(int i=0; i<hair.size(); i++)
    {

        vector<HairNode> currentStrand = hair[i];

        // Find the closest n roots to this strand (inside same triangle)
        vector<int> roots = ClosestRootsTriangle(currentStrand, hair, hairTri, i, numberInterp);

        addedHair[i] = vector<vector<HairNode>>(roots.size());

        // Interpolate this strand n times
        #pragma omp parallel for
        for(int inter=0; inter<roots.size(); inter++)
        {

            // Choose one at random
            int root = roots[inter];

            // Push to new hair
            addedHair[i][inter] = InterpolateStrands(currentStrand, hair[root], i, root);
        }
    }


    // Add to new hair
    for(int i=0; i<addedHair.size(); i++)
    {
        for(int j=0; j<addedHair[i].size(); j++)
        {
            newHair.push_back(addedHair[i][j]);
            Strands.push_back(addedHair[i][j]);

        }
    }

    return newHair;
}

vector<vector<HairNode>> HairGenerator::InterpolateHairThird(const vector<vector<HairNode>> &hair)
{
    int numberInterp = GenParams.NumInterpolated;

    vector<vector<HairNode>> newHair = vector<vector<HairNode>>(hair);

    // Iterate over strands
    vector<vector<vector<HairNode>>> addedHair = vector<vector<vector<HairNode>>>(hair.size());

    #pragma omp parallel for
    for(int i=0; i<hair.size(); i++)
    {

        vector<HairNode> currentStrand = hair[i];
        addedHair[i] = vector<vector<HairNode>>(numberInterp);

        // Find the closest root to this strand
        int closest = theClosestRoot(currentStrand, hair, numberInterp);
        //vector<int> roots = ClosestRootsTriangle(currentStrand, hair, numberInterp);

        // Interpolate this strand n times
        for(int inter=0; inter<numberInterp; inter++)
        {

            // Push to new hair
            addedHair[i][inter] = InterpolateStrands(currentStrand, hair[closest], i, closest);
        }
    }


    // Add to new hair
    for(int i=0; i<addedHair.size(); i++)
    {
        for(int j=0; j<addedHair[i].size(); j++)
        {
            newHair.push_back(addedHair[i][j]);
            Strands.push_back(addedHair[i][j]);

        }
    }

    return newHair;
}

vector<vector<HairNode>> HairGenerator::InterpolateHair(const vector<vector<HairNode>> &hair)
{
    int numberInterp = GenParams.NumInterpolated;

    vector<vector<HairNode>> newHair = vector<vector<HairNode>>(hair);

    // Iterate over strands
    vector<vector<vector<HairNode>>> addedHair = vector<vector<vector<HairNode>>>(hair.size());

    #pragma omp parallel for
    for(int i=0; i<hair.size(); i++)
    {

        vector<HairNode> currentStrand = hair[i];
        addedHair[i] = vector<vector<HairNode>>(numberInterp);

        // Find the closest n roots to this strand
        vector<int> roots = ClosestRoots(currentStrand, hair, numberInterp);
        //vector<int> roots = ClosestRootsTriangle(currentStrand, hair, numberInterp);

        // Interpolate this strand n times
        for(int inter=0; inter<numberInterp; inter++)
        {

            // Choose one at random
            int root = roots[inter];

            // Push to new hair
            addedHair[i][inter] = InterpolateStrands(currentStrand, hair[root], i, root);
        }
    }


    // Add to new hair
    for(int i=0; i<addedHair.size(); i++)
    {
        for(int j=0; j<addedHair[i].size(); j++)
        {
            newHair.push_back(addedHair[i][j]);
            Strands.push_back(addedHair[i][j]);

        }
    }

    return newHair;
}

vector<int> HairGenerator::ClosestRoots(const vector<HairNode> &target, const vector<vector<HairNode>> &hair, int n)
{
    // Create a vector of pairs (distance, index)
    std::vector<std::pair<double, int>> distances(hair.size());

    // Calculate the distance from the target for each point in the list
    #pragma omp parallel for
    for (int i = 0; i < hair.size(); ++i) 
    {
        double dist = RootDistance(target, hair[i]);
        distances[i] = std::make_pair(dist, i);
    }

    // Sort the distances vector by distance
    std::sort(distances.begin(), distances.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
        return a.first < b.first;
    });

    // Extract the indices of the n closest strands
    std::vector<int> closestIndices;
    for (int i = 1; i <= n; i++) 
    {
        closestIndices.push_back(distances[i].second);
    }

    return closestIndices;
}

int HairGenerator::theClosestRoot(const vector<HairNode> &target, const vector<vector<HairNode>> &hair, int n)
{
    // Create a vector of pairs (distance, index)
    std::vector<std::pair<double, int>> distances(hair.size());

    // Calculate the distance from the target for each point in the list
    #pragma omp parallel for
    for (int i = 0; i < hair.size(); ++i) 
    {
        double dist = RootDistance(target, hair[i]);
        distances[i] = std::make_pair(dist, i);
    }

    // Sort the distances vector by distance
    std::sort(distances.begin(), distances.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
        return a.first < b.first;
    });

    return distances[1].second;
}

vector<int> HairGenerator::ClosestRootsTriangle(const vector<HairNode> &target, const vector<vector<HairNode>> &hair, const vector<int>& hairTri, int targetIdx, int n)
{

    // Find closest triangle to target
    int targetTri = hairTri[targetIdx];

    // Get roots in the same triangle
    vector<int> triangleRoots;
    for(int i=0; i<hair.size(); i++)
    {
        if(hairTri[i]==targetTri) triangleRoots.push_back(i);
    }

    // Create a vector of pairs (distance, index)
    std::vector<std::pair<double, int>> distances(triangleRoots.size());

    // Calculate the distance from the target for each point in the list
    #pragma omp parallel for
    for (int i = 0; i < triangleRoots.size(); ++i) 
    {
        double dist = RootDistance(target, hair[triangleRoots[i]]);
        distances[i] = std::make_pair(dist, triangleRoots[i]);
    }

    // Sort the distances vector by distance
    std::sort(distances.begin(), distances.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
        return a.first < b.first;
    });

    // Extract the indices of the n closest strands with similar direction
    std::vector<int> closestIndices;
    for(int i=1; i<triangleRoots.size(); i++)
    {
        if(i>=n) break;

        closestIndices.push_back(distances[i].second);
    }

    return closestIndices;
}

double HairGenerator::RootDistance(const vector<HairNode> &a, const vector<HairNode> &b)
{
    return length(a[0].Pos - b[0].Pos);
}

int HairGenerator::SampleRoot(const vector<int> &roots)
{
    // Random number generator setup
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<> distr(0, roots.size() - 1); // Define the range

    // Select a random index
    int randomIndex = distr(gen);

    // Access the element at the random index
    return roots[randomIndex];
}

vector<HairNode> HairGenerator::InterpolateStrands(const vector<HairNode> &a, const vector<HairNode> &b, const int& na, const int& nb)
{
    vector<HairNode> strand = vector<HairNode>(a.size());

    // Interpolation parameter
    float lambda = Rand(0.1, 0.9);

    // Interpolate all hair quantities
    for(int i=0; i<a.size(); i++)
    {
        strand[i].Pos = Lerp(a[i].Pos, b[i].Pos, lambda);
        strand[i].Ortho = Lerp(a[i].Ortho, b[i].Ortho, lambda);
        strand[i].Dir = Lerp(a[i].Dir, b[i].Dir, lambda);
        strand[i].Par = Lerp(a[i].Par, b[i].Par, lambda);
        strand[i].Thick = Lerp(a[i].Thick, b[i].Thick, lambda);

        strand[i].Interpolated = true;
        strand[i].InterpolIdx = make_int2(na, nb);
        strand[i].Lambda = lambda;
    }

    return strand;
}

vector<vector<HairNode>> HairGenerator::LoadHair(const string& fname, const float4& move)
{
    // Loads file and transform to framework strands
    HairLoader.load(fname);
    std::vector<std::vector<HairNode>> loaded = External2Strand(move);

    // Interpolate
    if(GenParams.Interpolate) loaded = InterpolateHairThird(loaded);

    return loaded;
}

// creates a "hair rectangle" for testing hair-hair siction
vector<vector<HairNode>> HairGenerator::CreateTestSiction(const int& numStrands, const float4& limits, const float& height)
{
    // Brush parameters
    float hairThick = 0.004f;
    float dz = 1.f;
    int numSegments = 20;

    // Iterate, assigning a random position within
    // the rectangle to each root
    vector<vector<HairNode>> tempStrands;
    for (int i = 0; i < numStrands; i++)
    {

        vector<HairNode> hair = vector<HairNode>(numSegments);
        float3 pos = make_float3(Rand(limits.x, limits.z), height, Rand(limits.y, limits.w));
        for (int j = 0; j < numSegments; j++)
        {
            hair[j] = HairNode(pos, make_float3(0.f), hairThick);
            pos += make_float3(0.f, -Rand(dz, 2 * dz), 0.f);
        }

        // Compute dir
        #pragma omp parallel for
        for (int j = 1; j < numSegments; j++)
        {
            hair[j].Dir = normalize(hair[j].Pos - hair[j - 1].Pos);
        }
        hair[0].Dir = hair[1].Dir;

        LoadComputePFTs(hair);
        tempStrands.push_back(hair);
        Strands.push_back(hair);
    }

    printf("Created hair brush with",tempStrands.size(), "strands\n");
    return tempStrands;
}

void HairGenerator::GenerateRoots()
{
    // Clears last roots
    Roots.clear();

    // Selects random positions for hair growth on each selected triangle
    for (int i = 0; i < MainMesh->GetNumTriangles(); i++)
    {
        SimpleTriangle& tri = *MainMesh->GetTriangles()[i];
        if (tri.Selected)
        {
            // Geometry information
            SimpleVertex& v0 = *MainMesh->GetVertices()[tri.V[0]];
            SimpleVertex& v1 = *MainMesh->GetVertices()[tri.V[1]];
            SimpleVertex& v2 = *MainMesh->GetVertices()[tri.V[2]];

            // Taking data
            float3 na = v0.Normal;
            float3 nb = v1.Normal;
            float3 nc = v2.Normal;

            for (int k = 0; k < GenParams.HairsPerTri; k++)
            {
                float ra = Rand(0.0f, 1.0f);
                float rb = Rand(0.0f, 1.0f);
                float rc = Rand(0.0f, 1.0f);

                float3 s = Lerp(v0.Pos, v1.Pos, ra);
                float3 t = Lerp(v0.Pos, v2.Pos, rb);
                float3 u = Lerp(s, t, rc);

                float3 sn = Lerp(na, nb, ra);
                float3 tn = Lerp(na, nc, rb);
                float3 un = Lerp(sn, tn, rc);

                HairRoot root;
                root.Pos = u;
                root.Normal = normalize(un);
                root.Steps = Rand(GenParams.StepsMin, GenParams.StepsMax);
                root.TriangleIdx = i;

                Roots.push_back(root);
            }

        }
    }
}

vector<vector<HairNode>> HairGenerator::GenerateStrands()
{
    // Temporal container
    vector<vector<HairNode>> tempStrands;

    // Prepare useful data
    vector<shared_ptr<SimpleTriangle>>& triangles = MainMesh->GetTriangles();
    vector<shared_ptr<SimpleVertex>>& vertices = MainMesh->GetVertices();
    float3 up = make_float3(0, 1, 0);

    // Iterate over roots
    for (int i = 0; i < Roots.size(); i++)
    {

        HairRoot& r = Roots[i];
        vector<HairNode> strand;

        // Initial strand data
        float3 noise_vec = make_float3(Rand(-1.0f, 1.0f), 0.0f, Rand(-1.0f, 1.0f));
        float3 prev_p = r.Pos;
        float3 inital_p = r.Pos;
        float3 initial_n = r.Normal + normalize(noise_vec) * GenParams.DirNoise;
        float3 prev_dir = initial_n;

        // Attached to this triangle
        SimpleTriangle& triRoot = *triangles[r.TriangleIdx];
        float3 ortho = normalize(vertices[triRoot.V[0]]->Pos - prev_p);

        // Creates hair extension
        HairNode node = HairNode(prev_p, prev_dir, GenParams.HairThickness);
        node.Ortho = ortho;
        strand.push_back(node);

        // Procedural iterations
        float3 grav = make_float3(0.f);
        for (int j = 0; j < r.Steps; ++j)
        {

            // New direction
            float3 dir = normalize(prev_dir + grav * max(GenParams.GravityDotInfluence, (1 - abs(dot(dir, up)))));

            // Spiral influence
            float sx = GenParams.SpiralRad * cos(j * GenParams.FreqMult);
            float sz = GenParams.SpiralRad * sin(j * GenParams.FreqMult);
            float3 spiral = make_float3(sx, GenParams.SpiralY, sz);
            dir = Lerp(dir, dir + spiral, GenParams.SpiralImpact);

            // Parting influence
            if (inital_p.x < 0) {
                dir = lerp(dir, dir + make_float3(-GenParams.PartingStrengthX, GenParams.PartingStrengthY, 0.0) + grav, GenParams.PartingImpact);
            }
            else {
                dir = lerp(dir, dir + make_float3(GenParams.PartingStrengthX, GenParams.PartingStrengthY, 0.0) + grav, GenParams.PartingImpact);
            }

            // Final position
            float3 p = prev_p + dir * GenParams.StepLength;
            HairNode new_node = HairNode(p, dir, GenParams.HairThickness);
            strand.push_back(new_node);

            // Update params
            prev_p = p;
            prev_dir = dir;
            grav += make_float3(Rand(-GenParams.GravNoise, GenParams.GravNoise), -0.981f * GenParams.GravityInfluence, Rand(-GenParams.GravNoise, GenParams.GravNoise));
        }

        ComputePFTs(strand);

        // Push into temporal and final containers
        tempStrands.push_back(strand);
        Strands.push_back(strand);
    };

    // Clear used roots
    Roots.clear();

    return tempStrands;
}

void HairGenerator::GenerateParticles()
{
    // First, uses temporal container to fill data
    // and flats afterwards. This is necessary to handle
    // strands of different sizes
    //vector<vector<Particle>> temp = vector<vector<Particle>>(Strands.size());
    //StrandsLength = vector<int>(Strands.size());

    vector<vector<Particle>> temp;

    for(int i=0; i<Strands.size(); i++)
    {
        if(!Strands[i][0].Interpolated)
        {
            StrandsLength.push_back(Strands[i].size());
            vector<Particle> newStrand = vector<Particle>(Strands[i].size());
            int strandSize = Strands[i].size();

            #pragma omp parallel for
            for(int j=0; j<strandSize; j++)
            {
                // Geometry data
                newStrand[j].LocalIdx = j;
                newStrand[j].StrandLength = strandSize;
                newStrand[j].VboIdx = Strands[i][j].VboIdx;

                // Dynamics data
                newStrand[j].InitialPosition = Strands[i][j].Pos;
                newStrand[j].Position0 = Strands[i][j].Pos;
                newStrand[j].Position = Strands[i][j].Pos;
                newStrand[j].Velocity = make_float3(0.f);
            }

            temp.push_back(newStrand);
        }
    }

    //#pragma omp parallel for
    //for (int i = 0; i < Strands.size(); i++)
    //{
    //    StrandsLength[i] = Strands[i].size();
    //    temp[i] = vector<Particle>(Strands[i].size());
    //    int strandSize = Strands[i].size();
    //    #pragma omp parallel for
    //    for (int j = 0; j < strandSize; j++)
    //    {
//
  //          // Geometry data
    //        temp[i][j].LocalIdx = j;
      //      temp[i][j].StrandLength = strandSize;
        //    temp[i][j].VboIdx = Strands[i][j].VboIdx;

            // Dynamics data
//            temp[i][j].InitialPosition = Strands[i][j].Pos;
  //          temp[i][j].Position0 = Strands[i][j].Pos;
    //        temp[i][j].Position = Strands[i][j].Pos;
      //      temp[i][j].Velocity = make_float3(0.f);
        //}
    //}

    // Sets springs
    for (int i = 0; i < temp.size(); i++)
    {
        vector<Particle>& strand = temp[i];
        for (int j = 0; j < strand.size(); j++)
        {
            // edge
            if (j + 1 < strand.size())
            {
                float l0 = length(strand[j + 1].Position - strand[j].Position);
                RestLengths.push_back(l0);
                strand[j].EdgeRestIdx.y = RestLengths.size() - 1;
                strand[j + 1].EdgeRestIdx.x = RestLengths.size() - 1;
            }

            // bending
            if (j + 2 < strand.size())
            {
                float l0 = length(strand[j + 2].Position - strand[j].Position);
                RestLengths.push_back(l0);
                strand[j].BendRestIdx.y = RestLengths.size() - 1;
                strand[j + 2].BendRestIdx.x = RestLengths.size() - 1;
            }

            // torsion
            if (j + 3 < strand.size())
            {
                float l0 = length(strand[j + 3].Position - strand[j].Position);
                RestLengths.push_back(l0);
                strand[j].TorsRestIdx.y = RestLengths.size() - 1;
                strand[j + 3].TorsRestIdx.x = RestLengths.size() - 1;
            }
        }
    }

    // Sets flat indices
    int counter = 0;
    for (int i = 0; i < temp.size(); i++)
    {
        for (int j = 0; j < temp[i].size(); j++)
        {
            temp[i][j].GlobalIdx = counter;
            if (j == 0) RootIdx.push_back(counter);
            counter++;
        }
    }

    // Interpolation
    for(int i=0; i<Strands.size(); i++)
    {
        if(Strands[i][0].Interpolated)
        {
            for(int j=0; j<Strands[i].size(); j++)
            {
                InterParticle particle;
                int2 idx = Strands[i][j].InterpolIdx;
                particle.InterIdx = make_int2(temp[idx.x][j].GlobalIdx, temp[idx.y][j].GlobalIdx);
                particle.VboIdx = Strands[i][j].VboIdx;
                particle.Lambda = Strands[i][j].Lambda;

                mInterParticles.push_back(particle);
            }
        }
    }

    // Flattens into vector
    for (auto& v : temp)
    {
        Particles.insert(Particles.end(), v.begin(), v.end());
    }

    // Max strand size
    MaxStrandLength = *max_element(StrandsLength.begin(), StrandsLength.end());

}

void HairGenerator::Clear()
{
    Strands.clear();
    Roots.clear();
    Particles.clear();
    RestLengths.clear();
    RootIdx.clear();
    mInterParticles.clear();
    StrandsLength.clear();
    MaxStrandLength = 0;
}

vector<vector<HairNode>> HairGenerator::External2Strand(const float4& move)
{

    // Temporal container
    vector<vector<HairNode>> tempStrands;

    float scaling = move.w;
    float hairThick = 0.004f;//0.001f;

    // Checks max hair size
    int stepStrands = HairLoader.strands.size() <= GenParams.MaxLoadNumStrands ? 1 : HairLoader.strands.size() / GenParams.MaxLoadNumStrands;
    
    for (int i = 0; i < HairLoader.strands.size(); i += stepStrands)
    {
        if (HairLoader.strands[i].size() <= GenParams.MaxLoadStrandSize)
        {
            vector<HairNode> hair = vector<HairNode>(HairLoader.strands[i].size());
            #pragma omp parallel for
            for (int j = 0; j < HairLoader.strands[i].size(); j++)
            {
                hair[j] = HairNode(scaling * Eigen2float(HairLoader.strands[i][j]) + make_float3(move), make_float3(0.f), hairThick);
            }

            // Compute dir
            #pragma omp parallel for
            for (int j = 1; j < HairLoader.strands[i].size(); ++j)
            {
                hair[j].Dir = normalize(hair[j].Pos - hair[j - 1].Pos);
            }
            hair[0].Dir = hair[1].Dir;

            LoadComputePFTs(hair);
            tempStrands.push_back(hair);
            Strands.push_back(hair);
        }
        else
        {
            int step = HairLoader.strands[i].size() / GenParams.MaxLoadStrandSize;
            vector<HairNode> hair;
            for (int counter = 0; counter < HairLoader.strands[i].size(); counter += step)
            {
                HairNode node = HairNode(scaling * Eigen2float(HairLoader.strands[i][counter]) + make_float3(move), make_float3(0.f), hairThick);
                hair.push_back(node);
            }

            // Compute dir
            #pragma omp parallel for
            for (int j = 1; j < hair.size(); ++j)
            {
                hair[j].Dir = normalize(hair[j].Pos - hair[j - 1].Pos);
            }
            hair[0].Dir = hair[1].Dir;

            LoadComputePFTs(hair);
            tempStrands.push_back(hair);
            Strands.push_back(hair);
        }
    }

    printf("Loaded hair with %i strands\n", tempStrands.size());
    return tempStrands;
}

void HairGenerator::InitParams()
{

    // GenParams.NumInterpolated = 10;
    GenParams.NumInterpolated = 1;
    GenParams.Interpolate = true;

    // Procedural Params
    GenParams.StepLength = 0.026f;
    GenParams.StepsMin = 20;
    GenParams.StepsMax = 20;
    GenParams.HairsPerTri = 5;

    GenParams.HairThickness = 0.001f;
    GenParams.GravityInfluence = 0.01f;
    GenParams.DirNoise = 0.0f;
    GenParams.GravNoise = 0.001f;
    GenParams.GravityDotInfluence = 0.2f;

    GenParams.SpiralRad = 1.f;
    GenParams.FreqMult = 1.f;
    GenParams.SpiralAmount = 1.f;
    GenParams.SpiralY = 1.f;
    GenParams.SpiralImpact = 0.f;

    GenParams.PartingImpact = 0.f;
    GenParams.PartingStrengthX = 0.f;
    GenParams.PartingStrengthY = 0.f;

    // Interaction Params
    GenParams.SelectionRadius = 0.3f;
    //GenParams.HairColor[0] = 250.f; {0.82f, 0.42f, 0.24f, 1.f};

    // Loader Params
    GenParams.MaxLoadStrandSize = 20; //20
    GenParams.MaxLoadNumStrands = 10000;
}

void HairGenerator::ComputePFTs(vector<HairNode>& strand)
{
    MATH_CONSTANTS
    float na = 0.01f;
    for (int j = 0; j < strand.size() - 1; ++j)
    {
        HairNode& s = strand[j];
        HairNode& t = strand[j + 1];

        float3 s_dir = s.Dir;
        float3 t_dir = t.Dir + make_float3(Rand(-na, na), Rand(-na, na), Rand(-na, na));

        float angles = Math_degrees * Angle(s_dir, t_dir);

        float3 w = normalize(cross(s_dir, t_dir));
        Mat4 R = Mat4::Rotate(-angles, w);

        t.Ortho = R * s.Ortho;
        t.Par = normalize(cross(t.Dir, t.Ortho));
    }
}

void HairGenerator::LoadComputePFTs(vector<HairNode>& strand)
{
    MATH_CONSTANTS
    float na = 0.01f;
    strand[0].Ortho = make_float3(0.f, 1.f, 0.f);

    for (int j = 0; j < strand.size() - 1; ++j)
    {
        HairNode& s = strand[j];
        HairNode& t = strand[j + 1];

        float3 s_dir = s.Dir;
        float3 t_dir = t.Dir + make_float3(Rand(-na, na), Rand(-na, na), Rand(-na, na));

        float angle = Math_degrees * Angle(s_dir, t_dir);

        float3 w = normalize(cross(s_dir, t_dir));
        t.Ortho = Mat4::Rotate(-angle, w) * s.Ortho;
        t.Par = normalize(cross(t.Dir, t.Ortho));
    }
}
