#include <stdlib.h>
#include <iostream>
#include <Rendering/Engine.h>

// Helpers for setting different simulations
void SetFramework(const ExpOpt& opt, HairInfo& hair, HeadInfo& head);
void SetSandBox(HairInfo& hair, HeadInfo& head);

int main()
{

    // Select mode (experiments/sandbox)
    ExpOpt opt = SANDBOX;

    // Prepare framework
    Engine* engine = new Engine();
    HairInfo hair;
    HeadInfo head;

    // Set specific scene
    SetFramework(opt, hair, head);

    // Run framework
    engine->LoadScene(hair, head, opt);
    engine->Run();

    return 0;

}

void SetSandBox(HairInfo& hair, HeadInfo& head)
{
    // Loads Hair 
    //string hairTest = "../data/hair/strands00001.data";
    //float4 hairMove = make_float4(0.f, 0.f, 0.f, 1.f);
    //map<string, float4> hairData;
    //hairData[hairTest] = hairMove;


    int headN = 64;
    int hairN = 64;

    // Set head
    //head.FileName = "../data/animations/headmo2/head_1.obj";
    head.FileName = "../data/meshes/head.obj";
    // head.FileName = "../data/meshes/alejandro-aligned.obj";
    //head.FileName = "../data/meshes/body_scaled.obj";
    head.Move = make_float4(0.f, 0.f, 0.f, 1.f);
    head.SdfDim = make_int3(headN, 2 * headN, headN);
    head.Animate = false;
    head.NumFrames = 1;

    // Loads hair
    //hair.FileName = "../data/hair/harolongo.data";
    hair.FileName = "../data/hair/Jewfro.data";
    //hair.FileName = "../tmp/copilot/tmp2.data";
    //hair.FileName = "../data/pinscreenhair/Double_tail_curly_V08.data";
    hair.Move = make_float4(0.f, 0.f, 0.f, 1.f);
    hair.EulerDim = make_int3(hairN);
    hair.ID = 0;
}

void SetFramework(const ExpOpt& opt, HairInfo& hair, HeadInfo& head)
{
    switch (opt)
    {
        case ANIMATION:
        {

            int headN = 64;
            int hairN = 64;

            // Set head
            head.FileName = "../data/animations/bbj/bbj_1.obj";
            //head.FileName = "../data/pinscreenhair/body_scaled.dae";
            head.Move = make_float4(0.f, 0.f, -6.f, -18.f);
            head.SdfDim = make_int3(headN, 2 * headN, headN);
            head.Animate = true;
            head.NumFrames = 200;

            // Loads hair
            hair.FileName = "../data/pinscreenhair/Double_tail_curly_V08.data";
            hair.Move = make_float4(0.f, 0.f, 0.f, 1.f);
            hair.EulerDim = make_int3(hairN);
            hair.ID = 0;

            break;
        }
        case CANTI_E:
        {
            // Dummy head
            int resGrid = 4;
            string headUSC = "../data/objs/head_rigged.dae";
            float4 headUSCPos = make_float4(0.f, -30.f, 0.f, 20.f);
            int3 headUSCSize = make_int3(resGrid, 2 * resGrid, resGrid);
            vector<string> names = { headUSC };
            vector<float4> pos = { headUSCPos };
            vector<int3> sizes = { headUSCSize };

            // Empty Hair
            map<string, float4> hairData;

            //engine->LoadScene(names, sizes, pos, hairData, opt);
            break;
        }
        case CANTI_L:
        {
            break;
        }
        case SAG_WIND:
        {
            // Set head/mesh
            int resGrid = 128;
            string headName = "../data/objs/head_full.dae";
            float4 headPos = make_float4(0.f, 0.f, 0.f, 1.f);
            int3 headSize = make_int3(resGrid, 2 * resGrid, resGrid);
            vector<string> names = { headName };
            vector<float4> pos = { headPos };
            vector<int3> sizes = { headSize };

            // Loads hair
            string hairTest = "../data/objs/fixed_strands00001.data";
            float4 hairMove = make_float4(0.f, -29.9f, 0.f, 20.f);
            float4 noMove = make_float4(0.f, 0.f, 0.f, 1.f);
            map<string, float4> hairData;
            hairData[hairTest] = noMove;

            //engine->LoadScene(names, sizes, pos, hairData, opt);
            break;
        }
        case SAG_HEAD:
        {
            // Set head/mesh
            int resGrid = 256;
            string headName = "../data/objs/head_full.dae";
            float4 headPos = make_float4(0.f, 0.f, 0.f, 1.f);
            int3 headSize = make_int3(resGrid, 2 * resGrid, resGrid);
            vector<string> names = { headName };
            vector<float4> pos = { headPos };
            vector<int3> sizes = { headSize };

            // Loads hair
            string hairTest = "../data/input/fixed_test_chengan.data";
            float4 hairMove = make_float4(0.f, -29.9f, 0.f, 20.f);
            float4 noMove = make_float4(0.f, 0.f, 0.f, 1.f);
            map<string, float4> hairData;
            hairData[hairTest] = noMove;

            //engine->LoadScene(names, sizes, pos, hairData, opt);
            break;
        }
        case WIG:
        {
            // Set head/mesh
            int resGrid = 128;
            string headName = "../data/objs/wig_head.obj";
            float4 headPos = make_float4(0.f, 0.f, 0.f, 1.f);
            int3 headSize = make_int3(resGrid, 2 * resGrid, resGrid);
            vector<string> names = { headName };
            vector<float4> pos = { headPos };
            vector<int3> sizes = { headSize };

            // Loads hair
            string hairTest = "../data/hair/wig_alice.data";
            float4 hairMove = make_float4(0.f, 0.f, 0.f, 0.05f);
            float4 noMove = make_float4(0.f, 0.f, 0.f, 1.f);
            map<string, float4> hairData;
            hairData[hairTest] = noMove;

            //engine->LoadScene(names, sizes, pos, hairData, opt);
            break;
        }
        case SALON:
        {
            // Set head/mesh
            int resGrid = 128;
            string headName = "../data/objs/head_full.dae";
            float4 headPos = make_float4(0.f, 0.f, 0.f, 1.f);
            int3 headSize = make_int3(resGrid, 2 * resGrid, resGrid);
            vector<string> names = { headName };
            vector<float4> pos = { headPos };
            vector<int3> sizes = { headSize };

            // Loads hair
            string hairTest = "../data/extra/seed0071_ftl.data";
            float4 hairMove = make_float4(0.f, 0.f, 0.f, 0.05f);
            float4 noMove = make_float4(0.f, 0.f, 0.f, 1.f);
            map<string, float4> hairData;
            hairData[hairTest] = noMove;

            //engine->LoadScene(names, sizes, pos, hairData, opt);
            break;
        }
        case CHENGAN:
        {
            // Set head/mesh
            int resGrid = 128;
            string headName = "../data/objs/head_full.dae";
            float4 headPos = make_float4(0.f, 0.f, 0.f, 1.f);
            int3 headSize = make_int3(resGrid, 2 * resGrid, resGrid);
            vector<string> names = { headName };
            vector<float4> pos = { headPos };
            vector<int3> sizes = { headSize };

            // Loads hair
            string hairTest = "../data/extra/seed0078_ftl.data";
            float4 hairMove = make_float4(0.f, 0.f, 0.f, 0.05f);
            float4 noMove = make_float4(0.f, 0.f, 0.f, 1.f);
            map<string, float4> hairData;
            hairData[hairTest] = noMove;

            //engine->LoadScene(names, sizes, pos, hairData, opt);
            break;
        }
        case ROLLER:
        {
            // Set head/mesh
            int resGrid = 128;
            string headName = "../data/objs/head_full.dae";
            float4 headPos = make_float4(0.f, 0.f, 0.f, 1.f);
            int3 headSize = make_int3(resGrid, 2 * resGrid, resGrid);
            vector<string> names = { headName };
            vector<float4> pos = { headPos };
            vector<int3> sizes = { headSize };

            // Loads hair
            string hairTest = "../data/extra/seed0020_ftl.data";
            float4 hairMove = make_float4(0.f, 0.f, 0.f, 0.05f);
            float4 noMove = make_float4(0.f, 0.f, 0.f, 1.f);
            map<string, float4> hairData;
            hairData[hairTest] = noMove;

            //engine->LoadScene(names, sizes, pos, hairData, opt);
            break;
        }
        case CURLY:
        {
            // Set head/mesh
            int resGrid = 128;
            string headName = "../data/objs/head_full.dae";
            float4 headPos = make_float4(0.f, 0.f, 0.f, 1.f);
            int3 headSize = make_int3(resGrid, 2 * resGrid, resGrid);
            vector<string> names = { headName };
            vector<float4> pos = { headPos };
            vector<int3> sizes = { headSize };

            // Loads hair
            string hairTest = "../data/extra/curly.data";
            float4 hairMove = make_float4(0.f, 0.f, 0.f, 0.05f);
            float4 noMove = make_float4(0.f, 0.f, 0.f, 1.f);
            map<string, float4> hairData;
            hairData[hairTest] = noMove;

            //engine->LoadScene(names, sizes, pos, hairData, opt);
            break;
        }
        case KATE:
        {
            // Set head/mesh
            int resGrid = 32;
            string headName = "../data/objs/head_full.dae";
            float4 headPos = make_float4(0.f, 0.f, 0.f, 1.f);
            int3 headSize = make_int3(resGrid, 2 * resGrid, resGrid);
            vector<string> names = { headName };
            vector<float4> pos = { headPos };
            vector<int3> sizes = { headSize };

            // Loads hair
            string hairTest = "../data/extra/kate.data";
            float4 hairMove = make_float4(0.f, 0.f, 0.f, 0.05f);
            float4 noMove = make_float4(0.f, 0.f, 0.f, 1.f);
            map<string, float4> hairData;
            hairData[hairTest] = noMove;

            //engine->LoadScene(names, sizes, pos, hairData, opt);
            break;
        }
        case SPACE:
        {
            // Set head/mesh
            int resGrid = 32;
            string headName = "../data/objs/head_full.dae";
            float4 headPos = make_float4(0.f, 0.f, 0.f, 1.f);
            int3 headSize = make_int3(resGrid, 2 * resGrid, resGrid);
            vector<string> names = { headName };
            vector<float4> pos = { headPos };
            vector<int3> sizes = { headSize };

            // Loads hair
            string hairTest = "../data/extra/kate.data";
            float4 hairMove = make_float4(0.f, 0.f, 0.f, 0.05f);
            float4 noMove = make_float4(0.f, 0.f, 0.f, 1.f);
            map<string, float4> hairData;
            //hairData[hairTest] = noMove;

            //engine->LoadScene(names, sizes, pos, hairData, opt);
            break;
        }
        case MODERN:
        {
            // Set head/mesh
            int resGrid = 256;
            string headName = "../data/objs/head_full.dae";
            float4 headPos = make_float4(0.f, 0.f, 0.f, 1.f);
            int3 headSize = make_int3(resGrid, 2 * resGrid, resGrid);
            vector<string> names = { headName };
            vector<float4> pos = { headPos };
            vector<int3> sizes = { headSize };

            // Loads hair
            string hairTest = "../data/extra/modern_long.data";
            float4 hairMove = make_float4(0.f, 0.f, 0.f, 0.05f);
            float4 noMove = make_float4(0.f, 0.f, 0.f, 1.f);
            map<string, float4> hairData;
            hairData[hairTest] = noMove;

            //engine->LoadScene(names, sizes, pos, hairData, opt);
            break;
        }
        case FACIAL:
        {
            // Set head/mesh
            int resGrid = 128;
            string headName = "../data/objs/head_full.dae";
            float4 headPos = make_float4(0.f, 0.f, 0.f, 1.f);
            int3 headSize = make_int3(resGrid, 2 * resGrid, resGrid);
            vector<string> names = { headName };
            vector<float4> pos = { headPos };
            vector<int3> sizes = { headSize };

            // Loads hair
            string hairTest = "../data/hair/barba_corta.data";
            float4 hairMove = make_float4(0.f, 0.f, 0.f, 0.05f);
            float4 noMove = make_float4(0.f, 0.f, 0.f, 1.f);
            map<string, float4> hairData;
            hairData[hairTest] = noMove;

            //engine->LoadScene(names, sizes, pos, hairData, opt);
            break;
        }
        case SICTION:
        {
            // Solid mesh file names
            string headFull = "../data/meshes/head_full.dae";
            string bunny = "../data/meshes/bunny_simple.obj";

            // Translate and scale (last entry of float4) the object
            float4 noMove = make_float4(0.f, 0.f, 0.f, 1.f);
            float4 bunnyPos = make_float4(10.f, 3.f, 0.f, 10.f);

            // Grid size for SDF/Hybrid volumes
            int resGrid = 128;
            int3 headSize = make_int3(resGrid);

            // Packs everything (multiple objs possible)
            vector<string> names = { headFull };
            vector<float4> pos = { noMove };
            vector<int3> sizes = { headSize };

            // Loads Hair 
            string hairTest = "../data/hair/strands00001.data";
            float4 hairMove = make_float4(0.f, 0.f, 0.f, 1.f);
            map<string, float4> hairData;
            //hairData[hairTest] = hairMove;

            // Runs framework
            //engine->LoadScene(names, sizes, pos, hairData, opt);
            break;
        }
        case SANDBOX:
        {
            SetSandBox(hair, head);
            break;
        }
        case NONE:
        {
            printf("No simulation selected\n");
            break;
        }
        default:
            break;
    }
}