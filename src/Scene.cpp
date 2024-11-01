#include "Scene.h"
#include "Rendering/NiceGrid.h"
#include "Rendering/Light.h"
#include "Rendering/Shader.h"
#include "Rendering/VertexBufferObject.h"
#include "Rendering/Mesh.h"
#include "Rendering/CameraManager.h"
#include "Rendering/Object.h"
#include "Rendering/Geometry.h"

#include <boost/filesystem.hpp> // Include Boost.Filesystem

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// #include <ppl.h>

Scene::Scene(CameraManager *camManager, const HairInfo &hairInfo, const HeadInfo &headInfo, ExpOpt opt)
    : m_cameraManager(camManager),
      VboGridBox(nullptr),
      HairData(hairInfo),
      HeadData(headInfo)
{
    // Creates export folder
    SetupExportFolder();

    // Loads head
    InitHead(headInfo, true);

    // Embedded features
    mPerm = Perm();
    mCopilot = Copilot();
    mAIRender = AIRender();

    // Init general parameters, growth module and solver
    SimSolver = make_shared<Solver>(hairInfo, headInfo);
    HairGen = make_shared<HairGenerator>(Meshes);
    SimSolver->GetParamsSim().NumBones = Meshes[0]->GetNumBones();
    SimSolver->GetParamsSim().Parity = headInfo.Move.w < 0 ? -1.f : 1.f;
    SimSolver->HeadInformation = headInfo;

    // Init specific experiment parameters
    SetupExperiment(opt);

    // Loads hair
    if (HairData.ID != -1)
    {
        std::vector<std::vector<HairNode>> strands = HairGen->LoadHair(HairData.FileName, HairData.Move);
        BuildVboHair(strands);
    }

}

Scene::~Scene()
{
}

void Scene::InitHead(const HeadInfo &headInfo, const bool &objSeq)
{
    // Light and Grid
    m_lights.push_back(new Light(this, Light::SPOT_LIGHT, vec3(0.9f), vec3(20.0f, 20.0f, 20.0f), vec3(), vec3(1.2f), vec3(), vec3(0.7f, 0.001f, 0.0001f)));
    m_lights.push_back(new Light(this, Light::DIRECTIONAL_LIGHT, vec3(0.9f), vec3(0.f, 30.f, 100.f), vec3(0.f,-2.f,-1.f), vec3(1.2f), vec3(), vec3(0.7f, 0.001f, 0.0001f)));
    m_niceGrid = new NiceGrid(2000.0f, 40.0f);

    // Load Head
    shared_ptr<SimpleObject> obj = make_shared<SimpleObject>(headInfo.FileName, headInfo.Move, headInfo.Animate, headInfo.NumFrames);
    Meshes.push_back(obj);

    // Load Bones (for Debugging)
    BoneMesh = make_shared<SimpleObject>("../data/meshes/bone.obj");
    BuildVboBone();

    // Build boxes and vbos
    BuildVboMeshes();

    for (int i = 0; i < Meshes.size(); i++)
    {
        if (VERBOSE)
            printf("Loaded mesh with %i triangles\n", int(Meshes[i]->GetNumTriangles()));
    }
}

void Scene::RenderWorld(const Transform &trans)
{
    m_niceGrid->render(trans);

    if (params::inst()->renderMisc)
    {
        for (int i = 0; i < m_lights.size(); ++i)
        {
            m_lights[i]->render(trans);
        }

        m_cameraManager->renderCameras(trans);
    }
}

void Scene::RenderObjects(const Transform &trans)
{
    glPointSize(2.0f);
    mat4 model = mat4::translate(0.0f, 0.0f, 0.0f);
    Shader *shader = shaders::inst()->standard;
    Params *p = params::inst();
    SimulationParams &simPam = SimSolver->GetParamsSim();

    if (simPam.DrawHead)
    {
        // Set WF mode
        if (simPam.DrawHeadWF)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        // Main Obj
        Material mate(vec3(0.3f), vec3(0.1f), vec3(0.1f), 200.0f, "");
        shader = shaders::inst()->head;
        shader->bind();
        shader->setLights(params::inst()->lights);
        shader->set3f("camPos", params::inst()->camPos);
        shader->setMatrices(trans, model, true, true, true, true, true, true);
        shader->seti("numLights", params::inst()->lights.size());
        shader->setf("shadowIntensity", params::inst()->shadowIntensity);
        shader->set3f("lightPosition", params::inst()->lights[0]->position());
        shader->setf("lightIntensity", params::inst()->lightIntensity);
        shader->seti("drawWeight", simPam.DrawBoneWeight);
        shader->seti("drawFrame", simPam.DrawHeadWF);
        shader->setMaterial(mate);
        for (auto &vbo : VboMeshes)
            vbo->render();
        shader->release();

        // Get back to fill mode
        if (simPam.DrawHeadWF)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // Draw Hair
    {
        shader = shaders::inst()->hair;
        shader->bind();
        shader->setLights(params::inst()->lights);
        shader->set3f("camPos", params::inst()->camPos);
        shader->set3f("strandColor", HairGen->GetParamsGen().HairColor);
        shader->seti("hairDebbug", HairGen->GetParamsGen().HairDebbug);
        shader->setMatrices(trans, model, true, true, true, true, true, true);
        shader->seti("numLights", params::inst()->lights.size());
        shader->setf("shadowIntensity", params::inst()->shadowIntensity);
        shader->setf("lightIntensity", params::inst()->lightIntensity);

        for (auto vbo : VboHair)
            vbo->render();
        shader->release();
    }

    // Draw SDF box obj
    if (simPam.DrawGrid && Generated)
    {
        // center
        float3 pointPos = SimSolver->GetParamsSim().HairCenter;
        glPointSize(35.0f);
        glEnableFixedFunction(trans);
        glBegin(GL_POINTS);
        glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
        glVertex3f(pointPos.x, pointPos.y, pointPos.z);
        glEnd();
        glDisableFixedFunction();

        // origin
        pointPos = SimSolver->GetParamsSim().HairOrig;
        glPointSize(45.0f);
        glEnableFixedFunction(trans);
        glBegin(GL_POINTS);
        glColor4f(1.0f, 0.0f, 1.0f, 1.0f);
        glVertex3f(pointPos.x, pointPos.y, pointPos.z);
        glEnd();
        glDisableFixedFunction();

        // axis
        vec4 r = vec4(1.f, 0.f, 0.f, 1.f);
        vec4 g = vec4(0.f, 1.f, 0.f, 1.f);
        vec4 b = vec4(0.f, 0.f, 1.f, 1.f);
        mat4 scale = mat4::scale(4.f * vec3(5.f, 2.f, 5.f));
        mat4 move = mat4::translate(float2vec(SimSolver->GetParamsSim().HairOrig));
        mat4 modelX = move * mat4::rotateZ(270) * scale;
        mat4 modelY = move * scale;
        mat4 modelZ = move * mat4::rotateX(90) * scale;

        shader = shaders::inst()->axis;
        shader->bind();
        // x
        shader->setMatrices(trans, modelX, true, true, true);
        shader->set4f("col", r);
        VboBone->render();
        // y
        shader->setMatrices(trans, modelY, true, true, true);
        shader->set4f("col", g);
        VboBone->render();
        // z
        shader->setMatrices(trans, modelZ, true, true, true);
        shader->set4f("col", b);
        VboBone->render();
        shader->release();

        // box
        shader = shaders::inst()->gridBox;
        shader->bind();
        shader->setMatrices(trans, model, true, true, true);
        VboGridBox->render();
        shader->release();
    }
}

void Scene::RenderObjectsDepth(const Transform &trans)
{
    mat4 model = mat4::translate(0.0f, 0.0f, 0.0f);

    // Hair Depth
    Shader *shader = shaders::inst()->hairDepth;
    shader->bind();
    shader->setMatrices(trans, model, true, true, true, true, true, true);
    for (auto vbo : VboHair)
    {
        vbo->render();
    }
    shader->release();

    // Head Depth
    if (SimSolver->GetParamsSim().DrawHead)
    {
        shader = shaders::inst()->MeshDepth;
        shader->bind();
        shader->setMatrices(trans, model, true, true, true, true, true, true);
        if (VboMeshes[0])
        {
            VboMeshes[0]->render();
        }

        shader->release();
    }
}

void Scene::EmbeddedGUI()
{


    CopilotParameters& paramsCopilot = mCopilot.Params();
    PermParameters& paramsPerm = mPerm.Params();

    ImGui::Begin("Gen Settings");
    if (ImGui::BeginTabBar("AdvTabBar", ImGuiTabBarFlags_None))
    {

        // Copilot GUI
        if (ImGui::BeginTabItem("Copilot"))
        {
            // Child to contain the chat history
            if (ImGui::BeginChild("ChatHistory", ImVec2(0, 650), true)) {

                if(messages.size()==0)
                {
                    AddMessage("Tony Sensei: Hi there!");
                    AddMessage("Tony Sensei: What are we thinking for your hair today?");
                    AddMessage("Tony Sensei: Any particular style or look you have in mind?");
                }

                for (const auto& message : messages) {
                    
                    if(message.text.find("Me: ") != std::string::npos)
                    {
                        // Set the text color
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(32/255.0f, 240/255.0f, 234/255.0f, 1.0f)); // Red color
                        ImGui::TextWrapped("%s", message.text.c_str());
                        // Restore the previous text color
                        ImGui::PopStyleColor();
                    }
                    else
                    {
                        ImGui::TextWrapped("%s", message.text.c_str());
                    }

                    
                }
                

                scrollToBottom = true;

                if(showImageSelection==true)
                {
                    
                    if (ImGui::ImageButton((void*)(intptr_t)image1, ImVec2(400, 400))) {
                        selectedImage = 1;  // User selected the first image
                        //showImageSelection = false;  // Hide the selection UI after choosing
                    }

                    ImGui::SameLine();
                    if (ImGui::ImageButton((void*)(intptr_t)image2, ImVec2(400, 400))) {
                        selectedImage = 2;  // User selected the second image
                        //showImageSelection = false;
                    }

                    ImGui::SameLine();
                    if (ImGui::ImageButton((void*)(intptr_t)image3, ImVec2(400,400))) {
                        selectedImage = 3;  // User selected the third image
                        //showImageSelection = false;
                    }

                    scrollToBottom = true;
                }

                // Automatically scroll to the bottom if we are near the bottom
                if (scrollToBottom || ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
                    ImGui::SetScrollHereY(1.0f);
                }

                
            }
            ImGui::EndChild();

            
            if(paramsCopilot.PromptSent==true)
            {
                mCopilot.Generate(paramsCopilot.Prompt);
                //clear the previous prompts first
                memset(mAIRender.prompt_hairstyle, 0, 1024);
                paramsCopilot.Prompt.copy(mAIRender.prompt_hairstyle, paramsCopilot.Prompt.size(), 0);

                //receive three candidate images.
                if(boost::filesystem::exists(mCopilot.img1_path.data()))
                    image1 = LoadImage(mCopilot.img1_path.data());
                else
                    image1 = LoadImage(mCopilot.img1_path_batckup.data());

                if(boost::filesystem::exists(mCopilot.img2_path.data()))
                    image2 = LoadImage(mCopilot.img2_path.data());
                else
                    image2 = LoadImage(mCopilot.img2_path_batckup.data());

                if(boost::filesystem::exists(mCopilot.img3_path.data()))
                    image3 = LoadImage(mCopilot.img3_path.data());
                else
                    image3 = LoadImage(mCopilot.img3_path_batckup.data());

                showImageSelection=true;
                
                paramsCopilot.PromptSent=false;
            }

            if(selectedImage!=-1 && showImageSelection)
            {
                std::string data_path;
                if(selectedImage==1)
                    data_path=mCopilot.data1_path;
                else if(selectedImage==2)
                    data_path=mCopilot.data2_path;
                else
                    data_path=mCopilot.data3_path;

                if(boost::filesystem::exists(data_path))
                {
                    HairData.FileName=data_path;
                }
                else
                {
                    HairData.FileName=mCopilot.data_path_backup;
                }
                selectedImage=-1;
                ResetAll(HairData.FileName);
            }

            // Input text box for sending a new message
            static char inputText[256] = "";
            ImGui::PushItemWidth(params::inst()->window_x-300); // 300 pixels wide

            ImVec2 textBoxSize = ImVec2(params::inst()->window_x, ImGui::GetTextLineHeight() * 2 + ImGui::GetStyle().ItemSpacing.y);

            // Create a multi-line text input box
            //ImGui::InputTextMultiline("##InputTextMultiline", inputBuf, IM_ARRAYSIZE(inputBuf), textBoxSize);

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(32/255.0f, 240/255.0f, 234/255.0f, 1.0f)); // Red color
            //if (ImGui::nputTextMultiline("##ChatInput", inputBuf, IM_ARRAYSIZE(inputBuf), ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (ImGui::InputTextMultiline("##ChatInput", inputText, IM_ARRAYSIZE(inputText), textBoxSize, ImGuiInputTextFlags_EnterReturnsTrue)) {
                if (strlen(inputText) > 0) {
                    std::string message(inputText);
                    message="Me: "+message;
                    AddMessage(message);  // Add the input text as a new message
                    strcpy(inputText, "");  // Clear the input text box
                    scrollToBottom = true;  // Scroll to the bottom after sending a message
                }
            }
            ImGui::PopItemWidth();
            ImGui::PopStyleColor();

            //ImGui::SameLine();
            if (ImGui::Button("Send")) {
                if (strlen(inputText) > 0) {
                    std::string message(inputText);
                    
                    if((message=="create some wind?") || (message=="add some wind?")||(message=="blow some wind?"))
                    {
                        std::cout<<"set wind speed.\n";
                        SimulationParams &simParams = SimSolver->GetParamsSim();
                        simParams.GravityK=100;
                        
                        simParams.WindSpeed.x = 100 + rand()%50;
                        simParams.WindSpeed.y = 10 + rand()%10;
                        simParams.WindSpeed.z = -100 - rand()&50;

                        // Start simulation : s
                        if (is_simulation_init==false && !Generated)
                        {
                            std::cout<<"Start simulation.\n";
                            InitSimulation();
                            is_simulation_init = true;
                        }

                        AddMessage("Me: "+message);
                        strcpy(inputText, "");
                        scrollToBottom = true;
                        
                        std::string tony="Tony Sensei: Sure! Let's see how your hair moves with a little breeze.";
                        AddMessage(tony);
                        showImageSelection=false;
                    }
                    else if((message=="create strong wind?")||(message=="add strong wind?"))
                    {
                        std::cout<<"set wind speed.\n";
                        SimulationParams &simParams = SimSolver->GetParamsSim();
                        simParams.GravityK=1;
                        simParams.WindSpeed.x = 70+ rand()%10;
                        simParams.WindSpeed.y = 30+ rand()%10;
                        simParams.WindSpeed.z = -10 - rand()%5;

                        // Start simulation : s
                        if (is_simulation_init==false && !Generated)
                        {
                            std::cout<<"Start simulation.\n";
                            InitSimulation();
                            is_simulation_init = true;
                        }

                        AddMessage("Me: "+message);
                        strcpy(inputText, "");
                        scrollToBottom = true;
                        
                        std::string tony="Tony Sensei: How do you like it!";
                        AddMessage(tony);
                        showImageSelection=false;
                    }
                    else
                    {
                        std::cout<<"change hairstyle.\n";
                        paramsCopilot.Prompt = message;
                        paramsCopilot.PromptSent = true;

                        AddMessage("Me: "+message);
                        strcpy(inputText, "");
                        scrollToBottom = true;
                        
                        
                        int choice = rand() % 3;
                        if(choice ==0 )
                        {
                            AddMessage("Tony Sensei: I totally get what you're going for!");
                            AddMessage("Tony Sensei: I have a few ideas in mind that would look amazing. Let me share three options, and you can pick the one that feels right.");
                            AddMessage("Tony Sensei: Please click the one you prefer, and I will make the look!");
                        }

                        if(choice ==1 )
                        {
                            AddMessage("Tony Sensei: Sure thing! For the three looks I shared, which one do you prefer?");
                        }

                        if(choice ==2 )
                        {
                            AddMessage("Tony Sensei: Sounds great! I have three amazing ideas. Tell me which one you like?");
                        }


                        
                    }

                    
                }
            }

            


            ImGui::EndTabItem();

        }
        
        ////////////////////////////
        // AI Renderer GUI
        if (ImGui::BeginTabItem("AI Renderer"))
        {
            ImGui::PushItemWidth(params::inst()->window_x-300); // 300 pixels wide
            ImGui::InputText("gender", mAIRender.prompt_gender, IM_ARRAYSIZE(mAIRender.prompt_gender));

            ImGui::InputText("hairstyle", mAIRender.prompt_hairstyle, IM_ARRAYSIZE(mAIRender.prompt_hairstyle));
            
            ImGui::InputText("head pose", mAIRender.prompt_headpose, IM_ARRAYSIZE(mAIRender.prompt_headpose));

            ImGui::InputText("misc", mAIRender.prompt_misc, IM_ARRAYSIZE(mAIRender.prompt_misc));

            ImGui::PopItemWidth();
            // Button to load the image
            if (ImGui::Button("generate")) {
                int windowWidth=params::inst()->window_x;
                int windowHeight=params::inst()->window_y;
                //glfwGetWindowSize(window, &windowWidth, &windowHeight);
                SaveOpenGLImage(mAIRender.input_path.data(), windowWidth, windowHeight );
                mAIRender.Generate();
                mAIRender.imgReceived=true;
                showImageWindow = true;
                generated_image = LoadImage(mAIRender.output_path.data());
            }

            ImGui::EndTabItem();
        }

        


        // Perm GUI
        if (ImGui::BeginTabItem("Perm"))
        {
            ImGui::SliderFloat("Theta", &paramsPerm.Theta, 0.f, 10.f);


            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();

    // Show the image in a new window
    if (showImageWindow && mAIRender.imgReceived) {
        ImGui::Begin("Image Viewer", &showImageWindow);
        ImGui::Image((void*)(intptr_t)generated_image, ImVec2(512,512));

        ImGui::End();

    }
}

void Scene::UpdateEmbedded()
{
    mPerm.Apply();
    //mCopilot.Generate();
}

void Scene::ResetAll(std::string hairPath)
{
    // Cleans
    //SimSolver->Clear();
    HairGen->Clear();
    VboHair.clear();
    Generated = false;

    // Loads hair
    if (HairData.ID != -1)
    {
        std::vector<std::vector<HairNode>> strands = HairGen->LoadHair(hairPath, HairData.Move);
        BuildVboHair(strands);
    }

    SimSolver->InitParams();
}

void Scene::UpdateSimulation()
{
    if (Generated && !SimSolver->GetParamsSim().PauseSim)
    {
        // Updates either the simulation or pre-processor
        if (SimSolver->GetParamsSim().TooglePreProcess)
            SimSolver->UpdateProcessor();
        else
            SimSolver->UpdateSimulation();

        // Updates box info if drawing SDF
        if (SimSolver->GetParamsSim().DrawGrid)
            BuildVboHairGrid(true);
    }
}

void Scene::ResetSimulation()
{
    SimSolver->Reset();
}

void Scene::Update(Transform &trans, float delta)
{

    if(!ImGui::IsAnyItemActive() && !inputs::inst()->ctrl)
    {
            
        // Move light : holding alt + mouse
        if (inputs::inst()->alt)
        {
            m_lights[params::inst()->activeLight]->move(m_cameraManager, inputs::inst()->mouse_diff.x * 0.1f, inputs::inst()->mouse_diff.y * 0.1f);
        }

        // Start simulation : s
        if (inputs::inst()->s && !Generated)
        {
            InitSimulation();
            inputs::inst()->s = false;
            is_simulation_init = true;
        }

        // Select triangle : lclick + b
        if (inputs::inst()->left_button && inputs::inst()->b)
        {
            SelectTriangles(trans, 1);
        }

        // Deselect triangle : lclick + n
        if (inputs::inst()->left_button && inputs::inst()->n)
        {
            SelectTriangles(trans, 0);
        }

        // Cut Hair: lclick + toogle option
        if (inputs::inst()->right_button && inputs::inst()->a)
        {
            InitHairCut(trans);
        }

        //// Grab Hair: lclick + toogle option
        // if (inputs::inst()->left_button && inputs::inst()->i && SimSolver->GetParamsSim().ToogleGrabHair)
        //{
        //     UpdateHairGrab(trans);
        // }
        // else if (!SimSolver->GetParamsSim().ToogleGrabHair && LastGrabState)
        //{
        //     LastGrabState = false;
        //     SimSolver->KillGrab();
        // }

        // Restart Framework : r
        if (inputs::inst()->r)
        {
            Reset();
            inputs::inst()->r = false;
        }

        // Generate hair on selected triangles
        if (inputs::inst()->g)
        {
            GenerateHair();
            inputs::inst()->g = false;
        }

        // Restart particle's positions
        if (inputs::inst()->d)
        {
            ResetSimulation();
            inputs::inst()->d = false;
        }

        if (inputs::inst()->left_button && inputs::inst()->a) 
        {
            SimSolver->GetParamsSim().RigidAngle.x += 0.4 * inputs::inst()->mouse_diff.y;
            SimSolver->GetParamsSim().RigidAngle.y += 0.4 * inputs::inst()->mouse_diff.x;
	    }
    }
}

void Scene::SceneDrawGUI()
{

    SimulationParams &simParams = SimSolver->GetParamsSim();
    GenerationParams &genParams = HairGen->GetParamsGen();

    ImGui::Begin("Basic Settings");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    if (ImGui::BeginTabBar("SimTabBar", ImGuiTabBarFlags_None))
    {
        if (ImGui::BeginTabItem("Simulation"))
        {
            if (ImGui::CollapsingHeader("Toogle Options", ImGuiTreeNodeFlags_DefaultOpen))
            {
                if (ImGui::Checkbox("Build SimFramework", &Generated))
                {
                    if (!Generated)
                    {
                        InitSimulation();
                        is_simulation_init=true;
                    }
                }
                ImGui::Checkbox("Toogle Preprocess", &simParams.TooglePreProcess);
                ImGui::Checkbox("Update Sdf", &simParams.RecomputeSdf);
                ImGui::Checkbox("Attach Roots", &simParams.FixRootPositions);
                ImGui::Checkbox("Pause Simulation", &simParams.PauseSim);
                // ImGui::Checkbox("Export From Start", &simParams.ExportFromStart);
                // ImGui::SliderFloat("Eulerian Extension (%)", &simParams.LengthIncrease, 0.f, 1.f);
                ImGui::Checkbox("Head Interaction", &simParams.SolidCollision);
                ImGui::Checkbox("Hair-Hair Interaction", &simParams.HairCollision);
                ImGui::Checkbox("Hair color toogle", &temp);
                ImGui::Checkbox("Interpolate Hair", &genParams.Interpolate);
                ImGui::SliderInt("Num Interpolated", &genParams.NumInterpolated, 1, 10);
                ImGui::SliderInt("Min Cut", &simParams.CutMin, 1, 15);
                if(ImGui::Button("Reload Hair"))
                {
                    ResetAll("../data/hair/harolongo.data");
                }
                if(temp)
                {
                    genParams.HairDebbug = 1;
                }
                else
                {
                    genParams.HairDebbug = 0;
                }
                if (ImGui::Checkbox("Start Animation", &simParams.Animate))
                {
                    // simParams.ExportSequence = true;
                    // simParams.ExportHair = true;
                    // simParams.ExportMesh = true;
                    // simParams.ExportVDB = true;
                }
                if (ImGui::Checkbox("Wind Anim", &simParams.Wind))
                {
                    simParams.AnimationType = WIND;
                    simParams.Animate = true;
                }
                // ImGui::Checkbox("Toogle Cut Hair", &simParams.ToogleCutHair);
                // ImGui::Checkbox("Toogle Grab Hair", &simParams.ToogleGrabHair);
                // ImGui::SliderFloat("Cut Radious", &simParams.CutRadious, 0.f, 5.f);
                // ImGui::SliderFloat("Grab Radious", &simParams.GrabRadius, 0.f, 50.f);
                // ImGui::SliderFloat("Grab K", &simParams.GrabK, 0.f, 400.f);
            }
            if (ImGui::CollapsingHeader("Basic Parameters", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::SliderFloat("Move x", &simParams.RigidPos.x, -5.f, 5.f);
                ImGui::SliderFloat("Rot x", &simParams.RigidAngle.x, -90.f, 90.f);
                ImGui::SliderFloat("Rot y", &simParams.RigidAngle.y, -90.f, 90.f);
                ImGui::SliderFloat("Rot z", &simParams.RigidAngle.z, -90.f, 90.f);
                ImGui::SliderFloat("Move roots x", &simParams.RootMove.x, -10.f, 10.f);
                ImGui::SliderFloat("Time Step", &simParams.Dt, 0.001f, 0.1f);
                ImGui::SliderFloat("Gravity", &simParams.Gravity, 0.f, 100.f);
                ImGui::SliderFloat("Wind x", &simParams.WindSpeed.x, -80.f, 80.f);
                ImGui::SliderFloat("Wind y", &simParams.WindSpeed.y, -80.f, 80.f);
                ImGui::SliderFloat("Wind z", &simParams.WindSpeed.z, -80.f, 80.f);
                ImGui::SliderFloat("Angular K", &simParams.AngularK, 0.f, 9000.f);
                ImGui::SliderFloat("Gravity K", &simParams.GravityK, 0.f, 1000.f);
                ImGui::SliderFloat("Edge K", &simParams.EdgeK, 50.f, 90000.f);
                ImGui::SliderFloat("Bend K", &simParams.BendK, 50.f, 90000.f);
                ImGui::SliderFloat("Torsion K", &simParams.TorsionK, 50.f, 90000.f);
                ImGui::SliderFloat("Damping", &simParams.Damping, 0.f, 4.f);
                ImGui::SliderFloat("Hair Mass", &simParams.HairMass, 0.1f, 10.f);
                ImGui::SliderInt("Nested Steps", &simParams.NestedSteps, 1, 200);
                ImGui::SliderInt("Strain Steps", &simParams.StrainSteps, 1, 50);
                ImGui::SliderFloat("Strain Error", &simParams.StrainError, 0.1, 1);
            }
            if (ImGui::CollapsingHeader("SDF Parameters", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::SliderFloat("Additional SDF Extension", &simParams.LengthIncreaseGrid, 0.f, 1.f, "%.1f");
                ImGui::SliderFloat("Friction Head", &simParams.Friction, 0.f, 50.f);
                ImGui::SliderFloat("SDF Threshold", &simParams.ThreshSdf, 0.f, 3.f);
            }
            if (ImGui::CollapsingHeader("Eulerian Parameters", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::SliderInt("Voxel Neighbor Size", &simParams.NumGridNeighbors, 0, 10);
                ImGui::SliderInt("Number of Jacobi Steps", &simParams.NumStepsJacobi, 1, 50);
                ImGui::SliderFloat("Weight Jacobi", &simParams.JacobiWeight, 0.1, 1.0);
                ImGui::SliderFloat("FLIP-PIC Control", &simParams.FlipWeight, 0.f, 1.f);
            }
            if (ImGui::CollapsingHeader("Pre-Processing", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::Checkbox("Preprocess Hair", &simParams.TooglePreProcess);
                ImGui::Checkbox("Fix Root Positions", &simParams.FixRootPositions);
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Generation"))
        {
            if (ImGui::Checkbox("Save Triangle Selection", &simParams.SaveSelection))
            {
                SaveTriangleSelection();
                simParams.SaveSelection = false;
            }
            if (ImGui::Checkbox("Load Triangle Selection", &simParams.LoadSelection))
            {
                LoadTriangleSelection();
                simParams.LoadSelection = false;
            }
            ImGui::ColorEdit3("Color", genParams.HairColor);
            ImGui::SliderFloat("Selection Radius", &genParams.SelectionRadius, 0.01f, 0.8f);
            ImGui::SliderFloat("Step Length", &genParams.StepLength, 0.001f, 2.2f);
            ImGui::SliderInt("Steps Min", &genParams.StepsMin, 1, 200);
            ImGui::SliderInt("Steps Max", &genParams.StepsMax, 1, 200);
            ImGui::SliderInt("Strands Per Triangle", &genParams.HairsPerTri, 1, 20);
            ImGui::SliderFloat("Gravity Influence", &genParams.GravityInfluence, -1.f, 1.f);
            ImGui::SliderFloat("Gravity Dot Influence", &genParams.GravityDotInfluence, 0.5f, 1.f);
            ImGui::SliderFloat("Direction Noise", &genParams.DirNoise, 0.f, 1.f);
            ImGui::SliderFloat("Gravity Noise", &genParams.GravNoise, 0.f, 1.f);
            ImGui::SliderFloat("Hair Thickness", &genParams.HairThickness, 0.f, 0.002f, " % .5f");
            ImGui::SliderFloat("Spiral Radius", &genParams.SpiralRad, 0.f, 50.f);
            ImGui::SliderFloat("Spiral Multiplier", &genParams.FreqMult, -5.f, 5.f);
            ImGui::SliderFloat("Spiral Amount", &genParams.SpiralAmount, -5.f, 5.f);
            ImGui::SliderFloat("Spiral Y", &genParams.SpiralY, -5.f, 5.f);
            ImGui::SliderFloat("Spiral Impact", &genParams.SpiralImpact, 0.f, 1.f);
            ImGui::SliderFloat("Parting Impact", &genParams.PartingImpact, 0.f, 1.f);
            ImGui::SliderFloat("Parting Strength X", &genParams.PartingStrengthX, -1.f, 1.f);
            ImGui::SliderFloat("Parting Strength Y", &genParams.PartingStrengthY, -1.f, 1.f);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();

    ImGui::Begin("Advanced Settings");
    if (ImGui::BeginTabBar("AdvTabBar", ImGuiTabBarFlags_None))
    {
        if (ImGui::BeginTabItem("Animation"))
        {
            if (ImGui::CollapsingHeader("Bones", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::SliderInt("Selected Bone", &simParams.SelectedBone, 0, max(0, simParams.NumBones - 1));
                Bone &b = Meshes[0]->GetBones()[simParams.SelectedBone];
                ImGui::SliderFloat("Bone Angle x", &b.Angles.x, -180.f, 180.f);
                ImGui::SliderFloat("Bone Angle y", &b.Angles.y, -180.f, 180.f);
                ImGui::SliderFloat("Bone Angle z", &b.Angles.z, -180.f, 180.f);
                ImGui::SliderFloat("Bone Move x", &b.MoveVec.x, -5.f, 5.f);
                ImGui::SliderFloat("Bone Move y", &b.MoveVec.y, -5.f, 5.f);
                ImGui::SliderFloat("Bone Move z", &b.MoveVec.z, -5.f, 5.f);
                ImGui::SliderFloat("Bone Scale", &b.ScaleNum, 0.1f, 10.f);
            }
            if (ImGui::CollapsingHeader("Sequence", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::SliderFloat("Duration Sequence", &simParams.AnimDuration, 0.1f, 100.f);
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Import/Export"))
        {
            if (ImGui::CollapsingHeader("Basic", ImGuiTreeNodeFlags_DefaultOpen))
            {
                bool temp = false;
                ImGui::Checkbox("Export Sequence", &simParams.ExportSequence);
                if (ImGui::Checkbox("Export VDB", &simParams.ExportVDB))
                    simParams.Export = true;
                if (ImGui::Checkbox("Export Hair", &simParams.ExportHair))
                    simParams.Export = true;
                if (ImGui::Checkbox("Export Mesh", &simParams.ExportMesh))
                    simParams.Export = true;
                ImGui::SliderInt("Export Step", &simParams.ExportStep, 1, 30);
                ImGui::Checkbox("Export DATA format", &simParams.ExportDataFormat);
                ImGui::Checkbox("Export Uniform Strands", &simParams.FixedHairLength);
                if (ImGui::Checkbox("Export Anim", &temp))
                {
                    simParams.Export = true;
                    simParams.ExportSequence = true;
                    simParams.ExportHair = true;
                    simParams.ExportMesh = true;
                }
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Rendering"))
        {
            ImGui::Checkbox("Draw Head", &simParams.DrawHead);
            ImGui::Checkbox("Draw Head Wireframe", &simParams.DrawHeadWF);
            ImGui::Checkbox("Draw Bone Weight", &simParams.DrawBoneWeight);
            ImGui::Checkbox("Draw Bones", &simParams.DrawBones);
            ImGui::Checkbox("Draw Bones Axis", &simParams.DrawBonesAxis);
            ImGui::Checkbox("Draw Euler Box", &simParams.DrawGrid);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void Scene::InitSimulation()
{

    // Init simulation data
    if (VERBOSE)
        printf("Building additional VBOs...");

    GenerateHairGrid();
    BuildVboHairGrid();
    BuildSimVBO();

    if (VERBOSE)
        printf("done\n");

    // Generating data for solver
    if (VERBOSE)
        printf("Building simulation data... ");

    HairGen->GenerateParticles();

    if (VERBOSE)
        printf("generated %i particles\n", HairGen->GetParticles().size());
    if (VERBOSE)
        printf("Sending data to solver... ");

    // Init Solver

    // Get OpenGL ID for VBO mapping
    vector<GLuint> allIdx = {VboMeshes[0]->id(), VboHair[0]->id()};

    SimSolver->InitBuffers(Meshes, allIdx, HairGen);
    SimSolver->InitParticles();
    SimSolver->ComputeSdfCPU();
    SimSolver->ComputeBary();
    Generated = true;

    // Clears memory
    Meshes.clear();
    HairGen->Clear();

    SimSolver->GetParamsSim().GravityK = 300.f;
}

// void Scene::UpdateSDFBox()
//{
//     // Gets information of the box
//     float3 origin = SimSolver->GetParamsSim().EulerOrigin;
//     float3 localX = SimSolver->GetParamsSim().EulerLocalX;
//     float3 localY = SimSolver->GetParamsSim().EulerLocalY;
//     float3 localZ = SimSolver->GetParamsSim().EulerLocalZ;
//     int nx = SimSolver->GetParamsSim().GridSize.x;
//     int ny = SimSolver->GetParamsSim().GridSize.y;
//     int nz = SimSolver->GetParamsSim().GridSize.z;
//     float dx = SimSolver->GetParamsSim().Ds.x;
//     float dy = SimSolver->GetParamsSim().Ds.y;
//     float dz = SimSolver->GetParamsSim().Ds.z;
//
//     // box vertices
//     float3 a = origin;
//     float3 b = a + nx * dx * localX;
//     float3 c = a + ny * dy * localY;
//     float3 d = a + nx * dx * localX + ny * dy * localY;
//     float3 e = a + nz * dz * localZ;
//     float3 f = a + nx * dx * localX + nz * dz * localZ;
//     float3 g = a + ny * dy * localY + nz * dz * localZ;
//     float3 h = a + nx * dx * localX + ny * dy * localY + nz * dz * localZ;
//
//     // Colors
//     float4 y = make_float4(0.f, 1.f, 1.f, 1.f);
//     float4 r = make_float4(1.f, 0.f, 0.f, 1.f);
//     float4 bl = make_float4(0.f, 0.f, 1.f, 1.f);
//
//     vector<float3> positions = {
//         a,b,d,
//         a,d,c,
//         a,e,c,
//         e,g,c,
//         b,f,d,
//         f,d,h,
//         a,b,e,
//         e,b,f,
//         c,d,g,
//         g,h,d,
//         e,f,g,
//         g,f,h
//     };
//
//     int size = 0;
//     VertexBufferObject::DATA* data = VboSDFBox->map(size);
//
//     for (int i = 0; i < positions.size(); ++i)
//     {
//         data[i].vx = positions[i].x;
//         data[i].vy = positions[i].y;
//         data[i].vz = positions[i].z;
//     }
//
//     VboSDFBox->unmap();
// }

void Scene::BuildSimVBO()
{
    // Clears temporal VBO
    VboHair.clear();

    // Fills final hair VBO
    BuildVboHair(HairGen->GetStrands());
}

void Scene::InitHairCut(Transform &trans)
{
    vec3 rayPnt;
    vec3 rayDir;

    Picking pick;

    float fov = params::inst()->fov;
    float ncp = params::inst()->ncp;
    float windowX = params::inst()->window_x;
    float windowY = params::inst()->window_y;
    float mx = inputs::inst()->mouse_cur.x;
    float my = inputs::inst()->mouse_cur.y;

    pick.getPickingRay(trans, fov, ncp, windowX, windowY, mx, my, rayPnt, rayDir);

    SimSolver->InitCut(vec2float(rayPnt), normalize(vec2float(rayDir)));
}

void Scene::SelectTriangles(Transform &trans, int val)
{

    vec3 rayPnt;
    vec3 rayDir;

    Picking pick;

    float fov = params::inst()->fov;
    float ncp = params::inst()->ncp;
    float windowX = params::inst()->window_x;
    float windowY = params::inst()->window_y;
    float mx = inputs::inst()->mouse_cur.x;
    float my = inputs::inst()->mouse_cur.y;

    pick.getPickingRay(trans, fov, ncp, windowX, windowY, mx, my, rayPnt, rayDir);

    float min_dist = numeric_limits<float>::max();
    vec3 min_point = vec3();
    int min_idx = 0;
    bool foundMin = false;

    // Intersect screen ray with main mesh
    for (int i = 0; i < Meshes[0]->GetTriangles().size(); ++i)
    {
        SimpleTriangle &t = *Meshes[0]->GetTriangles()[i];
        float3 &v0 = Meshes[0]->GetVertices()[t.V[0]]->Pos;
        float3 &v1 = Meshes[0]->GetVertices()[t.V[1]]->Pos;
        float3 &v2 = Meshes[0]->GetVertices()[t.V[2]]->Pos;

        vec3 point;
        float u = 0.0f;
        bool res = t.Intersect(vec2float(rayPnt), vec2float(rayDir), v0, v1, v2, u);

        if (res)
        {
            point = rayPnt + normalize(rayDir) * u;
            float dist = length(rayPnt - point);
            if (dist < min_dist)
            {
                min_dist = dist;
                min_point = point;
                min_idx = i;
                foundMin = true;
            }
        }
    }

    // Update selected status of all triangles within neighborhood
    // both VBO and CPU
    if (foundMin)
    {
        // Maps vbo data
        int size = 0;
        VertexBufferObject::DATA *data = VboMeshes[0]->map(size);

        // Iterates over triangles (parallel)
        float radius = HairGen->GetParamsGen().SelectionRadius;
#pragma omp parallel for
        for (int i = 0; i < Meshes[0]->GetTriangles().size(); ++i)
        {
            SimpleTriangle &t = *Meshes[0]->GetTriangles()[i];
            vec3 c = float2vec(t.Center);

            if (length(c - min_point) < radius)
            {
                // CPU Update
                t.Selected = val;

                // GPU Update
                data[3 * i].tw = val;
                data[3 * i + 1].tw = val;
                data[3 * i + 2].tw = val;
            }
        }
    }

    VboMeshes[0]->unmap();
}

// void Scene::InitHairCut(Transform& trans)
//{
//     vec3 rayPnt;
//     vec3 rayDir;
//
//     Picking pick;
//
//     float fov = params::inst()->fov;
//     float ncp = params::inst()->ncp;
//     float windowX = params::inst()->window_x;
//     float windowY = params::inst()->window_y;
//     float mx = inputs::inst()->mouse_cur.x;
//     float my = inputs::inst()->mouse_cur.y;
//
//     pick.getPickingRay(trans, fov, ncp, windowX, windowY, mx, my, rayPnt, rayDir);
//     SimSolver->TriggerHairCut(vec2float(rayPnt), normalize(vec2float(rayDir)));
// }

// void Scene::UpdateHairGrab(Transform& trans)
//{
//     vec3 rayPnt;
//     vec3 rayDir;
//
//     Picking pick;
//
//     float fov = params::inst()->fov;
//     float ncp = params::inst()->ncp;
//     float windowX = params::inst()->window_x;
//     float windowY = params::inst()->window_y;
//     float mx = inputs::inst()->mouse_cur.x;
//     float my = inputs::inst()->mouse_cur.y;
//
//     pick.getPickingRay(trans, fov, ncp, windowX, windowY, mx, my, rayPnt, rayDir);
//
//     // Intersection with plane in the same direction
//     vec3 planeNormal(rayDir.x, rayDir.y, rayDir.z);
//     vec3 planeP(0, 0, 0);
//     float lambda = rayPlaneIntersection(rayPnt, rayDir, planeNormal, planeP);
//     float3 centerPoint = vec2float(rayPnt + lambda * rayDir);
//
//     if (LastGrabState)
//     {
//         SimSolver->UpdateGrabPos(centerPoint);
//     }
//     else
//     {
//         SimSolver->TriggerHairGrab(centerPoint);
//         LastGrabState = true;
//     }
// }

// void Scene::UpdateGrabPosition(Transform& trans)
//{
//     vec3 rayPnt;
//     vec3 rayDir;
//
//     Picking pick;
//
//     float fov = params::inst()->fov;
//     float ncp = params::inst()->ncp;
//     float windowX = params::inst()->window_x;
//     float windowY = params::inst()->window_y;
//     float mx = inputs::inst()->mouse_cur.x;
//     float my = inputs::inst()->mouse_cur.y;
//
//     pick.getPickingRay(trans, fov, ncp, windowX, windowY, mx, my, rayPnt, rayDir);
//
//     // Intersection with plane in the same direction
//     vec3 planeNormal(rayDir.x, rayDir.y, rayDir.z);
//     vec3 planeP(0, 0, 0);
//     float lambda = rayPlaneIntersection(rayPnt, rayDir, planeNormal, planeP);
//     float3 centerPoint = vec2float(rayPnt + lambda * rayDir);
//
//     SimSolver->UpdateGrabPos(centerPoint);
// }

void Scene::SelectMesh(int val)
{
    // Maps vbo data
    int size = 0;
    VertexBufferObject::DATA *data = VboMeshes[0]->map(size);

    // Iterates over triangles (parallel)
    float radius = HairGen->GetParamsGen().SelectionRadius;
#pragma omp parallel for
    for (int i = 0; i < Meshes[0]->GetTriangles().size(); ++i)
    {
        SimpleTriangle &t = *Meshes[0]->GetTriangles()[i];

        // CPU Update
        t.Selected = val;

        // GPU Update
        data[3 * i].tw = val;
        data[3 * i + 1].tw = val;
        data[3 * i + 2].tw = val;
    }

    VboMeshes[0]->unmap();
}

void Scene::SetupExportFolder()
{
    // Check if export already exists

    namespace fs = boost::filesystem;
    const fs::path p{"./Export"};
    fs::file_status s = fs::file_status{};
    if (!(fs::status_known(s) ? fs::exists(s) : fs::exists(p)))
    {
        fs::create_directory(p);
    }
}

void Scene::Reset()
{
    Generated = false;
    params::inst()->lights[0]->m_moved = true;

    // Restart generator and solver
    // HairGen->Restart();
    SelectMesh(0);

    // Clear data
    VboGridBox.reset();
    VboHair.clear();
    HairGen->Clear();

    // Re-loads hair
    if (HairData.ID != -1)
    {
        std::vector<std::vector<HairNode>> strands = HairGen->LoadHair(HairData.FileName, HairData.Move);
        BuildVboHair(strands);
    }

    if (VERBOSE)
        printf("Framework Restart\n");
    if (VERBOSE)
        printf("---------------------\n");
}

void Scene::SetupExperiment(ExpOpt opt)
{
    switch (opt)
    {
    case ANIMATION:
    {
        //
        SimSolver->GetParamsSim().Dt = 0.06;
        break;
    }
    case CANTI_E:
    {
        // No head for this one
        SimSolver->GetParamsSim().SolidCollision = false;
        SimSolver->GetParamsSim().DrawHead = false;

        // 500 Sim Frames
        SimSolver->GetParamsSim().SimEndTime = 500.f;

        // Only gravitational spring for testing
        SimSolver->GetParamsSim().AngularK = 0.f;
        SimSolver->GetParamsSim().GravityK = 40.f;

        // We can push the timestep, and change damping for more elastic
        SimSolver->GetParamsSim().Dt = 0.04;
        SimSolver->GetParamsSim().Damping = 2.5f;
        // SimSolver->GetParamsSim().ExportFromStart = true;
        SimSolver->GetParamsSim().ExportStep = 1;

        // Init rod
        float3 x0 = make_float3(0.f, 5.f, 0.f);
        float3 xn = make_float3(15.f, 5.f, 0.f);
        int steps = 50;

        // HairGen->GenerateRod(x0, xn, steps);

        // Builds VBO
        // BuildVboHair(HairGen->GetTempStrands());

        break;
    }
    case SAG_WIND:
    {
        // SimSolver->GetParamsSim().ThreshSDF = 0.3f;
        SimSolver->GetParamsSim().Friction = 1.2f;
        // SimSolver->GetParamsSim().CutInteractionZ = -12.f;
        // SimSolver->GetParamsSim().Animate = true;
        break;
    }
    case SAG_HEAD:
    {
        // SimSolver->GetParamsSim().ThreshSDF = 0.4f;
        SimSolver->GetParamsSim().Friction = 1.5f;
        SimSolver->GetParamsSim().Dt = 0.01;
        SimSolver->GetParamsSim().AngularK = 80.f;
        SimSolver->GetParamsSim().GravityK = 140.f;
        SimSolver->GetParamsSim().EdgeK = 20000.f;
        SimSolver->GetParamsSim().BendK = 10000.f;
        SimSolver->GetParamsSim().TorsionK = 0.f; // 20000.f;
        SimSolver->GetParamsSim().Damping = 20.f; // 10.f;
        // SimSolver->GetParamsSim().CutInteractionZ = 0.f;
        SimSolver->GetParamsSim().Gravity = 50.f;
        break;
    }
    case WIG:
    {
        // No head for this one
        SimSolver->GetParamsSim().SolidCollision = true;
        SimSolver->GetParamsSim().DrawHead = true;
        SimSolver->GetParamsSim().Gravity = 100.f;
        SimSolver->GetParamsSim().EdgeK = 20000.f;
        SimSolver->GetParamsSim().BendK = 10000.f;
        SimSolver->GetParamsSim().TorsionK = 20000.f;
        // SimSolver->GetParamsSim().CutInteractionZ = 33.f;
        SimSolver->GetParamsSim().AngularK = 40.f;
        SimSolver->GetParamsSim().GravityK = 150.f; // 150.f;
        SimSolver->GetParamsSim().Damping = 5.f;
        SimSolver->GetParamsSim().Friction = 1.5f;

        // SimSolver->GetParamsSim().TightExport = true;
        // SimSolver->GetParamsSim().ExportSampleSize = 20;

        // Size for testing
        HairGen->GetParamsGen().MaxLoadNumStrands = 10000;
        HairGen->GetParamsGen().MaxLoadStrandSize = 20;

        // Move
        break;
    }
    case SALON:
    {
        // SimSolver->GetParamsSim().ThreshSDF = 0.4f;
        SimSolver->GetParamsSim().Friction = 1.5f;
        SimSolver->GetParamsSim().Dt = 0.01;
        SimSolver->GetParamsSim().AngularK = 80.f;
        SimSolver->GetParamsSim().GravityK = 140.f;
        SimSolver->GetParamsSim().EdgeK = 20000.f;
        SimSolver->GetParamsSim().BendK = 10000.f;
        SimSolver->GetParamsSim().TorsionK = 0.f; // 20000.f;
        SimSolver->GetParamsSim().Damping = 10.f; // 10.f;
        SimSolver->GetParamsSim().Gravity = 100.f;
        // SimSolver->GetParamsSim().CutInteractionZ = 0.7f;

        // Size for testing
        HairGen->GetParamsGen().MaxLoadNumStrands = 1000000;
        HairGen->GetParamsGen().MaxLoadStrandSize = 20;

        break;
    }
    case MODERN:
    {
        // SimSolver->GetParamsSim().ThreshSDF = 0.2f;
        SimSolver->GetParamsSim().Friction = 2.0f;
        SimSolver->GetParamsSim().Dt = 0.01;
        SimSolver->GetParamsSim().AngularK = 80.f;
        SimSolver->GetParamsSim().GravityK = 140.f;
        SimSolver->GetParamsSim().EdgeK = 20000.f;
        SimSolver->GetParamsSim().BendK = 10000.f;
        SimSolver->GetParamsSim().TorsionK = 0.f; // 20000.f;
        SimSolver->GetParamsSim().Damping = 10.f; // 10.f;
        // SimSolver->GetParamsSim().CutInteractionZ = 30.f;
        SimSolver->GetParamsSim().Gravity = 100.f;
        // SimSolver->GetParamsSim().CutInteractionZ = 0.7f;

        // Size for testing
        HairGen->GetParamsGen().MaxLoadNumStrands = 1000000;
        HairGen->GetParamsGen().MaxLoadStrandSize = 20;

        break;
    }
    case CURLY:
    {
        // SimSolver->GetParamsSim().ThreshSDF = 0.1f;
        SimSolver->GetParamsSim().Friction = 1.5f;
        SimSolver->GetParamsSim().Dt = 0.01;
        SimSolver->GetParamsSim().AngularK = 80.f;
        SimSolver->GetParamsSim().GravityK = 140.f;
        SimSolver->GetParamsSim().EdgeK = 20000.f;
        SimSolver->GetParamsSim().BendK = 10000.f;
        SimSolver->GetParamsSim().TorsionK = 0.f; // 20000.f;
        SimSolver->GetParamsSim().Damping = 10.f; // 10.f;
        SimSolver->GetParamsSim().Gravity = 100.f;
        // SimSolver->GetParamsSim().CutInteractionZ = 21.f;

        // Size for testing
        HairGen->GetParamsGen().MaxLoadNumStrands = 1000000;
        HairGen->GetParamsGen().MaxLoadStrandSize = 20;

        break;
    }
    case ROLLER:
    {
        // SimSolver->GetParamsSim().ThreshSDF = 0.5f;
        SimSolver->GetParamsSim().Friction = 2.5f;
        SimSolver->GetParamsSim().Dt = 0.01;
        SimSolver->GetParamsSim().AngularK = 80.f;
        SimSolver->GetParamsSim().GravityK = 140.f;
        SimSolver->GetParamsSim().EdgeK = 20000.f;
        SimSolver->GetParamsSim().BendK = 10000.f;
        SimSolver->GetParamsSim().TorsionK = 0.f;
        SimSolver->GetParamsSim().Damping = 15.f; // 10.f;
        // SimSolver->GetParamsSim().CutInteractionZ = 0.f;
        SimSolver->GetParamsSim().Gravity = 100.f;
        // SimSolver->GetParamsSim().CutInteractionZ = 30.7f;

        // Size for testing
        HairGen->GetParamsGen().MaxLoadNumStrands = 1000000;
        HairGen->GetParamsGen().MaxLoadStrandSize = 20;

        break;
    }
    case CHENGAN:
    {
        // SimSolver->GetParamsSim().ThreshSDF = 0.1f;
        SimSolver->GetParamsSim().Friction = 1.5f;
        SimSolver->GetParamsSim().Dt = 0.01;
        SimSolver->GetParamsSim().AngularK = 80.f;
        SimSolver->GetParamsSim().GravityK = 140.f;
        SimSolver->GetParamsSim().EdgeK = 20000.f;
        SimSolver->GetParamsSim().BendK = 10000.f;
        SimSolver->GetParamsSim().TorsionK = 0.f; // 20000.f;
        SimSolver->GetParamsSim().Damping = 10.f; // 10.f;
        SimSolver->GetParamsSim().Gravity = 100.f;
        // SimSolver->GetParamsSim().CutInteractionZ = 21.f;

        // Size for testing
        HairGen->GetParamsGen().MaxLoadNumStrands = 1000000;
        HairGen->GetParamsGen().MaxLoadStrandSize = 20;

        break;
    }
    case KATE:
    {
        // SimSolver->GetParamsSim().ThreshSDF = 0.4f;
        SimSolver->GetParamsSim().Friction = 1.5f;
        SimSolver->GetParamsSim().Dt = 0.01;
        SimSolver->GetParamsSim().AngularK = 80.f;
        SimSolver->GetParamsSim().GravityK = 140.f;
        SimSolver->GetParamsSim().EdgeK = 20000.f;
        SimSolver->GetParamsSim().BendK = 10000.f;
        SimSolver->GetParamsSim().TorsionK = 0.f; // 20000.f;
        SimSolver->GetParamsSim().Damping = 10.f; // 10.f;
        // SimSolver->GetParamsSim().CutInteractionZ = 0.f;
        SimSolver->GetParamsSim().Gravity = 100.f;
        // SimSolver->GetParamsSim().CutInteractionZ = 0.7f;

        // Size for testing
        HairGen->GetParamsGen().MaxLoadNumStrands = 1000000;
        HairGen->GetParamsGen().MaxLoadStrandSize = 20;

        break;
    }
    case SPACE:
    {
        SimSolver->GetParamsSim().SolidCollision = false;
        // SimSolver->GetParamsSim().ThreshSDF = 0.f;
        SimSolver->GetParamsSim().Friction = 1.5f;
        SimSolver->GetParamsSim().Dt = 0.01;
        SimSolver->GetParamsSim().AngularK = 80.f;
        SimSolver->GetParamsSim().GravityK = 140.f;
        SimSolver->GetParamsSim().EdgeK = 20000.f;
        SimSolver->GetParamsSim().BendK = 10000.f;
        SimSolver->GetParamsSim().TorsionK = 0.f; // 20000.f;
        SimSolver->GetParamsSim().Damping = 10.f; // 10.f;
        // SimSolver->GetParamsSim().CutInteractionZ = 0.f;
        SimSolver->GetParamsSim().Gravity = 100.f;
        // SimSolver->GetParamsSim().CutInteractionZ = 0.7f;

        // Size for testing
        HairGen->GetParamsGen().MaxLoadNumStrands = 1000000;
        HairGen->GetParamsGen().MaxLoadStrandSize = 20;
        HairGen->GetParamsGen().HairsPerTri = 3;

        // length vs. gravity influence
        // l : .001 - 0.3
        HairGen->GetParamsGen().DirNoise = 1.f;
        HairGen->GetParamsGen().SpiralImpact = .017f;
        HairGen->GetParamsGen().SpiralRad = 17.f;
        break;
    }
    case FACIAL:
    {
        // SimSolver->GetParamsSim().ThreshSDF = 0.f;
        SimSolver->GetParamsSim().Friction = 1.5f;
        SimSolver->GetParamsSim().Dt = 0.01;
        SimSolver->GetParamsSim().AngularK = 80.f;
        SimSolver->GetParamsSim().GravityK = 140.f;
        SimSolver->GetParamsSim().EdgeK = 20000.f;
        SimSolver->GetParamsSim().BendK = 10000.f;
        SimSolver->GetParamsSim().TorsionK = 0.f; // 20000.f;
        // SimSolver->GetParamsSim().Damping = 10.f;//10.f; //short
        SimSolver->GetParamsSim().Damping = 13.f; // long
        // SimSolver->GetParamsSim().CutInteractionZ = 0.f;
        SimSolver->GetParamsSim().Gravity = 100.f;
        // SimSolver->GetParamsSim().CutInteractionZ = 0.7f;

        // Size for testing
        HairGen->GetParamsGen().MaxLoadNumStrands = 1000000;
        HairGen->GetParamsGen().MaxLoadStrandSize = 20;

        break;
    }
    case SICTION:
    {
        // Loads hair square
        int numStrands = 4000;
        float4 limits = make_float4(-15.f, -15.f + 3.f, 15.f, -5.f + 3.f);
        float height = 13;
        
        std::vector<std::vector<HairNode>> strands = HairGen->CreateTestSiction(numStrands, limits, height);
        BuildVboHair(strands);
        break;
    }
    case SANDBOX:
        break;
    case NONE:
        break;
    default:
        break;
    }
}

void Scene::ExportExplorationLG()
{
    float LA = .001f;
    float LB = 0.25f;
    float stepL = (LB - LA) / 3.f;

    float GA = -1.f;
    float GB = 1.f;
    float stepG = (GB - GA) / 3.f;

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            // erase hair
            Reset();

            // change params
            HairGen->GetParamsGen().StepLength = LA + stepL * i;
            HairGen->GetParamsGen().GravityInfluence = GA + stepG * j;

            // get triangle selection
            LoadTriangleSelection();

            // generate hair
            GenerateHair();

            // export
            string fname = "./Export/hair_explore_LG_" + to_string(4 * i + j) + ".obj";
            // SmallExporter(fname);
        }
    }
}

void Scene::ExportExplorationLS()
{
    float LA = .001f;
    float LB = 0.25f;
    float stepL = (LB - LA) / 3.f;

    float SA = 0.f;
    float SB = 50.f;
    float stepS = (SB - SA) / 3.f;

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            // erase hair
            Reset();

            // change params
            HairGen->GetParamsGen().GravityInfluence = 0.118f;
            HairGen->GetParamsGen().SpiralAmount = 5.f;
            HairGen->GetParamsGen().StepLength = LA + stepL * i;
            HairGen->GetParamsGen().SpiralRad = SA + stepS * j;

            // get triangle selection
            LoadTriangleSelection();

            // generate hair
            GenerateHair();

            // export
            string fname = "./Export/hair_explore_LS_" + to_string(4 * i + j) + ".obj";
            // SmallExporter(fname);
        }
    }
}

// void Scene::SmallExporter(string fname)
//{
//     // Temporal object for saving
//     Loader hair;
//
//     // Gets generated strands
//     vector<vector<HairNode>> strands = HairGen->GetTempStrands();
//     hair.strands = vector<vector<Eigen::Vector3f>>(strands.size());
//     #pragma omp parallel for
//     for (int i = 0; i < strands.size(); i++)
//     {
//         hair.strands[i] = vector<Eigen::Vector3f>(strands[i].size());
//         #pragma omp parallel for
//         for (int j = 0; j < strands[i].size(); j++)
//         {
//             float3 u = strands[i][j].Pos;
//             hair.strands[i][j] = Eigen::Vector3f(u.x, u.y, u.z);
//         }
//
//     }
//
//     // Export
//     hair.save_obj(fname);
// }

void Scene::BuildVboMeshes()
{
    VboMeshes.clear();

    for (int i = 0; i < Meshes.size(); i++)
    {
        vector<float3> positions;
        vector<float3> normals;
        vector<float2> textures;
        vector<float> selections;

        for (int j = 0; j < Meshes[i]->GetNumTriangles(); j++)
        {
            SimpleTriangle &t = *Meshes[i]->GetTriangles()[j];
            for (int vertex = 0; vertex < 3; vertex++)
            {
                SimpleVertex &v = *Meshes[i]->GetVertices()[t.V[vertex]];
                positions.push_back(v.Pos);
                normals.push_back(v.Normal);
                textures.push_back(v.Tex);
                selections.push_back(t.Selected);
            }
        }

        VertexBufferObject::DATA *data_mesh = new VertexBufferObject::DATA[positions.size()];

        for (uint j = 0; j < positions.size(); j++)
        {
            float3 v = positions[j];
            float3 n = normals[j];
            float2 t = textures[j];
            float s = selections[j];

            data_mesh[j].vx = v.x;
            data_mesh[j].vy = v.y;
            data_mesh[j].vz = v.z;
            data_mesh[j].vw = 0.0f;

            data_mesh[j].cx = 1.0f;
            data_mesh[j].cy = 1.0f;
            data_mesh[j].cz = 1.0f;
            data_mesh[j].cw = 1.0f;

            data_mesh[j].nx = n.x;
            data_mesh[j].ny = n.y;
            data_mesh[j].nz = n.z;
            data_mesh[j].nw = 0.0f;

            data_mesh[j].tx = t.x;
            data_mesh[j].ty = t.y;
            data_mesh[j].tz = 0.f;
            data_mesh[j].tw = s;
        }

        shared_ptr<VertexBufferObject> vbo = make_shared<VertexBufferObject>();
        vbo->setData(data_mesh, GL_DYNAMIC_DRAW, (GLuint)positions.size(), GL_TRIANGLES);
        vbo->bindDefaultAttribs();

        VboMeshes.push_back(vbo);

        delete[] data_mesh;
    }
}

void Scene::BuildVboBone()
{
    vector<float3> positions;
    vector<float3> normals;
    for (int i = 0; i < BoneMesh->GetNumTriangles(); i++)
    {
        SimpleTriangle &t = *BoneMesh->GetTriangles()[i];
        for (int j = 0; j < 3; j++)
        {
            SimpleVertex &v = *BoneMesh->GetVertices()[t.V[j]];
            positions.push_back(v.Pos);
            normals.push_back(v.Normal);
        }
    }

    VertexBufferObject::DATA *data_mesh = new VertexBufferObject::DATA[positions.size()];
    for (uint i = 0; i < positions.size(); i++)
    {
        float3 v = positions[i];
        float3 n = normals[i];

        data_mesh[i].vx = v.x;
        data_mesh[i].vy = v.y;
        data_mesh[i].vz = v.z;
        data_mesh[i].vw = 0.0f;

        data_mesh[i].cx = v.y < 0.5 ? 0.f : 1.f;
        data_mesh[i].cy = v.y < 0.5 ? 0.f : 1.f;
        data_mesh[i].cz = v.y < 0.5 ? 0.f : 1.f;
        data_mesh[i].cw = 1.f;

        data_mesh[i].nx = n.x;
        data_mesh[i].ny = n.y;
        data_mesh[i].nz = n.z;
        data_mesh[i].nw = 0.0f;

        data_mesh[i].tx = 0.f;
        data_mesh[i].ty = 0.f;
        data_mesh[i].tz = 0.f;
        data_mesh[i].tw = 1.f;
    }

    VboBone = make_shared<VertexBufferObject>();
    VboBone->setData(data_mesh, GL_DYNAMIC_DRAW, (GLuint)positions.size(), GL_TRIANGLES);
    VboBone->bindDefaultAttribs();

    delete[] data_mesh;
}

void Scene::BuildVboHairGrid(const bool &update)
{
    // Gets information of the box
    float3 origin = SimSolver->GetParamsSim().HairOrig;
    float3 *axis = SimSolver->GetParamsSim().HairAxis;
    int3 ns = SimSolver->GetParamsSim().HairDim;
    float3 ds = SimSolver->GetParamsSim().HairDs;

    // box vertices
    float3 a = origin;
    float3 b = a + ns.x * ds.x * axis[0];
    float3 c = a + ns.y * ds.y * axis[1];
    float3 d = a + ns.x * ds.x * axis[0] + ns.y * ds.y * axis[1];
    float3 e = a + ns.z * ds.z * axis[2];
    float3 f = a + ns.x * ds.x * axis[0] + ns.z * ds.z * axis[2];
    float3 g = a + ns.y * ds.y * axis[1] + ns.z * ds.z * axis[2];
    float3 h = a + ns.x * ds.x * axis[0] + ns.y * ds.y * axis[1] + ns.z * ds.z * axis[2];

    // Colors
    float4 y = make_float4(0.f, 1.f, 1.f, 1.f);
    float4 r = make_float4(1.f, 0.f, 0.f, 1.f);
    float4 bl = make_float4(0.f, 0.f, 1.f, 1.f);

    vector<float3> positions = {
        a, b, d,
        a, d, c,
        a, e, c,
        e, g, c,
        b, f, d,
        f, d, h,
        a, b, e,
        e, b, f,
        c, d, g,
        g, h, d,
        e, f, g,
        g, f, h};

    vector<float4> colors = {
        y, y, y,
        y, y, y,
        bl, bl, bl,
        bl, bl, bl,
        bl, bl, bl,
        bl, bl, bl,
        r, r, r,
        r, r, r,
        r, r, r,
        r, r, r,
        y, y, y,
        y, y, y};

    if (update)
    {
        int size = 0;
        VertexBufferObject::DATA *data = VboGridBox->map(size);

        for (int i = 0; i < positions.size(); ++i)
        {
            data[i].vx = positions[i].x;
            data[i].vy = positions[i].y;
            data[i].vz = positions[i].z;
        }

        VboGridBox->unmap();
    }
    else
    {
        VertexBufferObject::DATA *data_mesh = new VertexBufferObject::DATA[positions.size()];

        for (uint i = 0; i < positions.size(); ++i)
        {
            float3 v = positions[i];
            float4 c = colors[i];

            data_mesh[i].vx = v.x;
            data_mesh[i].vy = v.y;
            data_mesh[i].vz = v.z;
            data_mesh[i].vw = 0.0f;

            data_mesh[i].cx = c.x;
            data_mesh[i].cy = c.y;
            data_mesh[i].cz = c.z;
            data_mesh[i].cw = c.w;

            data_mesh[i].nx = 0.f;
            data_mesh[i].ny = 0.f;
            data_mesh[i].nz = 0.f;
            data_mesh[i].nw = 0.0f;

            data_mesh[i].tx = 0.f;
            data_mesh[i].ty = 0.f;
            data_mesh[i].tz = 0.0f;
            data_mesh[i].tw = 0.f;
        }

        VboGridBox = make_shared<VertexBufferObject>();
        VboGridBox->setData(data_mesh, GL_DYNAMIC_DRAW, (GLuint)positions.size(), GL_TRIANGLES);
        VboGridBox->bindDefaultAttribs();

        delete[] data_mesh;
    }
}

void Scene::BuildVboHair(vector<vector<HairNode>> &strands)
{
    // Prepares VBO data
    vector<float3> positions;
    vector<float4> colors;
    vector<float3> orthos;
    vector<float3> pars;
    vector<float3> dirs;
    vector<float> thicknesses;

    // Fills
    int counter = 0;
    for (int i = 0; i < strands.size(); ++i)
    {
        float4 c = make_float4(rand(0.f, 1.f), rand(0.f, 1.f), rand(0.f, 1.f), 1.0f);
        for (int j = 0; j < strands[i].size() - 1; ++j)
        {
            float3 pos_1 = strands[i][j].Pos;
            float3 ortho_1 = strands[i][j].Ortho;
            float3 par_1 = strands[i][j].Par;
            float3 dir_1 = strands[i][j].Dir;
            float thick_1 = strands[i][j].Thick;

            float3 pos_2 = strands[i][j + 1].Pos;
            float3 ortho_2 = strands[i][j + 1].Ortho;
            float3 par_2 = strands[i][j + 1].Par;
            float3 dir_2 = strands[i][j + 1].Dir;
            float thick_2 = strands[i][j].Thick;

            if (j == 0)
            {
                ortho_1 = ortho_2;
                par_1 = par_2;
                dir_1 = dir_2;
                thick_1 = thick_2;
            }

            positions.push_back(pos_1);
            positions.push_back(pos_2);

            orthos.push_back(ortho_1);
            orthos.push_back(ortho_2);

            pars.push_back(par_1);
            pars.push_back(par_2);

            dirs.push_back(dir_1);
            dirs.push_back(dir_2);

            thicknesses.push_back(thick_1);
            thicknesses.push_back(thick_2);

            colors.push_back(c);
            colors.push_back(c);

            // Saves VBO indeces
            strands[i][j].VboIdx.x = counter;
            strands[i][j + 1].VboIdx.y = counter + 1;
            counter += 2;
        }
    }

    VertexBufferObject::DATA *data = new VertexBufferObject::DATA[positions.size()];

    float3 mi = make_float3(math_maxfloat);
    float3 ma = make_float3(math_minfloat);

    for (uint i = 0; i < positions.size(); ++i)
    {
        float3 v = positions[i];
        float3 o = orthos[i];
        float3 p = pars[i];
        float3 d = dirs[i];
        float t = thicknesses[i];
        float4 c = colors[i];

        data[i].vx = v.x;
        data[i].vy = v.y;
        data[i].vz = v.z;
        data[i].vw = 0.0f;

        data[i].cx = c.x;
        data[i].cy = c.y;
        data[i].cz = c.z;
        data[i].cw = 1.f;

        data[i].nx = d.x;
        data[i].ny = d.y;
        data[i].nz = d.z;

        data[i].tx = p.x;
        data[i].ty = p.y;
        data[i].tz = p.z;
        data[i].tw = t;

        data[i].ux = c.x;
        data[i].uy = c.y;
        data[i].uz = c.z;
        data[i].uw = c.w;
    }

    std::shared_ptr<VertexBufferObject> vbo = std::make_shared<VertexBufferObject>();
    vbo->setData(data, GL_DYNAMIC_DRAW, positions.size(), GL_LINES);
    vbo->bindDefaultAttribs();

    VboHair.push_back(vbo);

    delete[] data;
}

void Scene::GenerateHair()
{
    if (VERBOSE)
        printf("---------------------\n");
    if (VERBOSE)
        printf("Generating roots...");
    HairGen->GenerateRoots();
    if (VERBOSE)
        printf(" %i roots generated.\n", HairGen->GetNumRoots());
    if (HairGen->GetNumRoots() != 0)
    {
        if (VERBOSE)
            printf("Generating hair...");
        std::vector<std::vector<HairNode>> strands = HairGen->GenerateStrands();
        BuildVboHair(strands);
        if (VERBOSE)
            printf("built %i points.\n", VboHair.back()->nrDynamicVertices());
    }

    // Deselects all after generation
    SelectMesh(0);

    if (VERBOSE)
        printf("---------------------\n");
}

void Scene::SaveTriangleSelection()
{
    // Writes txt
    ofstream selectionTxt;
    string fileName = "./Export/triangle_selection.txt";
    selectionTxt.open(fileName, ofstream::trunc);

    // Writing
    for (int i = 0; i < Meshes[0]->GetTriangles().size(); ++i)
    {
        SimpleTriangle &t = *Meshes[0]->GetTriangles()[i];
        selectionTxt << i << " " << t.Selected << "\n";
    }

    selectionTxt.close();
}

void Scene::LoadTriangleSelection()
{
    std::fstream in("./Export/triangle_selection.txt");
    std::string line;

    while (std::getline(in, line))
    {
        int idx;
        int opt;
        std::stringstream ss(line);

        ss >> idx;
        ss >> opt;

        Meshes[0]->GetTriangles()[idx]->Selected = opt;
    }

    // Update VBO
    int size = 0;
    VertexBufferObject::DATA *data = VboMeshes[0]->map(size);

#pragma omp parallel for
    for (int i = 0; i < Meshes[0]->GetTriangles().size(); ++i)
    {
        SimpleTriangle &t = *Meshes[0]->GetTriangles()[i];
        bool val = t.Selected;

        // CPU Update
        t.Selected = val;

        // GPU Update
        data[3 * i].tw = val;
        data[3 * i + 1].tw = val;
        data[3 * i + 2].tw = val;
    }

    VboMeshes[0]->unmap();
}

void Scene::GenerateHairGrid()
{
    // Number of cells per dimension
    int nx = SimSolver->GetParamsSim().HairDim.x;
    int ny = SimSolver->GetParamsSim().HairDim.y;
    int nz = SimSolver->GetParamsSim().HairDim.z;

    int nxS = SimSolver->GetParamsSim().SdfDim.x;
    int nyS = SimSolver->GetParamsSim().SdfDim.y;
    int nzS = SimSolver->GetParamsSim().SdfDim.z;

    // Total size of grid embedding obj
    vector<float3> minMax = Meshes[0]->GetAABB();
    float3 distances = minMax[1] - minMax[0];

    // Expands grid around obj
    float scale = SimSolver->GetParamsSim().LengthIncreaseGrid;
    float3 dimIncrease = 0.5f * scale * distances;
    vector<float3> newMinMax = {minMax[0] - dimIncrease, minMax[1] + dimIncrease};

    // Stream info into solver
    float3 newDist = (newMinMax[1] - newMinMax[0]);

    float dx = newDist.x / (1.f * nx);
    float dy = newDist.y / (1.f * ny);
    float dz = newDist.z / (1.f * nz);

    float dxS = newDist.x / (1.f * nxS);
    float dyS = newDist.y / (1.f * nyS);
    float dzS = newDist.z / (1.f * nzS);

    // Before bone transform
    SimSolver->GetParamsSim().HairCenter0 = 0.5 * (newMinMax[1] + newMinMax[0]);
    SimSolver->GetParamsSim().HairAxis0[0] = make_float3(1.f, 0.f, 0.f);
    SimSolver->GetParamsSim().HairAxis0[1] = make_float3(0.f, 1.f, 0.f);
    SimSolver->GetParamsSim().HairAxis0[2] = make_float3(0.f, 0.f, 1.f);
    SimSolver->GetParamsSim().HairOrig0 = newMinMax[0];

    // After bone transform (identity at t0)
    SimSolver->GetParamsSim().HairCenter = 0.5 * (newMinMax[1] + newMinMax[0]);
    SimSolver->GetParamsSim().HairAxis[0] = make_float3(1.f, 0.f, 0.f);
    SimSolver->GetParamsSim().HairAxis[1] = make_float3(0.f, 1.f, 0.f);
    SimSolver->GetParamsSim().HairAxis[2] = make_float3(0.f, 0.f, 1.f);
    SimSolver->GetParamsSim().HairOrig = newMinMax[0];

    // Static params
    SimSolver->GetParamsSim().MaxWeight = 1.5 * max(dx, max(dy, dz)) * (SimSolver->GetParamsSim().NumGridNeighbors + 1.f);
    SimSolver->GetParamsSim().HairInvSqDs = make_float3(1.f / (dx * dx), 1.f / (dy * dy), 1.f / (dz * dz));
    SimSolver->GetParamsSim().HairInvDs = make_float3(1.f / dx, 1.f / dy, 1.f / dz);
    SimSolver->GetParamsSim().HairDs = make_float3(dx, dy, dz);

    // Static params head
    SimSolver->GetParamsSim().SdfInvSqDs = make_float3(1.f / (dxS * dxS), 1.f / (dyS * dyS), 1.f / (dzS * dzS));
    SimSolver->GetParamsSim().SdfInvDs = make_float3(1.f / dxS, 1.f / dyS, 1.f / dzS);
    SimSolver->GetParamsSim().SdfDs = make_float3(dxS, dyS, dzS);

    // cout << "Sanity Check" << endl;
    // printf("Sdf Dim (%i,%i,%i)\n", nxS, nyS, nzS);
    // printf("Extents (%f,%f, %f)  (%f, %f,%f)\n", newMinMax[0].x, newMinMax[0].y, newMinMax[0].z,
    //     newMinMax[1].x, newMinMax[1].y, newMinMax[1].z);
    // printf("Ds (%f,%f,%f)\n", dxS, dyS, dzS);
}


GLuint Scene::LoadImage(const char* filename) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return 0;
    }

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, (channels == 4) ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(data);

    return texture;
}


void Scene::SaveOpenGLImage(const char* filename, int width, int height) {
    // Allocate memory to store the pixel data
    unsigned char* pixels = new unsigned char[width * height * 4]; // 4 channels (RGBA)

    // Read the pixels from the framebuffer (assuming GL_RGBA and GL_UNSIGNED_BYTE)
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    // Flip the image vertically
    for (int y = 0; y < height / 2; ++y) {
        unsigned char* row1 = pixels + y * width * 4;
        unsigned char* row2 = pixels + (height - y - 1) * width * 4;
        std::swap_ranges(row1, row1 + width * 4, row2);
    }

    // Save the image using stb_image_write
    if (stbi_write_png(filename, width, height, 4, pixels, width * 4)) {
        std::cout << "Image saved to " << filename << std::endl;
    } else {
        std::cerr << "Failed to save image." << std::endl;
    }

    // Clean up
    delete[] pixels;
}
