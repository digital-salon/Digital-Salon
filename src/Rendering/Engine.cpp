#include "Engine.h"
#include "Common.h"
#include "CameraManager.h"

Engine::Engine() 
: m_scene(nullptr),
  m_gui(nullptr),
  m_renderer(nullptr),
  m_cameraManager(nullptr)
{
}

void Engine::Run()
{
    Transform trans;

    // Setup IMGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    // io.FontGlobalScale = 1.5f; // Set-up font size for ImGui
    io.FontGlobalScale = 1.0f; // Set-up font size for ImGui
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
    ImGui::StyleColorsDark();

    // Load a custom font with a specific size
    ImFontConfig fontConfig;
    // fontConfig.SizePixels = 34.0f; // Set your desired font size here
    fontConfig.SizePixels = 20.0f; // Set your desired font size here

    // Load the font from a file and specify the size
    io.Fonts->AddFontFromFileTTF("../data/test_font.ttf", fontConfig.SizePixels, &fontConfig);


    ImGui_ImplGlfw_InitForOpenGL(Window, true);
    ImGui_ImplOpenGL3_Init();
    DockspaceFlags = ImGuiDockNodeFlags_None | ImGuiDockNodeFlags_PassthruCentralNode;

    while (!glfwWindowShouldClose(Window))
    {
        glfwSetInputMode(Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        m_scene->UpdateSimulation();
        m_cameraManager->currentPerspective(trans);
        m_cameraManager->currentCamParams();
        m_cameraManager->update();
        m_scene->Update(trans, 0.01f);
        m_renderer->render(trans);

        // Embedding
        m_scene->UpdateEmbedded();

        DrawGUI();

        updateShaders();
        clearInputs();

        glfwSwapBuffers(Window);
        glfwPollEvents();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(Window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

void Engine::LoadScene(const HairInfo& hairInfo, const HeadInfo& headInfo, ExpOpt opt)
{
    // random seed
    srand(time(NULL)); 

    // Initialize Params
    InitParams();

    // Window Set-up
    InitWindow(hairInfo, headInfo, opt);

    // Load Shaders
    InitShaders();
}

//void Engine::AddObjects(const vector<string>& objNames, const vector<ObjParamsCPU>& params, const vector<int3>& sizes)
//{
//    m_scene->AddObjectsScene(objNames, params, sizes);
//}

void Engine::InitWindow(const HairInfo& hairInfo, const HeadInfo& headInfo, ExpOpt opt)
{
    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    int argc = 1;
    char* argv[1] = { (char*)"Something" };
    glutInit(&argc, argv);

    glfwWindowHint(GLFW_SAMPLES, 16);

    Window = glfwCreateWindow(params::inst()->window_x, params::inst()->window_y, "Digital Salon", NULL, NULL);

    if (!Window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glewExperimental = GL_TRUE;

    glfwMakeContextCurrent(Window);
    glfwSwapInterval(1);
    glfwSetInputMode(Window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glfwSetKeyCallback(Window, key_callback);
    glfwSetMouseButtonCallback(Window, mouse_button_callback);
    glfwSetCursorPosCallback(Window, mouse_callback);
    glfwSetFramebufferSizeCallback(Window, resize);
    glfwSetScrollCallback(Window, scroll_callback);

    GLenum err = glewInit();
    if (GLEW_OK != err) {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    }

    m_cameraManager = new CameraManager();
    m_scene = new Scene(m_cameraManager, hairInfo, headInfo, opt);
    m_gui = new GUI(m_cameraManager, m_scene);
    m_renderer = new Renderer(m_scene, m_cameraManager, m_gui);
    params::inst()->lights = m_scene->m_lights;
    m_timer.reset();
}

void Engine::error_callback(int error, const char* description)
{
    fputs(description, stderr);
}

void Engine::mouse_callback(GLFWwindow* window, double x, double y)
{
    inputs::inst()->mouse_cur = vec2((float)x, (float)y);
    inputs::inst()->mouse_diff = inputs::inst()->mouse_cur - inputs::inst()->mouse_prev;
    inputs::inst()->mouse_prev = inputs::inst()->mouse_cur;
}

void Engine::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    //cout << key << " " << scancode << " " << action << " " << mods << endl;
    if (key == 256) {
        exit(0);
    }

    //Alt 
    if (key == 342 && action == 1) {
        inputs::inst()->alt = true;
    } 

    if (key == 342 && action == 0) {
        inputs::inst()->alt = false;
    }

    //Ctrl
    if (key == 341 && action == 1) {
        inputs::inst()->ctrl = true;
    }

    if (key == 341 && action == 0) {
        inputs::inst()->ctrl = false;
    }

    //A
    if (key == 65 && action >= 1) {
        inputs::inst()->a = true;
    }

    if (key == 65 && action == 0) {
        inputs::inst()->a = false;        
    }

    //D
    if (key == 68 && action >= 1) {
        inputs::inst()->d = true;
    }

    if (key == 68 && action == 0) {
        inputs::inst()->d = false;
    }

    //S
    if (key == 83 && action >= 1) {
        inputs::inst()->s = true;
    }

    if (key == 83 && action == 0) {
        inputs::inst()->s = false;
    }

    //W
    if (key == 87 && action >= 1) {
        inputs::inst()->w = true;
        inputs::inst()->key_pressed = true;
    }

    if (key == 87 && action == 0) {
        inputs::inst()->w = false;
        inputs::inst()->key_pressed = false;
    }

    //G
    if (key == 71 && action >= 1) {
        inputs::inst()->g = true;
        inputs::inst()->key_pressed = true;
    }

    if (key == 71 && action == 0) {
        inputs::inst()->g = false;
        inputs::inst()->key_pressed = false;
    }

    //B
    if (key == 66 && action >= 1) {
        inputs::inst()->b = true;
        inputs::inst()->key_pressed = true;
    }

    if (key == 66 && action == 0) {
        inputs::inst()->b = false;
        inputs::inst()->key_pressed = false;
    }

    //N
    if (key == 78 && action >= 1) {
        inputs::inst()->n = true;
        inputs::inst()->key_pressed = true;
    }

    if (key == 78 && action == 0) {
        inputs::inst()->n = false;
        inputs::inst()->key_pressed = false;
    }

    //R
    if (key == 82 && action >= 1) {
        inputs::inst()->r = true;
        inputs::inst()->key_pressed = true;
    }

    if (key == 82 && action == 0) {
        inputs::inst()->r = false;
        inputs::inst()->key_pressed = false;
    }

    //H
    if (key == 72 && action >= 1) {
        inputs::inst()->h = true;
        inputs::inst()->key_pressed = true;
    }

    if (key == 72 && action == 0) {
        inputs::inst()->h = false;
        inputs::inst()->key_pressed = false;
    }

    //P
    if (key == 80 && action >= 1) {
        inputs::inst()->p = true;
        inputs::inst()->key_pressed = true;
    }

    if (key == 80 && action == 0) {
        inputs::inst()->p = false;
        inputs::inst()->key_pressed = false;
    }

    //O
    if (key == 79 && action >= 1) {
        inputs::inst()->o = true;
        inputs::inst()->key_pressed = true;
    }

    if (key == 79 && action == 0) {
        inputs::inst()->o = false;
        inputs::inst()->key_pressed = false;
    }

    //T
    if (key == 84 && action >= 1) {
        inputs::inst()->t = true;
    }

    if (key == 84 && action == 0) {
        inputs::inst()->t = false;
    }

    //I
    if (key == 73 && action >= 1) {
        inputs::inst()->i = true;
    }

    if (key == 73 && action == 0) {
        inputs::inst()->i = false;
    }

    //F2
    if (key == 291 && action == 1) {
        inputs::inst()->f2 = true;
    }

    if (key == 291 && action == 0) {
        inputs::inst()->f2 = false;
    }

    //F4
    if (key == 293 && action == 1) {
        loop<int>(params::inst()->background_mode, 0, 2, 1);
    }

    //F5
    if (key == 293 && action == 1) {
        loop<int>(params::inst()->ui_mode, 0, 2, 1);
    }
}

void Engine::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == 0 && action == 1) {
        inputs::inst()->left_button = true;
    }

    if (button == 1 && action == 1) {
        inputs::inst()->right_button = true;
    }

    if (button == 0 && action == 0) {
        inputs::inst()->left_button = false;
        inputs::inst()->gui_clicked = false;
    }

    if (button == 1 && action == 0) {
        inputs::inst()->right_button = false;
    }

    //cout << button << " " << action << " " << mods << endl;
}

void Engine::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    inputs::inst()->scroll_x = (float)xoffset;
    inputs::inst()->scroll_y = (float)yoffset;
}

void Engine::resize(GLFWwindow* window, int x, int y)
{
    glfwGetFramebufferSize(window, &params::inst()->window_x, &params::inst()->window_y);
}

void Engine::setupWindow(GLFWwindow* window)
{
}

void Engine::InitShaders()
{
    shaders::inst()->extraObj = new Shader("../shaders/Extra.vert.glsl", "../shaders/Extra.frag.glsl");
    shaders::inst()->extraObj->bindAttribLocations();

    shaders::inst()->axis = new Shader("../shaders/Axis.vert.glsl", "../shaders/Axis.frag.glsl");
    shaders::inst()->axis->bindAttribLocations();

    shaders::inst()->standard = new Shader("../shaders/Default.vert.glsl", "../shaders/Default.frag.glsl");
    shaders::inst()->standard->bindAttribLocations();

    shaders::inst()->gridBox = new Shader("../shaders/SDFBox.vert.glsl", "../shaders/SDFBox.frag.glsl");
    shaders::inst()->gridBox->bindAttribLocations();

    shaders::inst()->defaultLight = new Shader("../shaders/DefaultLight.vert.glsl", "../shaders/DefaultLight.frag.glsl");
    shaders::inst()->defaultLight->bindAttribLocations();

    shaders::inst()->defaultDepth = new Shader("../shaders/Default.vert.glsl", "../shaders/DefaultDepth.frag.glsl");
    shaders::inst()->defaultDepth->bindAttribLocations();

    shaders::inst()->blur = new Shader("../shaders/Blur.vert.glsl", "../shaders/Blur.frag.glsl");
    shaders::inst()->blur->bindAttribLocations();

    shaders::inst()->grid = new Shader("../shaders/NiceGrid.vert.glsl", "../shaders/NiceGrid.frag.glsl");
    shaders::inst()->grid->bindAttribLocations();

    shaders::inst()->object = new Shader("../shaders/Object.vert.glsl", "../shaders/Object.frag.glsl");
    shaders::inst()->object->bindAttribLocations();

    shaders::inst()->objectDepth = new Shader("../shaders/ObjectDepth.vert.glsl", "../shaders/ObjectDepth.frag.glsl");
    shaders::inst()->objectDepth->bindAttribLocations();

    shaders::inst()->objectLines = new Shader("../shaders/ObjectLines.vert.glsl", "../shaders/ObjectLines.frag.glsl");
    shaders::inst()->objectLines->bindAttribLocations();

    shaders::inst()->gui = new Shader("../shaders/GUI.vert.glsl", "../shaders/GUI.frag.glsl");
    shaders::inst()->gui->bindAttribLocations();

    shaders::inst()->noise = new Shader("../shaders/Noise.vert.glsl", "../shaders/Noise.frag.glsl");
    shaders::inst()->noise->bindAttribLocations();

    shaders::inst()->cookTorrance = new Shader("../shaders/CookTorrance.vert.glsl", "../shaders/CookTorrance.frag.glsl");
    shaders::inst()->cookTorrance->bindAttribLocations();

    shaders::inst()->sphericalHarmonic = new Shader("../shaders/SphericalHarmonic.vert.glsl", "../shaders/SphericalHarmonic.frag.glsl");
    shaders::inst()->sphericalHarmonic->bindAttribLocations();

    shaders::inst()->tessellation = new Shader();
    shaders::inst()->tessellation->attachVertexShader("../shaders/TessInterp.vert.glsl");
    shaders::inst()->tessellation->attachControlShader("../shaders/TessInterp.cont.glsl");
    shaders::inst()->tessellation->attachEvaluationShader("../shaders/TessInterp.eval.glsl");
    shaders::inst()->tessellation->attachGeometryShader("../shaders/TessInterp.geom.glsl");
    shaders::inst()->tessellation->attachFragmentShader("../shaders/TessInterp.frag.glsl");
    shaders::inst()->tessellation->bindAttribLocations();

    shaders::inst()->hair = new Shader();
    shaders::inst()->hair->attachVertexShader("../shaders/Hair.vert.glsl");
    shaders::inst()->hair->attachGeometryShader("../shaders/Hair.geom.glsl");
    shaders::inst()->hair->attachFragmentShader("../shaders/Hair.frag.glsl");
    shaders::inst()->hair->bindAttribLocations();

    shaders::inst()->hairNew = new Shader();
    shaders::inst()->hairNew->attachVertexShader("../shaders/HairNew.vert.glsl");
    shaders::inst()->hairNew->attachFragmentShader("../shaders/HairNew.frag.glsl");
    shaders::inst()->hairNew->bindAttribLocations();

    shaders::inst()->hairDepth = new Shader();
    shaders::inst()->hairDepth->attachVertexShader("../shaders/Hair.vert.glsl");
    shaders::inst()->hairDepth->attachGeometryShader("../shaders/Hair.geom.glsl");
    shaders::inst()->hairDepth->attachFragmentShader("../shaders/HairDepth.frag.glsl");
    shaders::inst()->hairDepth->bindAttribLocations();

    shaders::inst()->head = new Shader("../shaders/Head.vert.glsl", "../shaders/Head.frag.glsl");
    shaders::inst()->head->bindAttribLocations();

    shaders::inst()->MeshDepth = new Shader("../shaders/Head.vert.glsl", "../shaders/HeadDepth.frag.glsl");
    shaders::inst()->MeshDepth->bindAttribLocations();

    shaders::inst()->headWireframe = new Shader("../shaders/HeadWireframe.vert.glsl", "../shaders/HeadWireframe.frag.glsl");
    shaders::inst()->headWireframe->bindAttribLocations();
}

void Engine::InitParams()
{
    Params* p = params::inst();

    p->camPos = vec3(0.0f, 0.0f, 0.0f);
    p->blur = vec2(2.0f, 2.0f);
    p->shadowMapSize = vec2(2048, 2048);
    p->applyShadow = true;
    p->gridRenderMode = 0;
    p->shadowIntensity = 0.4f;
    p->lightIntensity = 2.2f;

    p->polygonMode = 0;
    p->activeLight = 0;

    p->window_x = 1920;
    p->window_y = 1080;

    p->renderMesh = false;
    p->renderObjects = true;
    p->renderTextures = false;
    p->renderWireframe = false;
    p->renderNormals = false;
    p->renderMisc = false;

    p->ncp = 0.1f;
    p->fcp = 1000.0f;
    p->fov = 45.0f;

    p->polygonOffsetUnits = 1.0f;
    p->polygonOffsetFactor = 0.5f;
    p->depthRangeMax = 1.0f;
    p->depthRangeMin = 0.0f;

    p->nrVertices = 0;
    p->nrActiveVertices = 0;
    p->background_mode = 1;
    p->ui_mode = 0;
}

void Engine::DrawGUI()
{

        // Start the ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();


    // Docking
    bool p_open = false;

    // We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
    // because it would be confusing to have two docking targets within each others.
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking; // | ImGuiWindowFlags_MenuBar;

    if (OptDockingFullscreen)
    {
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    }
    else
    {
        DockspaceFlags &= ~ImGuiDockNodeFlags_PassthruCentralNode;
    }

    //m_dockspace_flags |= ImGuiDockNodeFlags_PassthruCentralNode;

    // When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
    // and handle the pass-thru hole, so we ask Begin() to not render a background.
    if (DockspaceFlags & ImGuiDockNodeFlags_PassthruCentralNode)
    {
        window_flags |= ImGuiWindowFlags_NoBackground;
        //ImGui::SetNextWindowBgAlpha(0);  // redundant?
    }

    // Important: note that we proceed even if Begin() returns false (aka window is collapsed).
    // This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
    // all active windows docked into it will lose their parent and become undocked.
    // We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
    // any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
    if (!OptPadding) ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    ImGui::Begin("DockSpace", &p_open, window_flags);

    if (!OptPadding) ImGui::PopStyleVar();
    if (OptDockingFullscreen) ImGui::PopStyleVar(2);

    if (OptDockingFullscreen)
    {
        ImGui::SetNextWindowBgAlpha(0);
        //ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.0f);
        //  //ImGui::PushStyleColor(ImGuiCol_WindowBg, IM_COL32(1, 0, 0, 0.5)); //ImVec4(0.25f, 0.25f, 0.25f, 0.75f);
        //  //ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(1, 0, 0, 0.5));

        //ImGui::SetNextWindowBgAlpha(0);
    }

    // DockSpace
    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
    {
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), DockspaceFlags);
    }

    // Scene menus
    m_scene->SceneDrawGUI();
    m_scene->EmbeddedGUI();

    ImGui::End();

    // Render
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Engine::updateShaders() {

    if (m_timer.time() > 1.0) {
        m_timer.reset();

        shaders::inst()->extraObj->checkFileUpdate();
        shaders::inst()->standard->checkFileUpdate();
        shaders::inst()->gridBox->checkFileUpdate();
        shaders::inst()->grid->checkFileUpdate();
        shaders::inst()->hair->checkFileUpdate();
        shaders::inst()->hairNew->checkFileUpdate();
        shaders::inst()->object->checkFileUpdate();
        shaders::inst()->head->checkFileUpdate();
        shaders::inst()->headWireframe->checkFileUpdate();
        shaders::inst()->axis->checkFileUpdate();
    }
}