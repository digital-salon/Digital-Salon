#ifndef ENGINE_H
#define ENGINE_H

#include "Common.h"
#include "Renderer.h"
#include "Scene.h"
#include "GUI.h"
#include "CameraManager.h"
#include "Shader.h"

class CameraManager;
class Scene;
class Renderer;
class GUI;

class Engine
{
public:
    Engine();

    void Run();
    void LoadScene(const HairInfo& hairInfo, const HeadInfo& headInfo, ExpOpt opt = NONE);
    //void AddObjects(const vector<string>& objNames, const vector<ObjParamsCPU>& params, const vector<int3>& sizes);
    void InitWindow(const HairInfo& hairInfo, const HeadInfo& headInfo, ExpOpt opt = NONE);
    void InitShaders();
    void InitParams();
    void DrawGUI();
    void updateShaders();

    static void error_callback(int error, const char* description);
    static void mouse_callback(GLFWwindow* window, double x, double y);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void resize(GLFWwindow* window, int x, int y);
    static void setupWindow(GLFWwindow* window);
    

public:
    CameraManager* m_cameraManager;
    Scene* m_scene;
    GUI* m_gui;
    Renderer* m_renderer;
    GLFWwindow* Window;

    HPTimer m_timer;

    // imgui
    bool OptDockingFullscreen = true;
    bool OptPadding = false;
    ImGuiDockNodeFlags DockspaceFlags;
};

#endif