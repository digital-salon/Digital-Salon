#ifndef RENDERER_H
#define RENDERER_H

#include "Common.h"

class GUI;
class CameraManager;
class FrameBufferObjectMultisample;
class FrameBufferObject;
class Shader;
class Scene;
class VertexBufferObject;
class Texture;

class Renderer
{
public:
    Renderer(Scene *scene, CameraManager *camManager, GUI *gui);
    ~Renderer();

    void init();
    void render(Transform &trans);
    
    void setBackgroundColor();
    void togglePolygonMode();

private:
    void renderScene(const Transform &trans);

private:
   //GUI *m_gui;
    CameraManager *m_cameraManager;
	Scene *m_scene;

    vec4 m_bgColor; 

    const GLuint m_samples;
    GLuint m_bgMode;
};

#endif

 