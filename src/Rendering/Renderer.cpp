#include "Renderer.h"
#include "GUI.h"
#include "CameraManager.h"
#include "FrameBufferObject.h"
#include "Shader.h"
#include "VertexBufferObject.h"
#include "Scene.h"
#include "Mesh.h"
#include "Light.h"
#include "Texture.h"

Renderer::Renderer(Scene *scene, CameraManager *camManager, GUI *gui)
: m_scene(scene),
  //m_gui(gui),
  m_cameraManager(camManager),
  m_bgColor(0.5f, 0.1f, 0.1f, 1.0f),
  m_samples(16),
  m_bgMode(1)
{
    init();
}

Renderer::~Renderer()
{
}

void Renderer::init()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
}

void Renderer::render(Transform &trans)
{
    if(params::inst()->applyShadow)
    {
        trans.lightViews.clear();
        trans.lightViews.resize(m_scene->m_lights.size());

        for(int i=0; i<m_scene->m_lights.size(); ++i)
        {
            m_scene->m_lights[i]->setIntensity(params::inst()->lightIntensity);
            m_scene->m_lights[i]->setDirection(m_scene->m_lights[i]->position());
            m_scene->m_lights[i]->renderLightView(trans.lightViews[i]); 
        }
    }    

    renderScene(trans);	

    //if(params::inst()->renderTextures)
    //{
    //    for(int i=0; i<m_scene->m_lights.size(); ++i)
    //    {
    //        renderTexture(m_scene->m_lights[i]->m_fboLight->texAttachment(GL_COLOR_ATTACHMENT0), 220+i*210, m_height-200, 200, 200);
    //    }
    //}

    //m_gui->render();
}

void Renderer::renderScene(const Transform &trans)
{
    setBackgroundColor();

    glViewport(0, 0, params::inst()->window_x, params::inst()->window_y);
    glClearColor(m_bgColor.x, m_bgColor.y, m_bgColor.z, m_bgColor.w);    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);     
    glEnable(GL_MULTISAMPLE);        

    if (params::inst()->polygonMode == 1)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    		    
    m_scene->RenderObjects(trans);
    m_scene->RenderWorld(trans);    

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void Renderer::setBackgroundColor()
{
    if(params::inst()->background_mode == 0)
	{
        m_bgColor = vec4(0.1f, 0.1f, 0.1f, 1.0f);
		//m_gui->setFontColor(vec4(0.9f, 0.9f, 0.9f, 1.0f));
	}
   
    if(params::inst()->background_mode == 1)
	{
        m_bgColor = vec4(0.7f, 0.7f, 0.7f, 1.0f);
		//m_gui->setFontColor(vec4(0.0f, 0.0f, 0.0f, 1.0f));
	}

    if(params::inst()->background_mode == 2)
	{
        m_bgColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
		//m_gui->setFontColor(vec4(0.0f, 0.0f, 0.0f, 1.0f));
	}
}

