#include "CameraManager.h"
#include "Light.h"

CameraManager::CameraManager()
: m_active(0),
  m_width(1.0f),
  m_height(1.0f),
  m_aspect(1.0f),
  m_mouseSensivity(0.2f),
  m_camSpeed(10.0f),
  m_useCam(false),
  m_rotate(0.f, 0.f, 0.f),
  m_translate(0.0f, 0.0f, 0.0f),
  m_zoom(-50.0),
  m_fov(params::inst()->fov),
  m_ncp(params::inst()->ncp),
  m_fcp(params::inst()->fcp),
  m_rotHeight(0.0f)
{
	Camera *cam1 = new Camera(vec3(0, 2, 12), 0, 0, 0, m_fov, m_ncp, m_fcp);
	cam1->setColor(1.0f, 0.0f, 0.0f);
	cam1->setSpeed(m_camSpeed);
	cam1->update();
    //cam1->loadFrameDirectory("Data/Camera");
    //cam1->loadFramesFromProcedure();

	m_cameras.push_back(cam1);
}

CameraManager::~CameraManager()
{
	for(uint i=0; i<m_cameras.size(); ++i)
	{
		Camera *cam = m_cameras[i];
		delete cam;
	}
}

void CameraManager::currentPerspective(Transform &trans)
{
	if(m_useCam)
	{
        m_cameras[m_active]->determineMovement();        
		m_cameras[m_active]->update();		     
        m_cameras[m_active]->interpolate();
		m_cameras[m_active]->setPerspective(trans);
	}
	else
	{
        m_cameras[m_active]->determineMovement();        
		m_cameras[m_active]->update();	
        m_cameras[m_active]->interpolate();

		m_aspect = params::inst()->window_x / (float)params::inst()->window_y;
		mat4 projection = mat4::perspective(m_fov, m_aspect, m_ncp, m_fcp);
		mat4 view = mat4::identitiy();

		view = mat4::translate(vec3(0, m_rotHeight, 0));
		view *= mat4::translate(vec3(0, 0, m_zoom));
		view *= mat4::translate(m_translate);
		view *= mat4::rotate(m_rotate.x, 1, 0, 0);
		view *= mat4::rotate(m_rotate.y, 0, 1, 0); 

		trans.projection = projection;
		trans.view = view;
		trans.viewProjection = projection * view;
        trans.normal = trans.viewProjection.inverse().transpose();
	}
}

void CameraManager::resize(float width, float height)
{
	m_aspect = width / height;

	for(uint i=0; i<m_cameras.size(); ++i)
	{
		m_cameras[i]->updateCamInternals(m_aspect);
		m_cameras[i]->update();
	}
}

void CameraManager::update()
{	
	if (inputs::inst()->f2) {
		inputs::inst()->f2 = false;
		m_useCam = !m_useCam;
	}

	if (inputs::inst()->right_button && !inputs::inst()->alt && !inputs::inst()->a) {
		onMouseMove(inputs::inst()->mouse_diff.x, inputs::inst()->mouse_diff.y);
	}

	if(inputs::inst()->left_button) {
		changeRotHeight(-inputs::inst()->scroll_y);
	}
	else {
		onMouseWheel((int)inputs::inst()->scroll_y);
	}

	onKeyPress();

    m_cameras[m_active]->determineMovement();
	m_cameras[m_active]->update();
}

void CameraManager::renderCameras(const Transform &trans)
{
	if(!m_useCam)
	{
		for(uint i=0; i<m_cameras.size(); ++i)
		{		
            m_cameras[i]->update();
			m_cameras[i]->render(trans);
		}
	}
}

void CameraManager::toggleCam()
{
    if(!m_useCam)
    {
        m_useCam = true;   
    }
    else
    {
        m_active ++;

        if(m_active > (int)m_cameras.size()-1)
        {
            m_active = 0;
            m_useCam = false;
        }
    }
}

void CameraManager::onMouseMove(float dx, float dy)
{
	if (!inputs::inst()->key_pressed)
	{
		if (m_useCam)
		{
			m_cameras[m_active]->changeHeading(m_mouseSensivity * dx);
			m_cameras[m_active]->changePitch(m_mouseSensivity * dy);
		}
		else
		{
			m_rotate.y -= (0.5f * dx);
			m_rotate.x -= (0.5f * dy);
		}

		//params::inst()->lights[0]->m_moved = true;
	}
}

void CameraManager::onMouseWheel(int dir)
{
    float delta =  m_zoom * -0.1f;

	if (dir > 0) 
		m_zoom += delta;			
    else if (dir < 0)
	    m_zoom -= delta;	
}

void CameraManager::onKeyPress()
{
	if (inputs::inst()->w && inputs::inst()->ctrl)
	{
		//m_cameras[m_active]->moveForward(true);
		m_translate.y -= 0.1f;
	}

	if (inputs::inst()->s && inputs::inst()->ctrl)
	{
		//m_cameras[m_active]->moveForward(true);
		m_translate.y += 0.1f;
	}

	if (inputs::inst()->a && inputs::inst()->ctrl)
	{
		//m_cameras[m_active]->moveForward(true);
		m_translate.x += 0.1f;
	}

	if (inputs::inst()->d && inputs::inst()->ctrl)
	{
		//m_cameras[m_active]->moveForward(true);
		m_translate.x -= 0.1f;
	}

	if(inputs::inst()->s)
		m_cameras[m_active]->moveBackward(true);

	if (inputs::inst()->a)
		m_cameras[m_active]->strafeLeft(true);

	if (inputs::inst()->d)
		m_cameras[m_active]->strafeRight(true);

}

void CameraManager::increaseSpeed()
{
    m_camSpeed *= 2;

    for(uint i=0; i<m_cameras.size(); ++i)
    {
        m_cameras[i]->setDistPerSec(m_camSpeed);
    }
}

void CameraManager::decreaseSpeed()
{
    m_camSpeed /= 2;

    for(uint i=0; i<m_cameras.size(); ++i)
    {
        m_cameras[i]->setDistPerSec(m_camSpeed);
    }
}

vec3 CameraManager::lodCamPosition()
{
    return m_cameras[0]->position();
}

Camera *CameraManager::currentCam()
{
    return m_cameras[m_active];
}

void CameraManager::toggleInterpolation()
{
    m_cameras[0]->toggleInterpolation();
}

void CameraManager::addFrame()
{
    m_cameras[0]->autoAddFrame();
}

void CameraManager::clearFrameset()
{
    m_cameras[0]->clearFrames();
}

//void CameraManager::saveFrameset()
//{
//    m_cameras[0]->saveFrames();
//}
//
//void CameraManager::toggleFrameset()
//{
//    m_cameras[0]->changeFrameSet();
//}

//QString CameraManager::currentFramesetName()
//{
//    return m_cameras[0]->frameSetName();
//}

Camera *CameraManager::lodCamera()
{
    return m_cameras[0];
}

std::vector<Camera*> CameraManager::cameras()
{
    return m_cameras;
}

vec3 CameraManager::currentCamPos()
{
	if(m_useCam)
	{
		return lodCamPosition();
	} 
	else
	{
		mat4 view = mat4::identitiy();

		view = mat4::translate(vec3(0, 0, m_zoom));
		view *= mat4::rotate(m_rotate.x, 1, 0, 0);
		view *= mat4::rotate(m_rotate.y, 0, 1, 0); 

		vec4 camPos = view.inverse() * vec4(0.0, 0.0, 0.0, 1.0);

		return vec3(camPos.x, camPos.y, camPos.z);
	}
}

float CameraManager::currentCamFov()
{
	if(m_useCam)
	{
        return m_cameras[m_active]->fov();
    }
    else
    {
        return m_fov;
    }
}

float CameraManager::currentCamNcp()
{
    if(m_useCam)
	{
        return m_cameras[m_active]->ncp();
    }
    else
    {
        return m_ncp;
    }
}

float CameraManager::currentCamFcp()
{
    if(m_useCam)
	{
        return m_cameras[m_active]->fcp();
    }
    else
    {
        return m_fcp;
    }
    
}

void CameraManager::currentCamParams()
{
    if(m_useCam)
    {
        params::inst()->fov = m_cameras[m_active]->fov();
        params::inst()->ncp = m_cameras[m_active]->ncp();
        params::inst()->fcp = m_cameras[m_active]->fcp();
        params::inst()->camPos = lodCamPosition();
    }
    else
    {
        params::inst()->fov = m_fov;
        params::inst()->ncp = m_ncp;
        params::inst()->fcp = m_fcp;

		mat4 view = mat4::identitiy();

		view = mat4::translate(vec3(0, 0, m_zoom));
		view *= mat4::rotate(m_rotate.x, 1, 0, 0);
		view *= mat4::rotate(m_rotate.y, 0, 1, 0); 

		vec4 camPos = view.inverse() * vec4(0.0, 0.0, 0.0, 1.0);

        params::inst()->camPos = vec3(camPos.x, camPos.y, camPos.z);
    }
}

void CameraManager::changeRotHeight(float delta)
{
	m_rotHeight += delta;
}

void CameraManager::lockCurCamera()
{
    if (m_useCam)
        m_cameras[m_active]->lock();
}