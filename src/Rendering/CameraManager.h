#ifndef CAMERAMANAGER_H
#define CAMERAMANAGER_H

#include "Common.h"
#include "Camera.h"

class CameraManager
{
public:
   CameraManager();
   ~CameraManager();

   void currentPerspective(Transform &trans);
   void resize(float width, float height);
   void onMouseMove(float dx, float dy);
   void onKeyPress();
   void update();
   void onMouseWheel(int dir);
   void toggleCam();
   void renderCameras(const Transform &trans);
   void increaseSpeed();
   void decreaseSpeed();
   vec3 lodCamPosition();
   void toggleInterpolation();
   void addFrame();
   void clearFrameset();
   //void saveFrameset();
   //void toggleFrameset();
   //QString currentFramesetName();
   Camera *lodCamera();
   std::vector<Camera*> cameras();
   Camera *currentCam();
   vec3 currentCamPos();  
   float currentCamFov();
   float currentCamNcp();
   float currentCamFcp();
   void currentCamParams();
   void changeRotHeight(float delta);
   void lockCurCamera();

private:
	
	std::vector<Camera *> m_cameras;

	int m_active;
	float m_width;
	float m_height;
	float m_aspect;
	float m_mouseSensivity;
	float m_camSpeed;
	bool  m_useCam;


    vec3  m_rotate;
    vec3  m_translate;
	float m_zoom;
    float m_fov;
    float m_ncp;
    float m_fcp;
	float m_rotHeight;
};

#endif

