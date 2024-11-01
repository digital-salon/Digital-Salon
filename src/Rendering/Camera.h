#ifndef CAMERA_H
#define CAMERA_H

#include "Common.h"
#include <math.h>

class Frustum;
class Spline;

class Camera
{
private:
	float m_heading;
	float m_pitch;
    float m_roll;
	float m_forwardSpeed;
    float m_strafeSpeed;

    Quaternion m_qHeading;
	Quaternion m_qPitch;
    Quaternion m_qRoll;

	vec3 m_pos;
	vec3 m_dir;
    vec3 m_realPos;
    vec3 m_strafe;
    vec3 m_up;
    vec3 m_tempDir;

    vec3 m_cameraColor;

    vec3 m_fixUp;
    vec3 m_fixRight;
    vec3 m_fixDir;

    float m_matrix1[16];
    float m_matrix2[16];

    float m_lerpFactor;
    float m_ncp;
    float m_fcp;
    float m_fov;
	float m_aspect;

    float m_moveSpeed;
    float m_time;

    Frustum *m_frustum;
	Frustum *m_showFrustum;

    bool m_renderFrustum;
    bool m_interpolate;
    bool m_locked;

    struct CameraFrame
    {
        vec3 pos;
        float  headingDeg;
        float  pitchDeg;
		float  desiredDistance;
    };
    
    HPTimer m_hpTimer;

    float m_timer;
    float m_timerDiff; 
    float m_secsPerFrame;
    float m_desiredDistance;
    float m_movementValue;

    vector<float> m_secsPerFrameList;

    Spline *m_spline;
    Spline *m_splineView;
	Spline *m_splineSpeed;

    uint m_idxInterpolNew;
    uint m_idxInterpolOld;

    std::vector<CameraFrame> m_camFrames;

	//QFileInfoList m_frameFileNames;
	//int m_activeFrameSet;
	//QString m_activeFrameSetName;
	//QString m_frameSetFolder;

public:
	Camera(vec3 pos, float heading, float pitch, float roll, float fov, float ncp, float fcp);
	virtual ~Camera();

    void moveForward(bool move);
    void moveBackward(bool move);
    void strafeLeft(bool move);
    void strafeRight(bool move);
    void increaseDistPerSec(float delta);
    void decreaseDistPerSec(float delta);
    void setDistPerSec(float value);

	void changeHeading(float degrees);
	void changePitch(float degrees);
    void changeRoll(float degrees);

	void setPerspective(Transform &trans);
    void render(const Transform &trans);
    void update();
    vec3 position() const;
    vec3 direction() const;
    vec3 up() const;
    vec3 right() const;
    float pitch() const;
    float heading() const;
    float roll() const;
    float fov() const;
    float ncp() const;
    float fcp() const;

    void interpolate(float speed);
    void splineInterpolation();
    void updateCamInternals(float ratio);
    void renderFrames();
    void renderSpline();
    void setPosition(float x, float y, float z);
    void setColor(float r, float g, float b);
	void setHeading(float degrees);
	void setPitch(float degrees);
	void setSpeed(float s);
    
    void addFrame(vec3 pos, float heading, float pitch, float desiredDistance);
    void autoAddFrame();    
    void clearFrames();
 //   void saveFrames(QString filePath);
	//void saveFrames();
 //   void loadFrames(QString filePath);
	//void loadFrameDirectory(QString folderPath);
	//void changeFrameSet();
    void toggleInterpolation();
 //   void loadFramesFromProcedure();

    Frustum *frustum();

    void determineMovement();
    void interpolate();
	//QString frameSetName() const;
    void lock();
};

#endif