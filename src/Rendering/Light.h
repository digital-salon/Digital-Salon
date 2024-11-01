#ifndef LIGHT_H
#define LIGHT_H

#include "Common.h"
#include "Geometry.h"

class CameraManager;
class VertexBufferObject;
class Shader;
class FrameBufferObject;
class Scene;

class Light
{
public:

    enum Type 
    {
        SPOT_LIGHT = 0, 
        DIRECTIONAL_LIGHT
    };

   Light(Scene *scene, Type type, const vec3 &color, const vec3 &pos, const vec3 &dir = vec3(), const vec3 &intensity = vec3(1.0f, 1.0f, 1.0f), const vec4 &cone = vec4(), vec3 attenuation = vec3());
   ~Light();
    
   void blurShadowMap();
   void renderLightView(mat4 &lightView);

   void render(const Transform &trans);

   void setPosition(const vec3 &pos);
   void move(CameraManager *camManager, float diffX, float diffY);

   void loadPaths();
   void savePaths();
   void recordPath(bool record);
   void autoMove();
   void toggleMode();
   bool hasMoved();

   vec3 position() const;
   vec3 direction() const;
   vec3 intensity() const;
   vec4 cone() const;
   vec3 attenuation() const;
   vec3 color() const;
   Type type() const;

   void setIntensity(float intensity);
   void setDirection(const vec3 &dir);


   void update(float delta);

   GLuint shadowMapId() const;
   GLuint shadowMapBlurredId() const;

   bool m_moved;

private:
	vec3 m_position;
    vec3 m_direction;
    vec3 m_intensity;
    vec4 m_cone;
    vec3 m_attenuation;
    vec3 m_color;

    Type m_type;

	float m_renderSize;

    bool m_record;
    bool m_first;
    bool m_saved;

    uint32_t m_oldTime;

    int m_moveMode;

    float m_height;
    float m_angle;
    float m_radius;

    vec3 m_oldPosition;
    Spline m_spline;
    float m_time;

    float m_speed;
    float m_distance;
    float m_movement;

    VertexBufferObject *m_vbo;
    VertexBufferObject *m_vboBlur;

    GLuint m_shadowMapID;
    GLuint m_shadowMapBlurredID;

public:
    FrameBufferObject *m_fboLight;
    FrameBufferObject *m_fboBlurV;
    FrameBufferObject *m_fboBlurH;   

    int m_bufferWidth;
    int m_bufferHeight;
    float m_fcpLight;
    float m_ncpLight;
    float m_fovLight;

    Scene *m_scene;

    mat4 m_lightView;
};

#endif

