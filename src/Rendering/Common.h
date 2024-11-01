#ifndef HEADERS_H
#define HEADERS_H

#include "Global.h"
#include "math.h"

#include "Geometry.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/freeglut.h>

#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

#define VERBOSE true

class Shader;
class Light;

using namespace std;

#undef min
#undef max


struct Transform
{
    mat4 view;
    mat4 projection;    
    mat3 normal;
    mat4 viewProjection;
    mat4 modelViewProjection;
    vector<mat4> lightViews;
};

struct Params
{
    int window_x = 4000;
    int window_y = 4000;
    
    int   gridRenderMode = 0;
    int   polygonMode = 0;

    bool  applyShadow = true;
    bool  renderMesh = false;
    bool  renderObjects = false; 
    bool  renderTextures = false;
    bool  renderWireframe = false;
    bool  renderNormals = false;
    bool  renderMisc = false;

    float ncp = 0.0f;
    float fcp = 0.0f;
    float fov = 0.0f;
    float lightIntensity = 0.0f;
    float shadowIntensity = 0.0f;

    vec2 shadowMapSize = vec2(0.0f, 0.0f);
    vec3 camPos = vec3(0.0f ,0.0f, 0.0f);
    vec2 blur = vec2(0.0f, 0.0f);
   
    vector<Light *> lights;
    int activeLight = 0;
    int nrVertices = 0;
    int nrActiveVertices = 0;      

    float polygonOffsetUnits = 1.0f;
    float polygonOffsetFactor = 0.5f;
    float depthRangeMax = 1.0f;
    float depthRangeMin = 0.0f;

    int background_mode = 1;
    int ui_mode = 0;
};

struct Shaders
{
    Shader* extraObj;
    Shader *gridBox;
    Shader *standard;
    Shader* axis;
    Shader *defaultDepth;
    Shader *defaultLight;
    Shader *blur;
    Shader *grid;
    Shader *object;
    Shader *objectDepth;
    Shader *objectLines;
    Shader *gui;
    Shader *noise;
    Shader *tessellation;
    Shader *cookTorrance;
    Shader *sphericalHarmonic;
    Shader *hair;
    Shader* hairNew;
    Shader* hairDepth;
    Shader* head;
    Shader* MeshDepth;
    Shader* headWireframe;
};

struct Inputs 
{
    vec2 mouse_cur = vec2(0.0f, 0.0f);
    vec2 mouse_prev = vec2(0.0f, 0.0f);
    vec2 mouse_diff = vec2(0.0f, 0.0f);

    bool left_button = false;
    bool right_button = false;

    bool ctrl = false;
    bool alt = false;
    bool a = false;
    bool s = false;
    bool w = false;
    bool d = false;
    bool g = false;
    bool b = false;
    bool r = false;
    bool n = false;
    bool h = false;
    bool p = false;
    bool o = false;
    bool t = false;
    bool i = false;

    bool f2 = false;

    float scroll_x = 0.0f;
    float scroll_y = 0.0f;

    bool gui_clicked = false;
    bool key_pressed = false;
};

typedef Singleton<Params> params;
typedef Singleton<Shaders> shaders;
typedef Singleton<Inputs> inputs;


void glEnable2D(void);
void glDisable2D(void);
void glEnableFixedFunction(const Transform &trans);
void glDisableFixedFunction();

float cosineInterpolation(float a, float b, float s);
float hermiteInterpolation(float y0, float y1, float y2, float y3, float mu, float tension, float bias);

void renderTexture(uint texture, uint posX, uint posY, float width, float height);
void renderQuad(float size, float r, float g, float b, float a);
void renderQuad(float width, float height, float r, float g, float b, float a);
void renderQuad(float posX, float posY, float width, float height);
void renderOrigin(float lineWidth);
void screenSizeQuad(float width, float height, float fov);

void renderString(const char *str, int x, int y, vec4 color, void *font = GLUT_BITMAP_HELVETICA_18);
void renderString(const char *str, int x, int y, float r, float g, float b, float a, void *font = GLUT_BITMAP_HELVETICA_18);

void smoothBackground(vec4 top, vec4 bottom, float windowWidth, float windowHeight);

//void saveFrameBuffer(QGLWidget *widget);
//void saveFrameBuffer(QGLWidget *widget, int idx);

void getCameraFrame(const Transform &trans, vec3 &dir, vec3 &up, vec3 &right, vec3 &pos);
vec3 getCamPosFromModelView(const Transform &trans);
vec3 getViewDirFromModelView(const Transform &trans);
vec3 getUpDirFromModelView(const Transform &trans);

static float colorJet[] = { 0.000000f, 0.000000f, 0.562500f, 0.000000f, 0.000000f, 0.625000f, 0.000000f, 0.000000f, 0.687500f, 0.000000f, 0.000000f, 0.750000f, 0.000000f, 0.000000f, 0.812500f, 0.000000f, 0.000000f, 0.875000f, 0.000000f, 0.000000f, 0.937500f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.062500f, 1.000000f, 0.000000f, 0.125000f, 1.000000f, 0.000000f, 0.187500f, 1.000000f, 0.000000f, 0.250000f, 1.000000f, 0.000000f, 0.312500f, 1.000000f, 0.000000f, 0.375000f, 1.000000f, 0.000000f, 0.437500f, 1.000000f, 0.000000f, 0.500000f, 1.000000f, 0.000000f, 0.562500f, 1.000000f, 0.000000f, 0.625000f, 1.000000f, 0.000000f, 0.687500f, 1.000000f, 0.000000f, 0.750000f, 1.000000f, 0.000000f, 0.812500f, 1.000000f, 0.000000f, 0.875000f, 1.000000f, 0.000000f, 0.937500f, 1.000000f, 0.000000f, 1.000000f, 1.000000f, 0.062500f, 1.000000f, 0.937500f, 0.125000f, 1.000000f, 0.875000f, 0.187500f, 1.000000f, 0.812500f, 0.250000f, 1.000000f, 0.750000f, 0.312500f, 1.000000f, 0.687500f, 0.375000f, 1.000000f, 0.625000f, 0.437500f, 1.000000f, 0.562500f, 0.500000f, 1.000000f, 0.500000f, 0.562500f, 1.000000f, 0.437500f, 0.625000f, 1.000000f, 0.375000f, 0.687500f, 1.000000f, 0.312500f, 0.750000f, 1.000000f, 0.250000f, 0.812500f, 1.000000f, 0.187500f, 0.875000f, 1.000000f, 0.125000f, 0.937500f, 1.000000f, 0.062500f, 1.000000f, 1.000000f, 0.000000f, 1.000000f, 0.937500f, 0.000000f, 1.000000f, 0.875000f, 0.000000f, 1.000000f, 0.812500f, 0.000000f, 1.000000f, 0.750000f, 0.000000f, 1.000000f, 0.687500f, 0.000000f, 1.000000f, 0.625000f, 0.000000f, 1.000000f, 0.562500f, 0.000000f, 1.000000f, 0.500000f, 0.000000f, 1.000000f, 0.437500f, 0.000000f, 1.000000f, 0.375000f, 0.000000f, 1.000000f, 0.312500f, 0.000000f, 1.000000f, 0.250000f, 0.000000f, 1.000000f, 0.187500f, 0.000000f, 1.000000f, 0.125000f, 0.000000f, 1.000000f, 0.062500f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.937500f, 0.000000f, 0.000000f, 0.875000f, 0.000000f, 0.000000f, 0.812500f, 0.000000f, 0.000000f, 0.750000f, 0.000000f, 0.000000f, 0.687500f, 0.000000f, 0.000000f, 0.625000f, 0.000000f, 0.000000f, 0.562500f, 0.000000f, 0.000000f, 0.500000f, 0.000000f, 0.000000f };
static float colorHot[] = { 0.041667f, 0.000000f, 0.000000f, 0.083333f, 0.000000f, 0.000000f, 0.125000f, 0.000000f, 0.000000f, 0.166667f, 0.000000f, 0.000000f, 0.208333f, 0.000000f, 0.000000f, 0.250000f, 0.000000f, 0.000000f, 0.291667f, 0.000000f, 0.000000f, 0.333333f, 0.000000f, 0.000000f, 0.375000f, 0.000000f, 0.000000f, 0.416667f, 0.000000f, 0.000000f, 0.458333f, 0.000000f, 0.000000f, 0.500000f, 0.000000f, 0.000000f, 0.541667f, 0.000000f, 0.000000f, 0.583333f, 0.000000f, 0.000000f, 0.625000f, 0.000000f, 0.000000f, 0.666667f, 0.000000f, 0.000000f, 0.708333f, 0.000000f, 0.000000f, 0.750000f, 0.000000f, 0.000000f, 0.791667f, 0.000000f, 0.000000f, 0.833333f, 0.000000f, 0.000000f, 0.875000f, 0.000000f, 0.000000f, 0.916667f, 0.000000f, 0.000000f, 0.958333f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.041667f, 0.000000f, 1.000000f, 0.083333f, 0.000000f, 1.000000f, 0.125000f, 0.000000f, 1.000000f, 0.166667f, 0.000000f, 1.000000f, 0.208333f, 0.000000f, 1.000000f, 0.250000f, 0.000000f, 1.000000f, 0.291667f, 0.000000f, 1.000000f, 0.333333f, 0.000000f, 1.000000f, 0.375000f, 0.000000f, 1.000000f, 0.416667f, 0.000000f, 1.000000f, 0.458333f, 0.000000f, 1.000000f, 0.500000f, 0.000000f, 1.000000f, 0.541667f, 0.000000f, 1.000000f, 0.583333f, 0.000000f, 1.000000f, 0.625000f, 0.000000f, 1.000000f, 0.666667f, 0.000000f, 1.000000f, 0.708333f, 0.000000f, 1.000000f, 0.750000f, 0.000000f, 1.000000f, 0.791667f, 0.000000f, 1.000000f, 0.833333f, 0.000000f, 1.000000f, 0.875000f, 0.000000f, 1.000000f, 0.916667f, 0.000000f, 1.000000f, 0.958333f, 0.000000f, 1.000000f, 1.000000f, 0.000000f, 1.000000f, 1.000000f, 0.062500f, 1.000000f, 1.000000f, 0.125000f, 1.000000f, 1.000000f, 0.187500f, 1.000000f, 1.000000f, 0.250000f, 1.000000f, 1.000000f, 0.312500f, 1.000000f, 1.000000f, 0.375000f, 1.000000f, 1.000000f, 0.437500f, 1.000000f, 1.000000f, 0.500000f, 1.000000f, 1.000000f, 0.562500f, 1.000000f, 1.000000f, 0.625000f, 1.000000f, 1.000000f, 0.687500f, 1.000000f, 1.000000f, 0.750000f, 1.000000f, 1.000000f, 0.812500f, 1.000000f, 1.000000f, 0.875000f, 1.000000f, 1.000000f, 0.937500f, 1.000000f, 1.000000f, 1.000000f };

void colorMap(float x, float * out, float * cm);
void colorMapBgr(float x, float * out, float * cm);

void clearInputs();

#endif