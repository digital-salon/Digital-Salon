#ifndef MESH_H
#define MESH_H

#include "Common.h"
#include "Texture.h"
#include <memory>

class VertexBufferObject;

class Material
{
public:
    Material(const vec3 &_Ka, const vec3 &_Kd, const vec3 &_Ks, float _Ns, string dTexName = string()) : Ka(_Ka), Kd(_Kd), Ks(_Ks), Ns(_Ns), tex(nullptr)  
    { 
        if(dTexName.length() > 0)
        {
            tex = std::shared_ptr<Texture>(new Texture(dTexName));
            tex->setEnvMode(GL_REPLACE);
            tex->setWrapMode(GL_REPEAT);
        }
    }

    Material(const Material &m) : Ka(m.Ka), Kd(m.Kd), Ks(m.Ks), Ns(m.Ns), tex(m.tex) { }
    Material &operator = (const Material &m) { Ka = m.Ka; Kd = m.Kd; Ks = m.Ks; Ns = m.Ns; tex = m.tex; }

    ~Material() 
    {        
    }

    vec3 Ka;
    vec3 Kd;
    vec3 Ks;

    float Ns;

    std::shared_ptr<Texture> tex;
};


class Mesh
{
public:
   Mesh();
   ~Mesh();

   //static vector<VertexBufferObject*> obj(const QString &fileName, const vec3 &rot = vec3(0.0f, 0.0f, 0.0f), const vec3 &scale = vec3(1.0f, 1.0f, 1.0f), GLenum primitive = GL_TRIANGLES);
   //static vector<VertexBufferObject*> obj(QString fileName);
   static VertexBufferObject *quadLines(int startX, int startY, int width, int height, const vec4 &color = vec4());
   static VertexBufferObject *quad(int startX, int startY, int width, int height, const vec4 &color = vec4(), GLenum primitive = GL_QUADS);
   static VertexBufferObject *quad(int width, int height, const vec4 &color = vec4(), GLenum primitive = GL_QUADS);
   static VertexBufferObject *sphere(float radius, int iterations, const vec4 &color = vec4(), GLenum primitive = GL_TRIANGLES);
   static VertexBufferObject *box(const vec3 &mi, const vec3 &ma, const vec4 &color = vec4(), GLenum primitive = GL_QUADS);
   static VertexBufferObject *vbo(const vector<vec3> &vertices, const vector<vec4> &colors = vector<vec4>(), const vector<vec3> &normals = vector<vec3>(), GLenum primitive = GL_POINTS);

private:
    typedef struct 
    {
       double x,y,z;
    } XYZ;
    
    typedef struct 
    {
       vec3 p1,p2,p3;
    } FACET3;


    //Paul Bourke Sphere http://paulbourke.net/miscellaneous/sphere_cylinder/
    static int createNSphere(FACET3 *f, int iterations);
    static int CreateUnitSphere(FACET3 *facets, int iterations);
    static vec3 MidPoint(vec3 p1, vec3 p2);
};

#endif

