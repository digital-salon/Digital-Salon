#ifndef OBJECT_H
#define OBJECT_H

#include "Common.h"
#include "Geometry.h"
#include "ObjLoader.h"
#include "Mesh.h"

class Shader;
class VertexBufferObject;

class Object
{
public:
    Object(const string &fileName, bool normalize = true, bool buildLineVBO = false, bool buildNormalVBO = false, const vec3 &pos = vec3(), const vec3 &scale = vec3(1, 1, 1), const vec4 &rot = vec4(), const vec4 &color = vec4(1, 1, 1, 1));
    ~Object();

    void render(const Transform &trans);
    void renderDepth(const Transform &trans);

    bool m_isSelected;
    vec3 m_position;
    vec4 m_rotation;
    vec3 m_scale;
    vec4 m_color;
    bool m_normalize;
    bool m_lines;
    bool m_normals;

    BoundingBox m_bb;
    void move(int x, int y, const vec3 &dir, const vec3 &up, const vec3 &right, const vec3 &pos);

private:
    void prepareData(const string &fileName);
    void buildVBOMesh(vector<Vertex> &vertices, vector<uint> &indices);
    void buildVBOLines(vector<Vertex> &vertices, vector<uint> &indices);
    void buildVBONormals(vector<Vertex> &vertices, vector<uint> &indices);
    void normalizeGeometry(vector<vector<Vertex>> &vertices, const vec3 &translate, const vec3 &scale, const vec4 &rotate);        

public:
    string m_fileName;
    vector<vector<Vertex>> m_allVertices;
    vector<vector<uint>> m_allIndices;

    vector<VertexBufferObject *> m_vbosTriangles;
    vector<VertexBufferObject *> m_vbosLines;
    vector<VertexBufferObject *> m_vbosNormals;

    int m_nrTriangles;
    int m_nrVertices;    

    vector<Material> m_materials;

    float selected(Picking &pick, const Transform &trans, int sw, int sh, int mx, int my);
};

#endif