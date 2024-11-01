#ifndef VERTEXBUFFEROBJECT_H
#define VERTEXBUFFEROBJECT_H

#include "Common.h"

#define VERTEX_POSITION 0
#define VERTEX_NORMAL   1
#define VERTEX_COLOR    2
#define VERTEX_TEXTURE  3
#define VERTEX_TEMP1    4
#define VERTEX_TEMP2    5

class VertexBufferObject
{
public:
    struct DATA
    {
        float vx, vy, vz, vw;
        float nx, ny, nz, nw;
        float cx, cy, cz, cw;        
        float tx, ty, tz, tw;
        float ux, uy, uz, uw;
    };

    struct DATASimple
    {
        float vx, vy, vz;
        float nx, ny, nz;
        float cx, cy, cz, cw;
        float tx, ty;
    };

    struct DATAHair
    {
        float vx, vy, vz;
        float tx, ty, tz;
        float cx, cy, cz;
    };

public:
    VertexBufferObject();
    ~VertexBufferObject();

    void bind();
    GLuint id() const;
    void release();
    void render();
    
	GLuint createVBO(GLenum target, GLuint dataSize, const void* data, GLenum usage);
    void setData(const DATA *data, GLenum usage, GLuint nrVertices, GLenum primitiveMode);
    void setDataHair(const DATAHair* data, GLenum usage, GLuint nrVertices, GLenum primitiveMode);
    void setIndexData(const GLvoid *data, GLenum usage, GLint nrIndices);
    void addAttrib(GLint attribLoc);
    void setVerticesToRender(GLuint nrVertices); 
    void bindAttribs();
    void setDynamicRendering(GLboolean dynamicRendering);
    void bindDefaultAttribs();

    GLuint nrVertices() const;
    GLuint nrDynamicVertices() const;

    DATA *map(int &size);
    void unmap();

private:
    GLuint m_nrVertices;
    GLuint m_nrDynamicVertices;
    GLuint m_nrIndices;
    GLuint m_nrDynamicIndices;
    GLuint m_sizeBytesVertices;
    GLuint m_sizeBytesIndices;    
    GLuint m_sizeAsStride;
    GLuint m_bufferId;
    GLuint m_indexBufferId;
    
    GLenum m_primitiveMode;
    GLenum m_usage;    

    vector<GLint> m_attribLocations;

    GLboolean m_useIndexBuffer;
    GLboolean m_dynamicRendering;   
};

#endif

