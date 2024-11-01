#include <stdio.h>
#include "VertexBufferObject.h"

VertexBufferObject::VertexBufferObject()
: m_useIndexBuffer(GL_FALSE),
  m_nrVertices(0),
  m_nrDynamicVertices(0),
  m_nrIndices(0),
  m_nrDynamicIndices(0),
  m_sizeBytesVertices(0),
  m_sizeBytesIndices(0),   
  m_bufferId(0),
  m_indexBufferId(0),    
  m_primitiveMode(GL_POINTS),
  m_usage(GL_STATIC_DRAW),
  m_sizeAsStride(0),
  m_dynamicRendering(GL_FALSE)
{  
}

unsigned int VertexBufferObject::createVBO(GLenum target, GLuint dataSize, const void* data, GLenum usage)
{

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(target, vbo);
    glBufferData(target, dataSize, data, usage);
    glBindBuffer(target, 0);

    return vbo;
}

VertexBufferObject::~VertexBufferObject()
{
    if(m_bufferId != GL_INVALID_VALUE)
        glDeleteBuffers(1, &m_bufferId);

    if(m_indexBufferId != GL_INVALID_VALUE)
        glDeleteBuffers(1, &m_indexBufferId);
}

void VertexBufferObject::bindAttribs()
{
    int size = (int)m_attribLocations.size();
    glBindBuffer(GL_ARRAY_BUFFER, m_bufferId);
    for(int i=0; i<size; ++i)
    {
        GLint attribLoc = m_attribLocations[i];
        glVertexAttribPointer(attribLoc, 4, GL_FLOAT, GL_FALSE, sizeof(DATA), ((char*)NULL + 4 * sizeof(float) * i));
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void VertexBufferObject::render()
{    
    params::inst()->nrActiveVertices += m_nrVertices;

    int size = (int)m_attribLocations.size();
    glBindBuffer(GL_ARRAY_BUFFER, m_bufferId);
    
    for(int i=0; i<size; ++i)
    {
        glVertexAttribPointer(m_attribLocations[i], 4, GL_FLOAT, GL_FALSE, sizeof(DATA), ((char*)NULL + 4 * sizeof(float) * i));
        glEnableVertexAttribArray(m_attribLocations[i]);
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    if(m_useIndexBuffer)
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBufferId);
        
        if(m_dynamicRendering)
        {
            glDrawElements(m_primitiveMode, m_nrDynamicIndices, GL_UNSIGNED_INT, 0);
        }
        else
        {
            glDrawElements(m_primitiveMode, m_nrIndices, GL_UNSIGNED_INT, 0);
        }
    }
    else
    {
        if(m_dynamicRendering)
        {
            glDrawArrays(m_primitiveMode, 0, m_nrDynamicVertices);
        }
        else
        {
            glDrawArrays(m_primitiveMode, 0, m_nrVertices);
        }
    }

    for(int i=0; i<size; ++i)
    {
        glDisableVertexAttribArray(m_attribLocations[i]);	
    }
}

void VertexBufferObject::setData(const DATA *data, GLenum usage, GLuint nrVertices, GLenum primitiveMode)
{
    m_sizeAsStride = sizeof(DATA);
    m_sizeBytesVertices = m_sizeAsStride * nrVertices;
    m_primitiveMode = primitiveMode;
    m_usage = usage;
    m_nrVertices = nrVertices;
    m_nrDynamicVertices = nrVertices;

    params::inst()->nrVertices += nrVertices;

    m_bufferId = createVBO(GL_ARRAY_BUFFER, m_sizeBytesVertices, data, usage);
}

void VertexBufferObject::setDataHair(const DATAHair* data, GLenum usage, GLuint nrVertices, GLenum primitiveMode)
{
    m_sizeAsStride = sizeof(DATAHair);
    m_sizeBytesVertices = m_sizeAsStride * nrVertices;
    m_primitiveMode = primitiveMode;
    m_usage = usage;
    m_nrVertices = nrVertices;
    m_nrDynamicVertices = nrVertices;

    params::inst()->nrVertices += nrVertices;

    m_bufferId = createVBO(GL_ARRAY_BUFFER, m_sizeBytesVertices, data, usage);
}

void VertexBufferObject::addAttrib(GLint attribLoc)
{
    m_attribLocations.push_back(attribLoc);
}

void VertexBufferObject::setIndexData(const GLvoid *data, GLenum usage, GLint nrIndices)
{
    m_nrIndices = nrIndices;
    m_useIndexBuffer = GL_TRUE;

    m_sizeBytesIndices = sizeof(uint) * nrIndices;
    
    m_indexBufferId = createVBO(GL_ELEMENT_ARRAY_BUFFER, m_sizeBytesIndices, data, usage); 
}

void VertexBufferObject::setVerticesToRender(GLuint nrVertices)
{
    m_nrDynamicVertices = nrVertices;
}

void VertexBufferObject::bind()
{
    glBindBuffer(GL_ARRAY_BUFFER, m_bufferId);
}

void VertexBufferObject::release()
{
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void VertexBufferObject::setDynamicRendering(GLboolean dynamicRendering)
{
    m_dynamicRendering = dynamicRendering;
}

GLuint VertexBufferObject::nrVertices() const
{
    return m_nrVertices;
}

GLuint VertexBufferObject::nrDynamicVertices() const
{
    return m_nrDynamicVertices;
}

GLuint VertexBufferObject::id() const
{
    return m_bufferId;
}

VertexBufferObject::DATA *VertexBufferObject::map(int &size)
{
    this->bind();

	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
	size /= sizeof(VertexBufferObject::DATA);

	VertexBufferObject::DATA *data = (VertexBufferObject::DATA*) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

    return data;
}

void VertexBufferObject::unmap()
{
    glUnmapBuffer(GL_ARRAY_BUFFER);
    this->release();
}

void VertexBufferObject::bindDefaultAttribs()
{
    this->addAttrib(VERTEX_POSITION);
    this->addAttrib(VERTEX_NORMAL);
    this->addAttrib(VERTEX_COLOR);
    this->addAttrib(VERTEX_TEXTURE);
    this->addAttrib(VERTEX_TEMP1);
    this->bindAttribs();
}