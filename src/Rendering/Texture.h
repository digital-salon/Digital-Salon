#ifndef TEXTURE_H
#define TEXTURE_H

#include "Common.h"

class Texture
{

public:
   Texture();
   Texture(const Texture &);   
   Texture(GLsizei w, GLsizei h, GLint iFormat, GLint format, GLint type);
   Texture(GLsizei w, GLsizei h, GLint iFormat, GLint format, GLint type, GLvoid *data);
   Texture(const string &path);
   ~Texture();

   Texture &operator = (const Texture &t);

   void bind();
   void release();
   void create(GLvoid *data = nullptr);

   void setWrapMode(GLint wrap = GL_REPEAT);
   void setEnvMode(GLint envMode = GL_REPLACE);
   void setFilter(GLint minFilter = GL_NEAREST, GLint magFilter = GL_NEAREST);
   void setMaxIsotropy(GLfloat anisotropy = 1.0f);

   void deleteTex();
   void load(const string& path);

   void render(GLuint posX, GLuint posY, GLfloat width, GLfloat height);

   GLuint id() const;
   GLsizei width() const;
   GLsizei height() const;
   GLenum target() const;
   GLint mipLevel() const;
   GLint internalFormat() const;
   GLenum format() const;
   GLint border() const;
   GLenum type() const;
   GLint minFilter() const;
   GLint magFilter() const;
   GLint wrap() const;
   GLint envMode() const;
   GLfloat maxAnisotropy() const;
   GLboolean createMipMaps() const;

private:
    GLsizei m_width;
    GLsizei m_height;
    GLenum  m_target;
    GLint   m_mipLevel;
    GLint   m_internalFormat;
    GLenum  m_format;
    GLint   m_border;
    GLenum  m_type;
    GLint   m_minFilter;
    GLint   m_magFilter;
    GLint   m_wrap;
    GLint   m_envMode;
    GLboolean m_createMipMaps;
    GLfloat m_maxAnisotropy;
    GLuint m_id;
};

#endif

