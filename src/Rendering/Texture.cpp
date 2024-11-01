#include "Texture.h"
#include "freeImage/FreeImage.h"

Texture::Texture()
: m_id(0),
  m_width(0),
  m_height(0),
  m_target(GL_TEXTURE_2D),
  m_mipLevel(0),
  m_internalFormat(GL_RGBA),
  m_format(GL_RGBA),
  m_border(0),
  m_type(GL_UNSIGNED_BYTE),
  m_minFilter(GL_NEAREST),
  m_magFilter(GL_NEAREST),
  m_wrap(GL_CLAMP),
  m_envMode(GL_REPLACE),
  m_createMipMaps(GL_FALSE),
  m_maxAnisotropy(1.0f)
{
}

Texture::Texture(GLsizei w, GLsizei h, GLint iFormat, GLint format, GLint type)
: m_id(0),
  m_width(w),
  m_height(h),
  m_target(GL_TEXTURE_2D),
  m_mipLevel(0),
  m_internalFormat(iFormat),
  m_format(format),
  m_border(0),
  m_type(type),
  m_minFilter(GL_NEAREST),
  m_magFilter(GL_NEAREST),
  m_wrap(GL_CLAMP),
  m_envMode(GL_REPLACE),
  m_createMipMaps(GL_FALSE),
  m_maxAnisotropy(1.0f)
{
    create(nullptr);
}

Texture::Texture(GLsizei w, GLsizei h, GLint iFormat, GLint format, GLint type, GLvoid *data)
: m_id(0),
  m_width(w),
  m_height(h),
  m_target(GL_TEXTURE_2D),
  m_mipLevel(0),
  m_internalFormat(iFormat),
  m_format(format),
  m_border(0),
  m_type(type),
  m_minFilter(GL_NEAREST),
  m_magFilter(GL_NEAREST),
  m_wrap(GL_CLAMP),
  m_envMode(GL_REPLACE),
  m_createMipMaps(GL_TRUE),
  m_maxAnisotropy(0.0f)
{
    create(data);
}

Texture::Texture(const string& path)
: m_id(0),
  m_width(0),
  m_height(0),
  m_target(GL_TEXTURE_2D),
  m_mipLevel(0),
  m_internalFormat(GL_RGBA),
  m_format(GL_BGRA),
  m_border(0),
  m_type(GL_UNSIGNED_BYTE),
  m_minFilter(GL_LINEAR_MIPMAP_LINEAR),
  m_magFilter(GL_LINEAR),
  m_wrap(GL_CLAMP),
  m_envMode(GL_REPLACE),
  m_createMipMaps(GL_TRUE),
  m_maxAnisotropy(16.0f)
{
    load(path);
}

Texture::Texture(const Texture &t)
{
  m_id = t.id();
  m_width = t.width();
  m_height = t.height();
  m_target = t.target();
  m_mipLevel = t.mipLevel();
  m_internalFormat = t.internalFormat();
  m_format = t.format();
  m_border = t.border();
  m_type = t.type();
  m_minFilter = t.minFilter();
  m_magFilter = t.magFilter();
  m_wrap = t.wrap();
  m_envMode = t.envMode();
  m_createMipMaps = t.createMipMaps();
  m_maxAnisotropy = t.maxAnisotropy();
}

Texture &Texture::operator = (const Texture &t)
{
  m_id = t.id();
  m_width = t.width();
  m_height = t.height();
  m_target = t.target();
  m_mipLevel = t.mipLevel();
  m_internalFormat = t.internalFormat();
  m_format = t.format();
  m_border = t.border();
  m_type = t.type();
  m_minFilter = t.minFilter();
  m_magFilter = t.magFilter();
  m_wrap = t.wrap();
  m_envMode = t.envMode();
  m_createMipMaps = t.createMipMaps();
  m_maxAnisotropy = t.maxAnisotropy();

  return *this;
}

Texture::~Texture()
{
    deleteTex();
}

void Texture::load(const string& path)
{
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(path.c_str());

    if (format == FIF_UNKNOWN)      
        format = FreeImage_GetFIFFromFilename(path.c_str());

    if (format == FIF_UNKNOWN)      
        throw(std::runtime_error("File format not supported"));

    FIBITMAP* bitmap = FreeImage_Load(format, path.c_str());
    FIBITMAP* bitmap2 = FreeImage_ConvertTo32Bits(bitmap);
    FreeImage_Unload(bitmap);

    int w = FreeImage_GetWidth(bitmap2);
    int h = FreeImage_GetHeight(bitmap2);

    m_width = w;
    m_height = h;

    vector<char> out(w * h * 4);
    FreeImage_ConvertToRawBits((BYTE*)out.data(), bitmap2, FreeImage_GetWidth(bitmap2) * 4, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, false);
    FreeImage_Unload(bitmap2);

    create(out.data());    
}

void Texture::create(GLvoid *data)
{    
    glGenTextures(1, &m_id);	
    glBindTexture(m_target, m_id);    

    glTexImage2D(m_target, 0, m_internalFormat, m_width, m_height, m_border, m_format, m_type, data);     

    glTexParameteri(m_target, GL_TEXTURE_MIN_FILTER, m_minFilter);
    glTexParameteri(m_target, GL_TEXTURE_MAG_FILTER, m_magFilter); 

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, m_envMode);

    glTexParameterf(m_target, GL_TEXTURE_WRAP_S, (GLfloat)m_wrap);
    glTexParameterf(m_target, GL_TEXTURE_WRAP_T, (GLfloat)m_wrap);
    glTexParameterf(m_target, GL_TEXTURE_WRAP_R, (GLfloat)m_wrap);

    glTexParameterf(m_target, GL_TEXTURE_MAX_ANISOTROPY_EXT, m_maxAnisotropy);

    glGenerateMipmap(m_target);

    glBindTexture(m_target, 0);
}

void Texture::bind()
{
    glBindTexture(m_target, m_id);
}

void Texture::release()
{
    glBindTexture(m_target, 0);
}

void Texture::deleteTex()
{
	if (m_id != 0)
	{
		glDeleteTextures(1, &m_id);	
	}
}

GLuint Texture::id() const
{        
    return m_id;
}

void Texture::setWrapMode(GLint wrap)
{
    m_wrap = wrap;

    bind();

    glTexParameterf(m_target, GL_TEXTURE_WRAP_S, (float)m_wrap);
    glTexParameterf(m_target, GL_TEXTURE_WRAP_T, (float)m_wrap);
    glTexParameterf(m_target, GL_TEXTURE_WRAP_R, (float)m_wrap);

    release();
}

void Texture::setEnvMode(GLint envMode)
{
    m_envMode = envMode;

    bind();

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, m_envMode);

    release();
}

void Texture::setFilter(GLint minFilter, GLint magFilter)
{
    m_minFilter = minFilter;
    m_magFilter = magFilter; 

    bind();

    glTexParameteri(m_target, GL_TEXTURE_MIN_FILTER, m_minFilter);
    glTexParameteri(m_target, GL_TEXTURE_MAG_FILTER, m_magFilter); 

    release();
}

void Texture::setMaxIsotropy(GLfloat anisotropy)
{
    m_maxAnisotropy = anisotropy;

    bind();

    glTexParameterf(m_target, GL_TEXTURE_MAX_ANISOTROPY_EXT, m_maxAnisotropy);  

    release();
}

void Texture::render(GLuint posX, GLuint posY, GLfloat width, GLfloat height)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glColor4f(0.0, 0.0, 0.0, 1.0f);  
    
    glEnable(GL_TEXTURE_2D);     
    glActiveTexture(GL_TEXTURE0);
    
    bind();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


    glDisable(GL_DEPTH_TEST);      
    glEnable2D();
    glPushMatrix();            
        glTranslatef((float)posX, (float)posY, 0.0f);
        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 1.0f);
            glVertex3f(0.0f, 0.0f, 0.0f);

            glTexCoord2f(1.0f, 1.0f);
            glVertex3f(width, 0.0f, 0.0f);
            
            glTexCoord2f(1.0f, 0.0f);
            glVertex3f(width, height, 0.0f);

            glTexCoord2f(0.0f, 0.0f);
            glVertex3f(0.0, height, 0.0f);
        glEnd();
    glPopMatrix();
    glDisable2D();

    release();

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);  

    glPopClientAttrib();
    glPopAttrib();
}

GLsizei Texture::width() const
{
    return m_width;
}

GLsizei Texture::height() const
{
    return m_height;
}

GLenum Texture::target() const
{
    return m_target;
}

GLint Texture::mipLevel() const
{
    return m_mipLevel;
}

GLint Texture::internalFormat() const
{
    return m_internalFormat;
}

GLenum Texture::format() const
{
    return m_format;
}

GLint Texture::border() const
{
    return m_border;
}

GLenum Texture::type() const
{
    return m_type;
}

GLint Texture::minFilter() const
{
    return m_minFilter;
}

GLint Texture::magFilter() const
{
    return m_magFilter;
}

GLint Texture::wrap() const
{
    return m_wrap;
}

GLint Texture::envMode() const

{
    return m_envMode;
}

GLfloat Texture::maxAnisotropy() const
{
    return m_maxAnisotropy;
}

GLboolean Texture::createMipMaps() const
{
    return m_createMipMaps;
}