#include "FrameBufferObject.h"

FrameBufferObject::FrameBufferObject(GLuint width, GLuint height, GLuint nrTexAtt, GLuint nrBufferAtt, GLboolean attachDepthTex, GLint filter)
: m_width(max<GLuint>(1, width)),
  m_height(max<GLuint>(1, height)),
  m_fboID(0),
  m_savedFboID(0),
  m_nrTexAtt(nrTexAtt),
  m_nrBufferAtt(nrBufferAtt),
  m_maxColorAtt(0),
  m_filter(filter)
{
    glGenFramebuffers(1, &m_fboID);
    
    GLenum maxTexAtt = 0;
    for(uint i=0; i<nrTexAtt; ++i)
    {
        createTexAttachment(GL_COLOR_ATTACHMENT0 + i, GL_RGBA32F, GL_RGBA, GL_FLOAT);
        maxTexAtt ++;
    }

    for(uint i=maxTexAtt; i<maxTexAtt+nrBufferAtt; ++i)
    {
        createBufferAttachment(GL_COLOR_ATTACHMENT0 + i, GL_RGBA32F);
    }

    if(attachDepthTex)
        attachDepthTexture();

    //attachStencilRenderbuffer();
    //attachStencilDepthRenderbuffer();

    guardedBind();

    glReadBuffer(GL_NONE); 

    m_maxColorAtt = maxAttachments()-1;

    if(!checkStatus())
    {
        exit(0);
    }

    #ifdef FRAMEBUFFER_DEBUG
      //info();
    #endif

    release();
}

FrameBufferObject::~FrameBufferObject()
{
    for(uint attachment=GL_COLOR_ATTACHMENT0; attachment<GL_COLOR_ATTACHMENT0+m_maxColorAtt; ++attachment)
    {
        switchToTexAttachment(0, attachment);

        std::map<GLenum, GLuint>::const_iterator iterTex = m_texAtts.find(attachment);
        
        //Delete Texture Attachments
        if(iterTex != m_texAtts.end())
        {
            GLuint id = iterTex->second;

            #ifdef FRAMEBUFFER_DEBUG
            std::cout << "FRAMEBUFFER::" << lookupAttachment(iterTex->first) << " (Texture) deleted" << std::endl;
            #endif      

            if(id != 0)
            {
                glDeleteTextures(1, &id);
                m_texAtts[attachment] = 0;
            }
        }

        //Delete Renderbuffer Attachments
        std::map<GLenum, GLuint>::const_iterator iterBuffer = m_bufferAtts.find(attachment);
        
        if(iterBuffer != m_bufferAtts.end())
        {
            GLuint id = iterBuffer->second;
            
            #ifdef FRAMEBUFFER_DEBUG
            std::cout << "FRAMEBUFFER::" << lookupAttachment(iterTex->first) << " (Renderbuffer) deleted" << std::endl;
            #endif

            if(id != 0)
                glDeleteRenderbuffers(1, &id);
        }
    }

    //Delete Depth Texture Attachment
    std::map<GLenum, GLuint>::const_iterator iterDepth = m_texAtts.find(GL_DEPTH_ATTACHMENT);
    if(iterDepth != m_texAtts.end())
    {
        GLuint id = iterDepth->second;     
            
        #ifdef FRAMEBUFFER_DEBUG
        std::cout << "FRAMEBUFFER::" << lookupAttachment(iterDepth->first) << " (Texture) deleted" << std::endl;
        #endif

        if(id != 0)
            glDeleteTextures(1, &id);
    }


   //Delete Stencil Renderbuffer Attachment
    std::map<GLenum, GLuint>::const_iterator iterStencil = m_bufferAtts.find(GL_STENCIL_ATTACHMENT);
    if(iterStencil != m_bufferAtts.end())
    {
        GLuint id = iterStencil->second;     
        
        #ifdef FRAMEBUFFER_DEBUG
        std::cout << "FRAMEBUFFER::" << lookupAttachment(iterStencil->first) << " (Renderbuffer) deleted" << std::endl;
        #endif

        if(id != 0)
            glDeleteRenderbuffers(1, &id);
    }


    //Delete DepthStencil Renderbuffer Attachment
    std::map<GLenum, GLuint>::const_iterator iterDepthStencil = m_bufferAtts.find(GL_DEPTH_STENCIL_ATTACHMENT);
    if(iterDepthStencil != m_bufferAtts.end())
    {
        GLuint id = iterDepthStencil->second;     

        #ifdef FRAMEBUFFER_DEBUG
        std::cout << "FRAMEBUFFER::" << lookupAttachment(iterDepthStencil->first) << " (Renderbuffer) deleted" << std::endl;
        #endif

        if(id != 0)
            glDeleteRenderbuffers(1, &id);
    }
        

    glDeleteFramebuffers(1, &m_fboID);  

    #ifdef FRAMEBUFFER_DEBUG
    std::cout << "FRAMEBUFFER::deleted" << std::endl;
    #endif

}

GLvoid FrameBufferObject::checkDrawBuffers()
{
    std::vector<GLenum> attachments;
    GLint maxAtt = maxAttachments();

    for(int i=GL_COLOR_ATTACHMENT0; i<GL_COLOR_ATTACHMENT0 + maxAtt; ++i)
    {
        GLuint texAtt = texAttachment(i);
        GLuint buffAtt = bufferAttachment(i);

        if(texAtt != 0)
            attachments.push_back(i);

        if(buffAtt != 0)
            attachments.push_back(i);
    }

    GLint size = (GLint)attachments.size();
    GLenum *buffers = new GLenum[size];

    for(int i=0; i<size; ++i)
    {
        buffers[i] = attachments[i];
    }

    glDrawBuffers(size, buffers);
}

GLvoid FrameBufferObject::createTexAttachment(GLenum attachment, GLuint iFormat, GLuint format, GLuint type)
{
    uint texID = emptyTexture(m_width, m_height, iFormat, format, type);
    attachTexture(texID, attachment);
}

GLvoid FrameBufferObject::attachTexture(GLuint texID, GLenum attachment)
{
    guardedBind();
    
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texID, 0);

    #ifdef FRAMEBUFFER_DEBUG
    std::cout << "FRAMEBUFFER::" << lookupAttachment(attachment) << " (Texture) attached" << std::endl;
    #endif

    m_texAtts[attachment] = texID;

    checkDrawBuffers();

    release();    
}

GLvoid FrameBufferObject::attachTexture(GLuint texID, GLenum attachment, GLenum texTarget, GLint mipLevel, GLint zSlice)
{
    guardedBind();
    
    if(texTarget == GL_TEXTURE_1D)
    {
        glFramebufferTexture1D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_1D, texID, mipLevel);
    }
    else if(texTarget == GL_TEXTURE_3D)
    {
        glFramebufferTexture3D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_3D, texID, mipLevel, zSlice);
    }
    else
    {
        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texID, mipLevel);
        glGenerateMipmap(GL_TRUE); //??
    }

    #ifdef FRAMEBUFFER_DEBUG
    std::cout << "FRAMEBUFFER::" << lookupAttachment(attachment) << " (Texture) attached" << std::endl;
    #endif

    m_texAtts[attachment] = texID;

    checkDrawBuffers();

    release();
}

GLvoid FrameBufferObject::switchToTexAttachment(GLuint texID, GLenum attachment)
{
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texID, 0);
}

GLvoid FrameBufferObject::attachDepthTexture()
{
    //createTexAttachment(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
    createTexAttachment(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE);
}

GLvoid FrameBufferObject::createBufferAttachment(GLenum attachment, GLenum bufferFormat)
{
    uint bufferID = 0;
    glGenRenderbuffers(1, &bufferID);

    glBindRenderbuffer(GL_RENDERBUFFER, bufferID);
    glRenderbufferStorage(GL_RENDERBUFFER, bufferFormat, m_width, m_height);

    attachBuffer(bufferID, attachment);
}

GLvoid FrameBufferObject::attachBuffer(GLuint bufferID, GLenum attachment)
{
    guardedBind();
          
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, bufferID);

    #ifdef FRAMEBUFFER_DEBUG
    std::cout << "FRAMEBUFFER::" << lookupAttachment(attachment) << " (Renderbuffer) attached" << std::endl;
    #endif

    m_bufferAtts[attachment] = bufferID;

    checkDrawBuffers();

    release();
}

GLuint FrameBufferObject::texAttachment(GLenum attachment) const
{
    std::map<GLenum, GLuint>::const_iterator iter = m_texAtts.find(attachment);

    GLuint id = 0;
    if(iter != m_texAtts.end())
        id = iter->second;

    return id;
}

GLuint FrameBufferObject::bufferAttachment(GLenum attachment) const
{
    std::map<GLenum, GLuint>::const_iterator iter = m_bufferAtts.find(attachment);

    GLuint id = 0;
    if(iter != m_bufferAtts.end())
        id = iter->second;

    return id;
}

GLvoid FrameBufferObject::attachStencilDepthRenderbuffer()
{
    //createBufferAttachment(GL_DEPTH_STENCIL_ATTACHMENT, GL_DEPTH_STENCIL_EXT); 
    createBufferAttachment(GL_DEPTH_STENCIL_ATTACHMENT, GL_DEPTH32F_STENCIL8);

    //glGenRenderbuffersEXT(1, &m_rboID);
    //glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, m_rboID);
    //glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_STENCIL_EXT, width, height);
    //glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);

    //glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, m_rboID);
    //glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, m_rboID);
}

GLvoid FrameBufferObject::attachStencilRenderbuffer()
{
    createBufferAttachment(GL_STENCIL_ATTACHMENT_EXT, GL_STENCIL_INDEX16);
}

GLvoid FrameBufferObject::bind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_fboID);
    //guardedBind();
}

GLvoid FrameBufferObject::guardedBind()
{
    glGetIntegerv(GL_FRAMEBUFFER_BINDING_EXT, &m_savedFboID);

    if (m_fboID != (GLuint)m_savedFboID) 
    {
        glBindFramebuffer(GL_FRAMEBUFFER, m_fboID);
    }
}

GLvoid FrameBufferObject::release()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLuint FrameBufferObject::id() const
{
    return m_fboID;
}

GLuint FrameBufferObject::emptyTexture(GLuint width, GLuint height, GLuint iformat, GLuint format, GLuint type)
{
	GLuint ret;

	glGenTextures(1, &ret);
	glBindTexture(GL_TEXTURE_2D, ret);
	glTexImage2D(GL_TEXTURE_2D, 0, iformat, width, height, 0, format, type, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, m_filter);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, m_filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    //glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    float color[] = {1.0f, 1.0f, 1.0f, 1.0f};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color); 

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 0);
	
    return ret;
}

GLboolean FrameBufferObject::checkStatus()
{
    GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    GLboolean result = false;

    switch(status)
    {
    case GL_FRAMEBUFFER_COMPLETE_EXT:
        #ifdef FRAMEBUFFER_DEBUG
        std::cout << "FRAMEBUFFER::Complete" << std::endl;
        std::cout << "FRAMEBUFFER::Size: " << m_width << "x" << m_height << std::endl;
        #endif
        result = true;
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
        #ifdef FRAMEBUFFER_DEBUG    
        std::cout << "[ERROR]FRAMEBUFFER::incomplete: Attachment is NOT complete" << std::endl;
        #endif
        result = false;
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
        #ifdef FRAMEBUFFER_DEBUG    
        std::cout << "[ERROR]FRAMEBUFFER::incomplete: No image is attached to FBO" << std::endl;
        #endif
        result = false;
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
        #ifdef FRAMEBUFFER_DEBUG
        std::cout << "[ERROR]FRAMEBUFFER::incomplete: Attached images have different dimensions" << std::endl;
        #endif
        result = false;
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
        #ifdef FRAMEBUFFER_DEBUG    
        std::cout << "[ERROR]FRAMEBUFFER::incomplete: Color attached images have different internal formats" << std::endl;
        #endif
        result = false;
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
        #ifdef FRAMEBUFFER_DEBUG    
        std::cout << "[ERROR]FRAMEBUFFER::incomplete: Draw buffer" << std::endl;
        #endif
        result = false;
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
        #ifdef FRAMEBUFFER_DEBUG    
        std::cout << "[ERROR]FRAMEBUFFER::incomplete: Read buffer" << std::endl;
        #endif
        result = false;
        break;
    case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
        #ifdef FRAMEBUFFER_DEBUG    
        std::cout << "[ERROR]FRAMEBUFFER::Unsupported by FBO implementation" << std::endl;
        #endif
        result = false;
        break;
    default:
        #ifdef FRAMEBUFFER_DEBUG
        std::cout << "[ERROR]FRAMEBUFFER::Unknow error" << std::endl;
        #endif
        result = false;
        break;
    }

    return result;
}

GLuint FrameBufferObject::maxAttachments()
{
    GLint maxAtt = 0;
    
    glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxAtt);
    
    return maxAtt;
}

GLvoid FrameBufferObject::info()
{
    GLint maxAtt = 0;   
    glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxAtt);

    GLint maxRenderbufferSize;
    glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &maxRenderbufferSize);

    GLint attObjectType_0 = 0;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &attObjectType_0);

    GLint attObjectName_0 = 0;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &attObjectName_0);

    GLint attRedSize_0 = 0;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE, &attRedSize_0);

    GLint attGreenSize_0 = 0;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE, &attGreenSize_0);

    GLint attBlueSize_0 = 0;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE, &attBlueSize_0);

    GLint rbWidth = 0;
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &rbWidth); //?? RB not bound

    GLint rbHeight = 0;
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &rbHeight); //??

    std::cout << "FRAMEBUFFER::MaxAttachments: " << maxAtt << std::endl;
    std::cout << "FRAMEBUFFER::MaxRenderbufferSize: " << maxRenderbufferSize << std::endl;
    std::cout << "FRAMEBUFFER::AttachmentObjectType[0]: " << attObjectType_0 << std::endl;
    std::cout << "FRAMEBUFFER::AttachmentObjectName[0]: " << attObjectName_0 << std::endl;
    std::cout << "FRAMEBUFFER::AttachmentBlueSize[0]: " << attRedSize_0 << std::endl;
    std::cout << "FRAMEBUFFER::AttachmentGreenSize[0]: " << attGreenSize_0 << std::endl;
    std::cout << "FRAMEBUFFER::AttachmentBlueSize[0]: " << attBlueSize_0 << std::endl;
    std::cout << "FRAMEBUFFER::RenderbufferWidth: " << rbWidth << std::endl;
    std::cout << "FRAMEBUFFER::RenderbufferHeight: " << rbHeight << std::endl;


}

std::string FrameBufferObject::lookupAttachment(GLenum att)
{
    std::string text;

    switch(att)
    {
    case GL_COLOR_ATTACHMENT0:
        text = "GL_COLOR_ATTACHMENT0";
        break;
    case GL_COLOR_ATTACHMENT1:
        text = "GL_COLOR_ATTACHMENT1";
        break;
    case GL_COLOR_ATTACHMENT2:
        text = "GL_COLOR_ATTACHMENT2";
        break;
    case GL_COLOR_ATTACHMENT3:
        text = "GL_COLOR_ATTACHMENT3";
        break;
    case GL_COLOR_ATTACHMENT4:
        text = "GL_COLOR_ATTACHMENT4";
        break;
    case GL_COLOR_ATTACHMENT5:
        text = "GL_COLOR_ATTACHMENT5";
        break;
    case GL_COLOR_ATTACHMENT6:
        text = "GL_COLOR_ATTACHMENT6";
        break;
    case GL_COLOR_ATTACHMENT7:
        text = "GL_COLOR_ATTACHMENT7";
        break;
    case GL_COLOR_ATTACHMENT8:
        text = "GL_COLOR_ATTACHMENT8";
        break;
    case GL_DEPTH_ATTACHMENT:
        text = "GL_DEPTH_ATTACHMENT";
        break;
    case GL_STENCIL_ATTACHMENT:
        text = "GL_STENCIL_ATTACHMENT";
        break;
    case GL_DEPTH_STENCIL_ATTACHMENT:
        text = "GL_DEPTH_STENCIL_ATTACHMENT";
        break;
    }

    return text;
}


FrameBufferObjectMultisample::FrameBufferObjectMultisample(GLuint width, GLuint height, GLuint samples)
: m_width(width),
  m_height(height),
  m_filterParam(GL_NEAREST)
{
    GLsizei maxSamples = 0;
    glGetIntegerv(GL_MAX_SAMPLES, &maxSamples);
    std::cout <<"FRAMEBUFFERMULTISAMPLE::Samples: " << samples << std::endl;
    std::cout <<"FRAMEBUFFERMULTISAMPLE::MaxSamples: " << maxSamples << std::endl;

    //Color Renderbuffer
	glGenRenderbuffers(1, &m_colorRenderBuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, m_colorRenderBuffer);
	glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA, width, height);

    //Depth Renderbuffer
	glGenRenderbuffers(1, &m_depthRenderBuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, m_depthRenderBuffer);
	glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH_COMPONENT32F , width, height);

    //Framebuffer with 2 Renderbuffer Attachment
	glGenFramebuffers(1, &m_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_colorRenderBuffer);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depthRenderBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);



    //Texture
    glGenTextures(1, &m_texColor);
	glBindTexture(GL_TEXTURE_2D, m_texColor);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, m_filterParam);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, m_filterParam);
    //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16.0f); 
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, 0);

    //Depth Texture
    glGenTextures(1, &m_texDepth);
	glBindTexture(GL_TEXTURE_2D, m_texDepth);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, m_filterParam);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, m_filterParam);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16.0f); 
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    //Framebuffer with 2 Texture Attachment
	glGenFramebuffers(1, &m_fboResolve);
	glBindFramebuffer(GL_FRAMEBUFFER, m_fboResolve);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texColor, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_texDepth, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

    std::cout << "FRAMEBUFFERMULTISAMPLE::Size:" << width << "x" << height << std::endl;
}

FrameBufferObjectMultisample::~FrameBufferObjectMultisample()
{
    glDeleteTextures(1, &m_texDepth);
    glDeleteTextures(1, &m_texColor);

    glDeleteRenderbuffers(1, &m_colorRenderBuffer);
    glDeleteRenderbuffers(1, &m_depthRenderBuffer);

    glDeleteFramebuffers(1, &m_fbo);
    glDeleteFramebuffers(1, &m_fboResolve);

    m_texDepth = 0;
    m_texColor = 0;

    m_colorRenderBuffer = 0;
    m_depthRenderBuffer = 0;

    m_fbo = 0;
    m_fboResolve = 0;

    std::cout << "FRAMEBUFFERMULTISAMPLE::deleted" << std::endl;
}

void FrameBufferObjectMultisample::bind()
{
    glEnable(GL_MULTISAMPLE);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
}

void FrameBufferObjectMultisample::release()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_MULTISAMPLE);   
}

void FrameBufferObjectMultisample::blit()
{
	glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_fboResolve);
	glBlitFramebuffer(0, 0, m_width, m_height,  0, 0, m_width, m_height, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, m_filterParam);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FrameBufferObjectMultisample::blitDepth()
{
	glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_fboResolve);
	glBlitFramebuffer(0, 0, m_width, m_height,  0, 0, m_width, m_height, GL_DEPTH_BUFFER_BIT, m_filterParam);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FrameBufferObjectMultisample::blitColor()
{
	glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_fboResolve);
	glBlitFramebuffer(0, 0, m_width, m_height,  0, 0, m_width, m_height, GL_COLOR_BUFFER_BIT, m_filterParam);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLuint FrameBufferObjectMultisample::depthTex()
{
    return m_texDepth;
}

GLuint FrameBufferObjectMultisample::colorTex()
{
    return m_texColor;
}

void FrameBufferObjectMultisample::setDepthFromFBO(FrameBufferObjectMultisample *fbo)
{
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo->fboRender());
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, this->fboRender());

	glBlitFramebuffer(0, 0, m_width, m_height,  0, 0, m_width, m_height, GL_DEPTH_BUFFER_BIT, m_filterParam);

	glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

}

GLuint FrameBufferObjectMultisample::fboRender()
{
	return m_fbo;
}

GLuint FrameBufferObjectMultisample::fboTexture()
{
	return m_fboResolve;
}