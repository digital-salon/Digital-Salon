#pragma once
#include <algorithm>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
//#include <gl/GL.h>
#include <GL/gl.h>

//#define HALF_PRECISION
typedef unsigned short ushort;

#ifdef HALF_PRECISION
typedef ushort ftype;
#else
typedef float ftype;
#endif


struct cudaGraphicsResource;

class TextureResource
{
public:
  GLuint m_GLTexture;
  cudaGraphicsResource * m_CudaResource;

  void init(int w, int h, int d, int numCellValues, GLint wrap, float border_color);
  void init(int3 size, int numCellValues, GLint wrap, float border_color);
  void destroy();

  static TextureResource fromGLTexture(GLuint texture, GLuint type = GL_TEXTURE_2D);

  cudaSurfaceObject_t mapAsSurface();
  cudaTextureObject_t mapAsTexture(bool normalizedCoords = true);
  void unmap();

  bool m_Mapped;

private:
  bool m_mappedAsSurface;

  cudaSurfaceObject_t m_mappedSurface;
  cudaTextureObject_t m_mappedTexture;
};

class DoubleTextureResource
{
private:
  TextureResource res0, res1;
  TextureResource *current, *next;

public:
  ~DoubleTextureResource() { destroy(); }
  void init(int w, int h, int d, int numCellValues, GLint wrap, float border_color);
  void init(int3 size, int numCellValues, GLint wrap, float border_color);
  void destroy() { res0.destroy(); res1.destroy(); }

  cudaTextureObject_t mapCurrentAsTexture();
  cudaSurfaceObject_t mapCurrentAsSurface();
  cudaTextureObject_t mapNextAsTexture();
  cudaSurfaceObject_t mapNextAsSurface();
  void unmapAll();

  GLuint getCurrentGLTexture() { return current->m_GLTexture; }

  void swap() { std::swap(current, next); } // Swap only when not mapped!
};
