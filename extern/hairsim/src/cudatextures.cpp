#include "cudatextures.h"
#include <helper_cuda.h>
#include <cuda_gl_interop.h>

#include <glm/gtc/random.hpp>


void TextureResource::init(int3 size, int numCellValues, GLint wrap, float border_color)
{
  init(size.x, size.y, size.z, numCellValues, wrap, border_color);
}

void TextureResource::init(int w, int h, int d, int numCellValues, GLint wrap, float border_color)
{
  glGenTextures(1, &m_GLTexture);
  glBindTexture(GL_TEXTURE_3D, m_GLTexture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, wrap);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, wrap);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, wrap);
  float color[] = { border_color, border_color, border_color, border_color };
  glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, color);

  const int size = w*h*d*numCellValues;

#ifdef HALF_PRECISION
  int internalFormat = GL_R16F;
  int format = GL_RED;
  if (numCellValues == 2) { internalFormat = GL_RG16F; format = GL_RG; }
  else if (numCellValues == 3) { internalFormat = GL_RGB16F; format = GL_RGB; }
  else if (numCellValues == 4) { internalFormat = GL_RGBA16F; format = GL_RGBA; }
#else
  int internalFormat = GL_R32F;
  int format = GL_RED;
  if (numCellValues == 2) { internalFormat = GL_RG32F; format = GL_RG; }
  else if (numCellValues == 3) { internalFormat = GL_RGB32F; format = GL_RGB; }
  else if (numCellValues == 4) { internalFormat = GL_RGBA32F; format = GL_RGBA; }
#endif

  glTexImage3D(GL_TEXTURE_3D, 0, internalFormat, w, h, d, 0, format, GL_FLOAT, 0);

  checkCudaErrors(cudaGraphicsGLRegisterImage(
    &m_CudaResource,
    m_GLTexture,
    GL_TEXTURE_3D,
    cudaGraphicsRegisterFlagsSurfaceLoadStore));

  glBindTexture(GL_TEXTURE_3D, 0);
  m_Mapped = false;
}

void TextureResource::destroy()
{
  //cudaError_t err = cudaGraphicsUnregisterResource(CudaResource);
  //if (err != cudaSuccess)
  //{
  //	printf("cudaGraphicsUnregisterResource result [%d]:", err);
  //	if (err == cudaSuccess) printf("cudaSuccess");
  //	if (err == cudaErrorInvalidDevice) printf("cudaErrorInvalidDevice");
  //	if (err == cudaErrorInvalidValue) printf("cudaErrorInvalidValue");
  //	if (err == cudaErrorInvalidResourceHandle) printf("cudaErrorInvalidResourceHandle");
  //	if (err == cudaErrorUnknown) printf("cudaErrorUnknown");
  //	printf("\n");
  //}

  //checkCudaErrors(
  cudaGraphicsUnregisterResource(m_CudaResource);
  //);

    glDeleteTextures(1, &m_GLTexture);
}

cudaSurfaceObject_t TextureResource::mapAsSurface()
{
  checkCudaErrors(cudaGraphicsMapResources(1, &m_CudaResource));

  cudaArray_t viewCudaArray;
  checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, m_CudaResource, 0, 0));

  cudaResourceDesc viewCudaArrayResourceDesc;
  memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
  {
    viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
    viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
  }
  checkCudaErrors(cudaCreateSurfaceObject(&m_mappedSurface, &viewCudaArrayResourceDesc));

  m_Mapped = true;
  m_mappedAsSurface = true;

  return m_mappedSurface;
}

cudaTextureObject_t TextureResource::mapAsTexture(bool normalizedCoords)
{
  checkCudaErrors(cudaGraphicsMapResources(1, &m_CudaResource));

  cudaArray_t viewCudaArray;
  checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, m_CudaResource, 0, 0));

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = viewCudaArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.addressMode[2] = cudaAddressModeClamp;
  texDesc.normalizedCoords = normalizedCoords ? 1 : 0;

  checkCudaErrors(cudaCreateTextureObject(&m_mappedTexture, &resDesc, &texDesc, NULL));

  m_Mapped = true;
  m_mappedAsSurface = false;

  return m_mappedTexture;
}

void TextureResource::unmap()
{
  if (!m_Mapped)
  {
    //cerr << "ERROR: Unmapping a GPU resource that wasn't mapped" << endl;
    return;
  }

  if (m_mappedAsSurface)
    checkCudaErrors(cudaDestroySurfaceObject(m_mappedSurface));
  else
    checkCudaErrors(cudaDestroyTextureObject(m_mappedTexture));

  checkCudaErrors(cudaGraphicsUnmapResources(1, &m_CudaResource));

  m_Mapped = false;
}

TextureResource TextureResource::fromGLTexture(GLuint texture, GLuint type)
{
  glBindTexture(type, texture);

  TextureResource res;
  res.m_GLTexture = texture;

  cudaError_t err = cudaGraphicsGLRegisterImage(
    &res.m_CudaResource,
    res.m_GLTexture,
    type,
    cudaGraphicsRegisterFlagsSurfaceLoadStore);

  printf("cudaGraphicsGLRegisterImage error [%d]:", err);
  if (err == cudaSuccess) printf("cudaSuccess");
  if (err == cudaErrorInvalidDevice) printf("cudaErrorInvalidDevice");
  if (err == cudaErrorInvalidValue) printf("cudaErrorInvalidValue");
  if (err == cudaErrorInvalidResourceHandle) printf("cudaErrorInvalidResourceHandle");
  if (err == cudaErrorUnknown) printf("cudaErrorUnknown");
  printf("\n");

  glBindTexture(type, 0);

  res.m_Mapped = false;

  return res;
}

void DoubleTextureResource::init(int3 size, int numCellValues, GLint wrap, float border_color)
{
  init(size.x, size.y, size.z, numCellValues, wrap, border_color);
}

void DoubleTextureResource::init(int w, int h, int d, int numCellValues, GLint wrap, float border_color)
{
  res0.init(w, h, d, numCellValues, wrap, border_color);
  res1.init(w, h, d, numCellValues, wrap, border_color);
  current = &res0;
  next = &res1;
}

cudaTextureObject_t DoubleTextureResource::mapCurrentAsTexture()
{
  return current->mapAsTexture();
}

cudaSurfaceObject_t DoubleTextureResource::mapCurrentAsSurface()
{
  return current->mapAsSurface();
}

cudaTextureObject_t DoubleTextureResource::mapNextAsTexture()
{
  return next->mapAsTexture();
}

cudaSurfaceObject_t DoubleTextureResource::mapNextAsSurface()
{
  return next->mapAsSurface();
}

void DoubleTextureResource::unmapAll()
{
  if (current->m_Mapped) current->unmap();
  if (next->m_Mapped) next->unmap();
}
