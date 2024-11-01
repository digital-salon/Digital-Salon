#pragma once
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/freeglut.h>
//#include <GL/GL.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>

struct cudaGraphicsResource;

template<class T>
class VboResource
{
public:
	VboResource(const GLuint& glVbo) : GlVbo(glVbo)
	{
		glBindBuffer(GL_ARRAY_BUFFER, GlVbo);
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&CudaVbo, GlVbo, cudaGraphicsRegisterFlagsNone));
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	~VboResource() {}

	T* Map()
	{
		if (Mapped)
		{
			std::cerr << "Mapping resource that was already mapped!" << std::endl;
		}

		checkCudaErrors(cudaGraphicsMapResources(1, &CudaVbo, 0));
		size_t sizeCuda;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&DevicePtr, &sizeCuda, CudaVbo));
		Mapped = true;
		return DevicePtr;
	}

	void UnMap()
	{
		if (!Mapped)
		{
			std::cerr << "Unmapping resource that was not mapped!" << std::endl;
			return;
		}

		checkCudaErrors(cudaGraphicsUnmapResources(1, &CudaVbo, 0));
		Mapped = false;
	}


	GLuint GlVbo;
	cudaGraphicsResource* CudaVbo;

	T* GetPtr() { return DevicePtr; }
private:
	T* DevicePtr = nullptr;
	bool Mapped = false;

};