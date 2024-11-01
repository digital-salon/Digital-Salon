#pragma once
#include <cuda_runtime.h>
#include <helper_cuda.h>

template<class T>
class CuBuffer
{
public:
    CuBuffer(const size_t& num_elements, const bool& host_buffer = false) :
        m_num_elements(num_elements), m_host_buffer(host_buffer)
    {
        m_d_ptr = nullptr;
        m_h_ptr = nullptr;

        checkCudaErrors(cudaMalloc(&m_d_ptr, m_num_elements * sizeof(T)));
        checkCudaErrors(cudaMemset(m_d_ptr, 0, m_num_elements * sizeof(T)));

        if (m_host_buffer)
        {
            m_h_ptr = new T[m_num_elements];
        }
    }

    ~CuBuffer()
    {
        checkCudaErrors(cudaFree(m_d_ptr));
        delete[] m_h_ptr;
    }

    void CopyHostToDevice(const T* data)
    {
        checkCudaErrors(cudaMemcpy((void*)&m_d_ptr[0], (void*)&data[0], m_num_elements * sizeof(T), cudaMemcpyHostToDevice));
    }

    void CopyHostToDevice()
    {
        checkCudaErrors(cudaMemcpy((void*)&m_d_ptr[0], (void*)&m_h_ptr[0], m_num_elements * sizeof(T), cudaMemcpyHostToDevice));
    }

    void CopyDeviceToHost()
    {
        checkCudaErrors(cudaMemcpy((void*)&m_h_ptr[0], (void*)&m_d_ptr[0], m_num_elements * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void SetHostDevice(const T* data)
    {
        CopyHostToDevice(data);
        CopyDeviceToHost();
    }

    size_t NumElements() { return m_num_elements; }
    T* DevPtr() { return m_d_ptr; }
    T* HostPtr() { return m_h_ptr; }

private:
    T* m_d_ptr;
    T* m_h_ptr;
    const size_t m_num_elements;
    const bool m_host_buffer;
};