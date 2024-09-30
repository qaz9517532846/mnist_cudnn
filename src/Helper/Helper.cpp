#include "mnist_cudnn/Helper.h"

namespace CUDA_NETWORK
{
    CudaContext::CudaContext()
    {
        cublasCreate(&cublasHandle);
        CheckCudaErrors(cudaGetLastError());
        CheckCudnnErrors(cudnnCreate(&cudnnHandle));
    }

    CudaContext::~CudaContext()
    {
        cublasDestroy(cublasHandle);
        CheckCudnnErrors(cudnnDestroy(cudnnHandle));
    }

    cublasHandle_t CudaContext::Cublas()
    { 
        //std::cout << "Get cublas request" << std::endl; getchar();
        return cublasHandle; 
    }

    cudnnHandle_t CudaContext::Cudnn()
    {
        return cudnnHandle;
    }
}