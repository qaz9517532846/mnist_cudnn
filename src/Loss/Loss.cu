#include "mnist_cudnn/Loss.h"
#include "mnist_cudnn/Helper.h"

#include <cassert>
#include <cuda_runtime.h>

namespace CUDA_NETWORK
{
    CrossEntropyLoss::CrossEntropyLoss()
    {
        cudaMalloc((void**)&dLoss, sizeof(float));
    }

    CrossEntropyLoss::~CrossEntropyLoss()
    {
        if(dLoss != nullptr)
            cudaFree(dLoss);
            dLoss = nullptr;

        if(dWorkspace != nullptr)
            cudaFree(dWorkspace);
    }

    __device__ float Clip(float prediction, float epsilon = 1e-12)
    {
        return fmin(fmax(prediction, epsilon), 1.f - epsilon);
    }

    __global__ void SoftMaxLossKernel(float *reducedLoss, float *predict, float *target, float *workspace, int batchSize, int numOutputs)
    {
        int batchIdx = blockDim.x * blockIdx.x + threadIdx.x;

        extern __shared__ float sData[];
        float loss = 0.f;

        // each thread calculate entropy for each data and accumulate to shared memory
        if (batchIdx > 0) return;

        // each thread calculate entropy for each data and accumulate to shared memory
        for (int c = 0; c < numOutputs; c++)
            loss += target[batchIdx * numOutputs + c] * logf(predict[batchIdx * numOutputs + c]);
        workspace[batchIdx] = -loss;

        // then, we do reduction the result to calculate loss using 1 thread block
        if (blockIdx.x > 0) return;

        // cumulate workspace data
        sData[threadIdx.x] = 0.f;
        for (int i = 0; i < batchSize; i += blockDim.x)
        {
            sData[threadIdx.x] += workspace[threadIdx.x + i];
        }

        __syncthreads();

        // reduction
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x + stride < batchSize)
                sData[threadIdx.x] += sData[threadIdx.x + stride];

            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            reducedLoss[blockIdx.x] = sData[0];
        }
    }

    void CrossEntropyLoss::InitWorkspace(int batchSize)
    {
        if (dWorkspace == nullptr)
            cudaMalloc((void**)&dWorkspace, sizeof(float) * batchSize);
    }
    
    float CrossEntropyLoss::Loss(Blob<float> *predict, Blob<float> *target)
    {
        int numSms;
        int numBlocksPerSm;
        cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, SoftMaxLossKernel, BLOCK_DIM_1D, BLOCK_DIM_1D * sizeof(float));

        int batchSize = target->num;
        int numOutputs = target->channel;

        InitWorkspace(batchSize);

    #if (DEBUG_LOSS)
        std::cout << "[[ LOSS ]]" << std::endl;
        predict->Print("predict", true);
        target->Print("target", true);
    #endif // DEBUG_LOSS

        int numBlocks = min(numBlocksPerSm * numSms, (target->Size() + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D);

        SoftMaxLossKernel<<<numBlocks, BLOCK_DIM_1D, BLOCK_DIM_1D * sizeof(float), 0 >>>(dLoss, predict->Cuda(), target->Cuda(), dWorkspace, batchSize, numOutputs);
        cudaMemcpy(&hLoss, dLoss, sizeof(float), cudaMemcpyDeviceToHost);
    
        // batch mean loss 
        return hLoss / float(batchSize);
    }
}