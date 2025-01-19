#include "mnist_cudnn/Layer.h"

#include <random>

#include <cuda_runtime.h>
#include <curand.h>
#include <cassert>
#include <math.h>
#include <algorithm>

#include <sstream>
#include <fstream>
#include <iostream>

namespace CUDA_NETWORK
{
    /****************************************************************
     * Dropout definition                      *
     ****************************************************************/
    Dropout::Dropout(std::string name, Layer *inputFrom, float drop)
    {
        layerName = name;
        dropout = drop;

        CheckCudnnErrors(cudnnCreateDropoutDescriptor(&dropoutDesc));
    }

    Dropout::~Dropout()
    {
        cudnnDestroyDropoutDescriptor(dropoutDesc);
        if (states != nullptr) cudaFree(states);
        if (mPReserve != nullptr) cudaFree(mPReserve);
    }

    Blob<float> *Dropout::Forward(Blob<float> *input)
    {
        // initilaize input and output
        if (input_ == nullptr || batchSize_ != input->num)
        {
            input_ = input;
            batchSize_ = input->num;
            inputDesc  = input_->Tensor();

            if (output_ == nullptr)
                output_ = new Blob<float>(input->Shape());
            else
                output_->Reset(input->Shape());

            outputDesc = output_->Tensor();
            CheckCudnnErrors(cudnnDropoutGetStatesSize(cuda->Cudnn(), &stateSize));
            if (states != nullptr) CheckCudaErrors(cudaFree(states));
            CheckCudaErrors(cudaMalloc((void **) &states, stateSize));
            CheckCudnnErrors(cudnnDropoutGetReserveSpaceSize(inputDesc, &reserveSize));
            if (mPReserve != nullptr) CheckCudaErrors(cudaFree(mPReserve));
            CheckCudaErrors(cudaMalloc((void **) &mPReserve, reserveSize));

            CheckCudnnErrors(cudnnSetDropoutDescriptor(dropoutDesc,
                                                       cuda->Cudnn(),
                                                       dropout,
                                                       states,
                                                       stateSize,
                                                       seed));
        }

        CheckCudnnErrors(cudnnDropoutForward(cuda->Cudnn(),
                                             dropoutDesc,
                                             inputDesc,
                                             input_->Cuda(),
                                             outputDesc,
                                             output_->Cuda(),
                                             mPReserve,
                                             reserveSize));
        return output_;
    }

    Blob<float> *Dropout::Backward(Blob<float> *gradInput)
    {
        // initialize grad_output back-propagation space
        if (gradInput_ == nullptr || batchSize_ != gradInput->num)
        {
            gradOutput_ = gradInput;

        if (gradInput_ == nullptr)
            gradInput_ = new Blob<float>(input_->Shape());
        else
            gradInput_->Reset(input_->Shape());
        }

        CheckCudnnErrors(cudnnDropoutBackward(cuda->Cudnn(),
                                              dropoutDesc,
                                              outputDesc,
                                              gradOutput_->Cuda(),
                                              inputDesc,
                                              gradInput_->Cuda(),
                                              mPReserve, reserveSize));
                                              
        return gradInput_;
    }
}