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
     * Local Response Normalization definition                      *
    ****************************************************************/
    LRN::LRN(std::string name, unsigned n, double alpha, double beta, double k)
    {
        layerName = name;
        lrnN = n;
        lrnAlpha = alpha;
        lrnBeta = beta;
        lrnK = k;

        CheckCudnnErrors(cudnnCreateLRNDescriptor(&normDesc));
        CheckCudnnErrors(cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK));
    }

    LRN::~LRN()
    {
        cudnnDestroyLRNDescriptor(normDesc);
    }

    Blob<float> *LRN::Forward(Blob<float> *input)
    {
        // initilaize input and output
        if (input_ == nullptr || batchSize_ != input->num)
        {
            input_ = input;
            batchSize_ = input->num;
            inputDesc = input_->Tensor();

            if (output_ == nullptr)
                output_ = new Blob<float>(input->Shape());
            else
                output_->Reset(input->Shape());

            outputDesc = output_->Tensor();
        }
        
        CheckCudnnErrors(cudnnLRNCrossChannelForward(cuda->Cudnn(),
                                                     normDesc,
                                                     CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                                     &cuda->one,
                                                     inputDesc,
                                                     input_->Cuda(),
                                                     &cuda->zero,
                                                     outputDesc,
                                                     output_->Cuda()));
    #if (DEBUG_CONV & 0x01)
        input_->print(name_ + "::input", true, input_->n(), 28);
        output_->print(name_ + "::output", true);
    #endif

        return output_;
    }

    Blob<float> *LRN::Backward(Blob<float> *gradInput)
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

        CheckCudnnErrors(cudnnLRNCrossChannelBackward(cuda->Cudnn(),
                                                      normDesc,
                                                      CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                                      &cuda->one,
                                                      outputDesc,
                                                      output_->Cuda(),
                                                      outputDesc,
                                                      gradOutput_->Cuda(),
                                                      inputDesc,
                                                      input_->Cuda(),
                                                      &cuda->zero,
                                                      inputDesc,
                                                      gradInput_->Cuda()));

        return gradInput_;
    }
}