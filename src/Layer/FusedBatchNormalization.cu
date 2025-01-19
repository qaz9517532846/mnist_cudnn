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
     * FusedBatchNormalization definition                           *
     ****************************************************************/
    FusedBatchNormalization::FusedBatchNormalization(std::string name, Layer *inputFrom, cudnnBatchNormMode_t mode)
    {
        SetLayerRelationship(inputFrom);

        layerName = name;
        mode_ = mode;

        CheckCudnnErrors(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc));
    }

    FusedBatchNormalization::~FusedBatchNormalization()
    {
        cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc);
    }

    Blob<float> *FusedBatchNormalization::Forward(Blob<float> *input)
    {
        input = GetInput(input);
        // initialize weights and biases
        if (weights_ == nullptr)
        {
            // initialize weight, bias
            size = input->channel;
            weights_ = new Blob<float>(1, size, 1, 1);
            biases_ = new Blob<float>(1, size, 1, 1);
            weightsM_ = new Blob<float>(1, size, 1, 1);
            weightsV_ = new Blob<float>(1, size, 1, 1);
            biasesM_ = new Blob<float>(1, size, 1, 1);
            biasesV_ = new Blob<float>(1, size, 1, 1);
        }

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

            cudaMalloc(&resultRunningMean, sizeof(float) * size);
            cudaMalloc(&resultRunningVariance, sizeof(float) * size);

            cudaMalloc(&resultSaveMean, sizeof(float) * size);
            cudaMalloc(&resultSaveInvVariance, sizeof(float) * size);

            CheckCudnnErrors(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDesc, inputDesc, mode_));

            // initialize weights and biases
            if (loadPretrain && !freeze_)
            {
                if (LoadParameter())
                {
                    std::cout << "error occurred.." << std::endl;
                    exit(-1);
                }
            }
            else if (!freeze_)
            {
                InitWeightBias();
            }
            else
            {
                /* do nothing */
            }
        }
        
        //y = beta*y + alpha *[bnBias + (bnScale * (x-estimatedMean) / sqrt(epsilon + estimatedVariance)]
        CheckCudnnErrors(cudnnBatchNormalizationForwardTraining(cuda->Cudnn(),
                                                                mode_,
                                                                &cuda->one,
                                                                &cuda->zero,
                                                                inputDesc,
                                                                input->Cuda(),
                                                                outputDesc,
                                                                output_->Cuda(),
                                                                bnScaleBiasMeanVarDesc,
                                                                weights_->Cuda(),
                                                                biases_->Cuda(),
                                                                cuda->one,
                                                                resultRunningMean,
                                                                resultRunningVariance,
                                                                epison,
                                                                resultSaveMean,
                                                                resultSaveInvVariance));

    #if (DEBUG_FBN & 0x01)
        std::cout << layerName << "[FORWARD]" << std::endl;
        input_->print(layerName + "::input", true, input_->n(), input_->h());
        weights_->print(layerName + "::weight", true, weights_->n(), weights_->c());
        biases_->print(layerName + "::bias", true, biases_->n(), biases_->c());
        output_->print(layerName + "::output", true, output_->n(), output_->h());
    #endif

        return output_;
    }

    Blob<float> *FusedBatchNormalization::Backward(Blob<float> *gradOutput)
    {
        gradOutput = SumGradients(gradOutput);

        // initialize grad_output back-propagation space
        if (gradInput_ == nullptr || batchSize_ != gradOutput->num)
        {
            gradOutput_ = gradOutput;
            gradWeights_ = new Blob<float>(weights_->Shape());
            gradBiases_ = new Blob<float>(biases_->Shape());

            if (gradInput_ == nullptr)
                gradInput_ = new Blob<float>(input_->Shape());
            else
                gradInput_->Reset(input_->Shape());
        }
        
        CheckCudnnErrors(cudnnBatchNormalizationBackward(cuda->Cudnn(),
                                                         mode_,
                                                         &cuda->one,
                                                         &cuda->zero,
                                                         &cuda->one,
                                                         &cuda->zero,
                                                         inputDesc,
                                                         input_->Cuda(),
                                                         outputDesc,
                                                         gradOutput_->Cuda(),
                                                         inputDesc,
                                                         gradInput_->Cuda(),
                                                         bnScaleBiasMeanVarDesc,
                                                         weights_->Cuda(),
                                                         gradWeights_->Cuda(),
                                                         gradBiases_->Cuda(),
                                                         epison,
                                                         resultSaveMean,
                                                         resultSaveInvVariance));

    #if (DEBUG_FBN & 0x02)
        std::cout << layerName << "[BACKWARD]" << std::endl;
        gradOutput_->print(layerName + "::gradients", true, gradOutput_->num);
        gradWeights_->print(layerName + "::gfilter", true);
        gradBiases_->print(layerName + "::gbias", true);
        if (!gradientStop_)
            gradInput_->print(layerName + "::gdata", true);
    #endif // DEBUG_FBN

        return gradInput_;
    }
}