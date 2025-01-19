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
 	 * Layer definition                                             *
 	****************************************************************/
	Pooling::Pooling(std::string name, Layer *inputFrom, int kernelSize, int padding, int stride, cudnnPoolingMode_t mode) :  poolKernelSize(kernelSize),
																											poolPadding(padding),
																											poolStride(stride),
																											poolMode(mode)
	{
		SetLayerRelationship(inputFrom);

		layerName = name;
		cudnnCreatePoolingDescriptor(&poolDesc);
		cudnnSetPooling2dDescriptor(poolDesc, poolMode, CUDNN_PROPAGATE_NAN, poolKernelSize, poolKernelSize, poolPadding, poolPadding, poolStride, poolStride);
	}

	Pooling::~Pooling()
	{
		cudnnDestroyPoolingDescriptor(poolDesc);
	}

	Blob<float> *Pooling::Forward(Blob<float> *input)
	{
		input = GetInput(input);

		if(input == nullptr || batchSize_ != input->num)
		{
			input_ = input;

			// resource initialize
			inputDesc = input->Tensor();
			batchSize_ = input->num;
		
			// setting output
			cudnnGetPooling2dForwardOutputDim(poolDesc, inputDesc, &outputSize[0], &outputSize[1], &outputSize[2], &outputSize[3]);
			if(output_ == nullptr)
				output_ = new Blob<float>(outputSize);
			else
				output_->Reset(outputSize);
		
			outputDesc = output_->Tensor();
		}

		cudnnPoolingForward(cuda->Cudnn(), poolDesc,
			&cuda->one,   inputDesc,  input->Cuda(),
			&cuda->zero,  outputDesc, output_->Cuda());

		return output_;
	}

	Blob<float> *Pooling::Backward(Blob<float> *gradOutput)
	{
		gradOutput = SumGradients(gradOutput);

		if (gradInput_ == nullptr || batchSize_ != gradOutput->num)
		{
			gradOutput_ = gradOutput;

			if (gradInput_ == nullptr)
				gradInput_ = new Blob<float>(input_->Shape());
			else
				gradInput_->Reset(input_->Shape());
		}

		CheckCudnnErrors(
			cudnnPoolingBackward(cuda->Cudnn(), poolDesc,
				&cuda->one,  
				outputDesc, output_->Cuda(),
				outputDesc, gradOutput_->Cuda(), 
				inputDesc,  input_->Cuda(), 
				&cuda->zero, 
				inputDesc,  gradInput_->Cuda()));

		return gradInput_;
	}
}