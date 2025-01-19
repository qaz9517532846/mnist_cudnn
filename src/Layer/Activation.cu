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
 	* Activation Layer                                             *
 	****************************************************************/

	Activation::Activation(std::string name, Layer *inputFrom, cudnnActivationMode_t mode, float coef)
	{
		SetLayerRelationship(inputFrom);

		layerName = name;
		actMode = mode;
		actCoef = coef;

		cudnnCreateActivationDescriptor(&actDesc);
		cudnnSetActivationDescriptor(actDesc, actMode, CUDNN_PROPAGATE_NAN, actCoef);
	}

	Activation::~Activation()
	{
		cudnnDestroyActivationDescriptor(actDesc);
	}

	Blob<float> *Activation::Forward(Blob<float> *input)
	{
		input = GetInput(input);

		if(input == nullptr || batchSize_ != input->num)
		{
			input_ = input;
			inputDesc = input_->Tensor();
			batchSize_ = input_->num;

			if(output_ == nullptr)
				output_ = new Blob<float>(input_->Shape());
			else
				output_->Reset(input_->Shape());

			outputDesc = output_->Tensor();
		}

		cudnnActivationForward(cuda->Cudnn(),
			actDesc,
			&cuda->one,
			inputDesc,
			input_->Cuda(),
			&cuda->zero,
			outputDesc,
			output_->Cuda());

		return output_;
	}

	Blob<float> *Activation::Backward(Blob<float> *gradOutput)
	{
		gradOutput = SumGradients(gradOutput);

		if (gradInput_ == nullptr || batchSize_ != gradOutput->num)
		{
			gradOutput = gradOutput;

			if (gradInput_ == nullptr)
				gradInput_ = new Blob<float>(input_->Shape());
			else
				gradInput_->Reset(input_->Shape());		
		}

		cudnnActivationBackward(cuda->Cudnn(),
			actDesc,
			&cuda->one, 
			outputDesc, output_->Cuda(),
			outputDesc, gradOutput->Cuda(), 
			inputDesc, input_->Cuda(), 
			&cuda->zero, 
			inputDesc, gradInput_->Cuda());

		return gradInput_;
	}
}