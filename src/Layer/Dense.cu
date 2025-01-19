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
    * Dense Layer                                                  *
    ****************************************************************/

	Dense::Dense(std::string name, Layer *inputFrom, int outSize)
	{
		SetLayerRelationship(inputFrom);
		layerName = name;
		outputSize = outSize;
	}

	Dense::~Dense()
	{
		if(dOneVec != nullptr) cudaFree(dOneVec);
	}

	__global__ void InitOneVec(float* dOneVec, size_t length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i >= length) return;

		dOneVec[i] = 1.f;
	}

	Blob<float> *Dense::Forward(Blob<float> *input)
	{
		input = GetInput(input);

		// initialize weights and biases
		if(weights_ == nullptr)
		{
			// setup parameter size information
			inputSize  = input->channel * input->height * input->width;
		
			// initialize weight, bias, and output
			weights_ = new Blob<float>(1, 1, inputSize, outputSize);
			biases_  = new Blob<float>(1, 1, outputSize);
			weightsM_ = new Blob<float>(1, 1, inputSize, outputSize);
			weightsV_ = new Blob<float>(1, 1, inputSize, outputSize);
			biasesM_ = new Blob<float>(1, 1, outputSize);
			biasesV_ = new Blob<float>(1, 1, outputSize);
		}

		// initilaize input and output
		if(input_ == nullptr || batchSize_ != input_->num)
		{
			input_ = input;
			batchSize_  = input->num;

			if(output_ == nullptr)
				output_  = new Blob<float>(batchSize_, outputSize);
			else
				output_->Reset(batchSize_, outputSize);
		
			output_->Tensor();

			if(dOneVec != nullptr) cudaFree(dOneVec);

			CheckCudaErrors(cudaMalloc((void**)&dOneVec, sizeof(float) * batchSize_));
			//config = GetGpuLaunchConfig(batchSize_, InitOneVec, 0, 0);
			InitOneVec<<<(batchSize_ + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D, BLOCK_DIM_1D>>>(dOneVec, batchSize_);

			// initialize weights and biases
			if(loadPretrain && freeze_)
			{
				if(LoadParameter())
				{
					std::cout << "error occurred.." << std::endl;
					exit(-1);
				}
			}
			else if(!freeze_)
			{
				InitWeightBias();
			}
			else
			{
				/* do nothing */
			}
		}

		// output = weights^T * input (without biases)
		CheckCublasErrors(
			cublasSgemm(cuda->Cublas(),
				CUBLAS_OP_T, CUBLAS_OP_N, 
				outputSize, batchSize_, inputSize,
				&cuda->one,  
				weights_->Cuda(), inputSize, 
				input_->Cuda(), inputSize,
				&cuda->zero, 
				output_->Cuda(),  outputSize));

		// output += biases * dOneVec ^ T
		CheckCublasErrors(
			cublasSgemm(cuda->Cublas(),
				CUBLAS_OP_N, CUBLAS_OP_N, 
				outputSize, batchSize_, 1,
				&cuda->one, 
				biases_->Cuda(), outputSize, 
				dOneVec, 1, 
				&cuda->one, 
				output_->Cuda(), outputSize));

	#if (DEBUG_DENSE & 0x01)
		input->Print(layerName + "::input",  true);
		weights->Print(layerName + "::weight", true);
		biases->Print(layerName + "::bias",   true);
		output->Print(layerName + "::output", true);
	#endif // DEBUG_DENSE

		return output_;
	}

	Blob<float> *Dense::Backward(Blob<float> *gradOutput)
	{
		gradOutput = SumGradients(gradOutput);

		if(gradWeights_ == nullptr)
		{
			gradWeights_ = new Blob<float>(weights_->Shape());
			gradBiases_  = new Blob<float>(biases_->Shape());
		}

		if(gradInput_ == nullptr || batchSize_ != gradOutput->num)
		{
			gradOutput_ = gradOutput;

			if (gradInput_ == nullptr)
				gradInput_   = new Blob<float>(input_->Shape());
			else
				gradInput_->Reset(input_->Shape());
		}

		// db = (dy) * d_one_vec
		cublasSgemv(cuda->Cublas(),
				CUBLAS_OP_N,
				outputSize, batchSize_,
				&cuda->one,
				gradOutput->Cuda(), outputSize,
				dOneVec, 1,
				&cuda->zero,
				gradBiases_->Cuda(), 1);

		// dw = x * (dy)^T
		cublasSgemm(cuda->Cublas(),
			CUBLAS_OP_N, CUBLAS_OP_T,
			inputSize, outputSize, batchSize_,
			&cuda->one,
			input_->Cuda(),        inputSize,
			gradOutput->Cuda(),   outputSize,
			&cuda->zero,
			gradWeights_->Cuda(),  inputSize);

		// dx = W * dy
		if (!gradientStop)
			cublasSgemm(cuda->Cublas(),
				CUBLAS_OP_N, CUBLAS_OP_N,
				inputSize, batchSize_, outputSize,
				&cuda->one,
				weights_->Cuda(),    inputSize,
				gradOutput->Cuda(), outputSize,
				&cuda->zero, 
				gradInput_->Cuda(),  inputSize);

	#if (DEBUG_DENSE & 0x02)
		std::cout << layerName << "[BACKWARD]" << std::endl;
		gradOutput->Print(layerName + "::gradients", true, gradOutput->num);
		gradWeights->Print(layerName + "::gfilter", true);
		gradBiases->Print(layerName + "::gbias", true);
		if(!gradientStop) gradInput->Print(layerName + "::gdata", true);
	#endif // DEBUG_DENSE

		return gradInput_;
	}
}