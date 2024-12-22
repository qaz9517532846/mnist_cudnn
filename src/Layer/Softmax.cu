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
	 * Softmax definition                                           *
	 ****************************************************************/

	Softmax::Softmax(std::string name)
	{
		layerName = name;
	}

	Softmax::~Softmax()
	{

	}

	Blob<float> *Softmax::Forward(Blob<float> *input)
	{
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

	#if (DEBUG_SOFTMAX & 0x01)
		std::cout << layerName << "[FORWARD]" << std::endl;
		input_->Print(layerName + "::input", true, input_->num);
	#endif

		CheckCudnnErrors(
			cudnnSoftmaxForward(cuda->Cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
				&cuda->one,  inputDesc,  input_->Cuda(),
				&cuda->zero, outputDesc, output_->Cuda()));

	#if (DEBUG_SOFTMAX & 0x01)
		output_->Print(layerName + "::output", true, input_->num);
	#endif

		return output_;
	}

	Blob<float> *Softmax::Backward(Blob<float> *target)
	{
		CheckCudaErrors(cudaDeviceSynchronize());

		if(gradInput_ == nullptr || batchSize_ != target->num)
		{
			if (gradInput_ == nullptr)
				gradInput_ = new Blob<float>(input_->Shape());
			else
		 		gradInput_->Reset(input_->Shape());
		}

		// set grad_input_ as predict
		CheckCudaErrors(cudaMemcpyAsync(gradInput_->Cuda(), 
			output_->Cuda(), output_->BufSize(), 
			cudaMemcpyDeviceToDevice));

		// set gradInput = predict - target	
		CheckCublasErrors(
			cublasSaxpy(cuda->Cublas(), target->Length(),
				&cuda->minusOne, target->Cuda(), 1,
				gradInput_->Cuda(), 1));

		// normalize the grad_output by the batch size
		int gradOutputSize = target->num * target->channel * target->height * target->width;
		float scale = 1.f / static_cast<float>(target->num);
		CheckCublasErrors(cublasSscal(cuda->Cublas(), gradOutputSize, &scale, gradInput_->Cuda(), 1));

	#if (DEBUG_SOFTMAX & 0x02)
		std::cout << layerName << "[BACKWARD]" << std::endl;
		input_->Print( layerName + "::input", true);
		output_->Print(layerName + "::predict", true);
		target->Print( layerName + "::y", true, target->num);
		gradInput_->Print(layerName + "::dx", true, target->num);
	#endif

		return gradInput_;
	}

	float Softmax::GetLoss(Blob<float> *target)
	{
		return loss.Loss(output_, target);
	}

	int Softmax::GetAccuracy(Blob<float> *target)
	{
		int batchSize = output_->num;
		int outputSize = output_->Size();

		assert(batchSize == target->num);
		assert(outputSize == target->Size());

		float *hOutput, *hTarget;
		int idxOutput, idxTarget;
		int hitCount = 0;

		// get predicts and targets
		hOutput = output_->To(HOST);
		hTarget = target->To(HOST);

		// idxOutput = idxTarget = 0;
		for(int b = 0; b < batchSize; b++)
		{
			idxOutput = 0;
			idxTarget = 0;

			for (int i = 1; i < 10; i++)
			{
				if (hOutput[b * outputSize + i] > hOutput[b * outputSize + idxOutput])
					idxOutput = i;
				if (hTarget[b * outputSize + i] > hTarget[b * outputSize + idxTarget])
					idxTarget = i;
			}

			if (idxOutput == idxTarget)
				hitCount++;
		}

		return hitCount;
	}
}