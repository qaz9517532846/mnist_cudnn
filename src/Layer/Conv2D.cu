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

	/**
 	* Convolutional layer with bias
 	*/
	Conv2D::Conv2D(std::string name, Layer *inputFrom, int outChannels, int kernelSize, int stride, int padding, int dilation) :  outChannels(outChannels),
																																  kernelSize(kernelSize),
																																  stride(stride),
																																  padding(padding),
																																  dilation(dilation)
	{
		SetLayerRelationship(inputFrom);

		layerName = name;

		// create cudnn container handles
		cudnnCreateFilterDescriptor(&filterDesc);
		cudnnCreateConvolutionDescriptor(&convDesc);
		CheckCudnnErrors(
			cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride,  stride, dilation, dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

		dWorkspace = nullptr;
	}

	Conv2D::~Conv2D()
	{
		// distroy cudnn container resources
		cudnnDestroyFilterDescriptor(filterDesc);
		cudnnDestroyConvolutionDescriptor(convDesc);

		// terminate internal created blobs
		if(dWorkspace != nullptr)	cudaFree(dWorkspace);
	}

	void Conv2D::SetWorkspace()
	{
		size_t tempSize = 0;

		cudnnConvolutionFwdAlgoPerf_t			fwdAlgoPerfResults[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
		cudnnConvolutionBwdFilterAlgoPerf_t 	bwdFilterAlgoPerfResults[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
		cudnnConvolutionBwdDataAlgoPerf_t		bwdDataAlgoPerfResults[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];

		// forward
	#if CUDNN_MAJOR >= 8
		int algoMaxCount;
		CheckCudnnErrors(cudnnGetConvolutionForwardAlgorithmMaxCount(cuda->Cudnn(), &algoMaxCount));
		std::cout << this->layerName << ": Available Algorithm Count [FWD]: " << algoMaxCount << std::endl;
		CheckCudnnErrors(cudnnGetConvolutionForwardAlgorithm_v7(cuda->Cudnn(), inputDesc, filterDesc, convDesc, outputDesc, algoMaxCount, 0, fwdAlgoPerfResults));
		convFwdAlgo = fwdAlgoPerfResults[0].algo;
	#else
		CheckCudnnErrors(cudnnGetConvolutionForwardAlgorithm(cuda->Cudnn(), inputDesc, filterDesc, convDesc, outputDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convFwdAlgo));
	#endif
		CheckCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(cuda->Cudnn(), inputDesc, filterDesc, convDesc, outputDesc, convFwdAlgo, &tempSize));
		workspaceSize = std::max(workspaceSize, tempSize);

		// bwd - filter
	#if CUDNN_MAJOR >= 8
		CheckCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cuda->Cudnn(), &algoMaxCount));
		std::cout << this->layerName << ": Available Algorithm Count [BWD-filter]: " << algoMaxCount << std::endl;
		CheckCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cuda->Cudnn(), inputDesc, outputDesc, convDesc, filterDesc, algoMaxCount, 0, bwdFilterAlgoPerfResults));
		convBwdFilterAlgo = bwdFilterAlgoPerfResults[0].algo;
	#else
		CheckCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm(cuda->Cudnn(), inputDesc, outputDesc, convDesc, filterDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &convBwdFilterAlgo));
	#endif
		CheckCudnnErrors(cudnnGetConvolutionBackwardFilterWorkspaceSize(cuda->Cudnn(),
			inputDesc, outputDesc, convDesc, filterDesc,
			convBwdFilterAlgo, &tempSize));
		workspaceSize = std::max(workspaceSize, tempSize);

		// bwd - data
	#if CUDNN_MAJOR >= 8
		CheckCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cuda->Cudnn(), &algoMaxCount));
		std::cout << this->layerName << ": Available Algorithm Count [BWD-data]: " << algoMaxCount << std::endl;
		CheckCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm_v7(cuda->Cudnn(), filterDesc, outputDesc, convDesc, inputDesc, algoMaxCount, 0, bwdDataAlgoPerfResults));
		convBwdDataAlgo = bwdDataAlgoPerfResults[0].algo;
	#else
		CheckCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm(cuda->Cudnn(), filterDesc, outputDesc, convDesc, inputDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &convBwdDataAlgo));
	#endif
		CheckCudnnErrors(cudnnGetConvolutionBackwardDataWorkspaceSize(cuda->Cudnn(), filterDesc, outputDesc, convDesc, inputDesc, convBwdDataAlgo, &tempSize));
		workspaceSize = std::max(workspaceSize, tempSize);

		if(workspaceSize > 0)
		{
			if(dWorkspace != nullptr) CheckCudaErrors(cudaFree(dWorkspace));
			CheckCudaErrors(cudaMalloc((void**)&dWorkspace, workspaceSize));
		}
	}

	Blob<float> *Conv2D::Forward(Blob<float> *input)
	{
		input = GetInput(input);

		// initialize weights and bias
		if(weights_ == nullptr)
		{
			// initialize containers handles
			CheckCudnnErrors(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outChannels, input->channel, kernelSize, kernelSize));
			weights_ = new Blob<float>(outChannels, input->channel, kernelSize, kernelSize);
			biases_  = new Blob<float>(1, outChannels);	// bias size
			biasDesc = biases_->Tensor();
			weightsM_ = new Blob<float>(outChannels, input->channel, kernelSize, kernelSize);
			weightsV_ = new Blob<float>(outChannels, input->channel, kernelSize, kernelSize);
			biasesM_  = new Blob<float>(1, outChannels);	// bias size
			biasesV_  = new Blob<float>(1, outChannels);	// bias size
		}
 
		// initilaize input and output
		if(input == nullptr || batchSize_ != input->num)
		{
			// initialize input
			input_ = input;
			inputDesc = input_->Tensor();
			batchSize_  = input_->num;

			// initilaize output
			CheckCudnnErrors(cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &outputSize[0], &outputSize[1], &outputSize[2], &outputSize[3]));

			if (output_ == nullptr)
				output_  = new Blob<float>(outputSize);
			else
				output_->Reset(outputSize);

			outputDesc = output_->Tensor();

			// initialize workspace for cudnn
			SetWorkspace();

			// initialize weights
			if(loadPretrain && freeze_)
			{
				if(LoadParameter())
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

		CheckCudnnErrors(cudnnConvolutionForward(cuda->Cudnn(),
			&cuda->one,  inputDesc,  input_->Cuda(),
			filterDesc, weights_->Cuda(), convDesc, convFwdAlgo, dWorkspace,  workspaceSize,
			&cuda->zero, outputDesc, output_->Cuda()));

		CheckCudnnErrors(cudnnAddTensor(cuda->Cudnn(), 
			&cuda->one, biasDesc, biases_->Cuda(), 
			&cuda->one, outputDesc, output_->Cuda()));

	#if (DEBUG_CONV & 0x01)
		input_->Print(layerName + "::input", true, input->num, 28);
		weights_->Print(layerName + "::weight", true);
		biases_->Print(layerName + "::bias", true);
		output_->Print(layerName + "::output", true);
	#endif

		return output_;
	}

	Blob<float> *Conv2D::Backward(Blob<float> *gradOutput)
	{
		gradOutput = SumGradients(gradOutput);

		// initialize gradOutput back-propagation space
		if(gradInput_ == nullptr || batchSize_ != gradOutput->num)
		{
			gradOutput_  = gradOutput;
			gradWeights_ = new Blob<float>(weights_->Shape());
			gradBiases_  = new Blob<float>(1, biases_->channel);

			if(gradInput_ == nullptr)
				gradInput_ = new Blob<float>(input_->Shape());
			else
				gradInput_->Reset(input_->Shape());
		}

		// gradients of biases
		CheckCudnnErrors(cudnnConvolutionBackwardBias(cuda->Cudnn(), &cuda->one, outputDesc, gradOutput_->Cuda(), &cuda->zero, biasDesc, gradBiases_->Cuda()));
	
		// gradients of weights 
		CheckCudnnErrors(
			cudnnConvolutionBackwardFilter(cuda->Cudnn(),
				&cuda->one, 
				inputDesc, input_->Cuda(), 
				outputDesc, gradOutput_->Cuda(),
				convDesc, convBwdFilterAlgo, dWorkspace, workspaceSize,
				&cuda->zero, 
				filterDesc, gradWeights_->Cuda()));

		// gradients of input data
		if (!gradientStop)
			CheckCudnnErrors(
				cudnnConvolutionBackwardData(cuda->Cudnn(),
					&cuda->one, 
					filterDesc, weights_->Cuda(), 
					outputDesc, gradOutput_->Cuda(), 
					convDesc, convBwdDataAlgo, dWorkspace, workspaceSize,
					&cuda->zero, 
					inputDesc, gradInput_->Cuda()));

	#if (DEBUG_CONV & 0x02)
		std::cout << layerName << "[BACKWARD]" << std::endl;
		gradOutput_->Print(layerName + "::gradients", true);
		gradBiases_->Print(layerName + "gbias", true);
		gradWeights_->Print(layerName + "gfilter", true);
		if (!gradientStop)
			gradInput_->Print(layerName +"gdata", true);
	#endif

	#if (DEBUG_CONV & 0x04)
		gradOutput_->Print(layerName + "::gradients", true);
		gradBiases_->Print(layerName + "::gbias", true);
	#endif

		return gradInput_;
	}
}