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
 	* RNN layer with bias
 	*/
	RNN::RNN(std::string name, Layer *inputFrom, const int hiddenSize, const int numLayer, double dropout, cudnnDirectionMode_t bidirectional, cudnnRNNMode_t mode)
	{
		layerName = name;

		// create cudnn container handles
		cudnnCreateRNNDescriptor(&rnnDesc);
		cudnnCreateDropoutDescriptor(&dropoutDesc);

		hiddenSize_ = hiddenSize;
		numLayer_ = numLayer;
		inputSize_ = 28;
		seqLen_ = 28;
		bidirectional_ = bidirectional;
		dropout_ = dropout;
		mode_ = mode;

		dWorkspace = nullptr;
		reserveSpace = nullptr;
	}

	RNN::~RNN()
	{
		// distroy cudnn container resources
		cudnnDestroyRNNDescriptor(rnnDesc);

		// terminate internal created blobs
		if(dWorkspace != nullptr)	cudaFree(dWorkspace);
		if(reserveSpace != nullptr)	cudaFree(reserveSpace);
	}

	void RNN::SetWorkspace()
	{
		dimHidden[0] = numLayer_;
        dimHidden[1] = batchSize_;
        dimHidden[2] = hiddenSize_;
        strideHidden[0] = dimHidden[1] * dimHidden[2];
        strideHidden[1] = dimHidden[2];
        strideHidden[2] = 1;

		// aux vars for function calls
		float paddingFill = 0.0;
		seqLengthArray = new int[batchSize_];

		for (int i = 0; i < batchSize_; ++i)
		{
			seqLengthArray[i] = seqLen_;
		}

		cudnnCreateRNNDataDescriptor(&xDesc);
		cudnnCreateRNNDataDescriptor(&yDesc);

		cudnnSetRNNDataDescriptor(xDesc,
								  CUDNN_DATA_FLOAT,
								  CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
								  seqLen_,
								  batchSize_,
								  inputSize_,
								  seqLengthArray,
								  &paddingFill);

		cudnnSetRNNDataDescriptor(yDesc,
                              	  CUDNN_DATA_FLOAT,
								  CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
								  seqLen_,
								  batchSize_,
								  hiddenSize_,
								  seqLengthArray,
								  &paddingFill);

		// setup hidden tensors
		cudnnCreateTensorDescriptor(&hDesc);
		cudnnCreateTensorDescriptor(&cDesc);
		cudnnSetTensorNdDescriptor(hDesc, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden);
		cudnnSetTensorNdDescriptor(cDesc, CUDNN_DATA_FLOAT, 3, dimHidden, strideHidden);


		// setup dropout descriptor
		cudnnCreateDropoutDescriptor(&dropoutDesc);

		unsigned long long seed = 123ULL;

		cudnnDropoutGetStatesSize(cuda->Cudnn(), &dropoutStatesSize); 
		CheckCudaErrors(cudaMalloc(&dropoutStates, dropoutStatesSize));
		cudnnSetDropoutDescriptor(dropoutDesc, cuda->Cudnn(), dropout_, dropoutStates, dropoutStatesSize, seed);

		cudnnSetRNNDescriptor_v8(rnnDesc,
								 CUDNN_RNN_ALGO_STANDARD,
								 mode_,
								 CUDNN_RNN_NO_BIAS,
								 bidirectional_,
								 CUDNN_LINEAR_INPUT,
								 CUDNN_DATA_FLOAT,
								 CUDNN_DATA_FLOAT,
								 CUDNN_DEFAULT_MATH,
								 inputSize_,
								 hiddenSize_,
								 hiddenSize_,
								 numLayer_,
								 dropoutDesc,
								 0);

		CheckCudnnErrors(cudnnCreateTensorDescriptor(&weightDesc));
		CheckCudnnErrors(cudnnCreateTensorDescriptor(&biasDesc));

		cudnnGetRNNWeightSpaceSize(cuda->Cudnn(), rnnDesc, &weightSize);
	
		cudnnGetRNNTempSpaceSizes(cuda->Cudnn(), rnnDesc, CUDNN_FWD_MODE_TRAINING, xDesc, &workspaceSize, &reserveSize);

		if(batchSize_ > 0)
		{
			CheckCudaErrors(cudaMalloc(&devSeqLengthArray, batchSize_ * sizeof(int)));
			CheckCudaErrors(cudaMemcpy(devSeqLengthArray, seqLengthArray, batchSize_ * sizeof(int), cudaMemcpyHostToDevice));
		}

		if(workspaceSize > 0)
		{
			if(dWorkspace != nullptr) CheckCudaErrors(cudaFree(dWorkspace));
			CheckCudaErrors(cudaMalloc((void**)&dWorkspace, workspaceSize));
		}

		if(reserveSize > 0)
		{
			if(reserveSpace != nullptr) CheckCudaErrors(cudaFree(reserveSpace));
			CheckCudaErrors(cudaMalloc((void**)&reserveSpace, reserveSize));
		}
	}

	Blob<float> *RNN::Forward(Blob<float> *input)
	{
		// initilaize input and output
		if(input == nullptr || batchSize_ != input->num)
		{
			// initialize input
			printf("initialize input\n");
			input_ = input;
			batchSize_  = input_->num;

			// initilaize hx
			if (hx_ == nullptr)
				hx_  = new Blob<float>(batchSize_, input->channel, numLayer_, hiddenSize_);
			else
				hx_->Reset(batchSize_, input->channel, numLayer_, hiddenSize_);

			// initilaize hy
			if (hy_ == nullptr)
				hy_  = new Blob<float>(batchSize_, input->channel, numLayer_, hiddenSize_);
			else
				hy_->Reset(batchSize_, input->channel, numLayer_, hiddenSize_);

			// initilaize cx
			if (cx_ == nullptr)
				cx_  = new Blob<float>(batchSize_, input->channel, numLayer_, hiddenSize_);
			else
				cx_->Reset(batchSize_, input->channel, numLayer_, hiddenSize_);

			// initilaize cy
			if (cy_ == nullptr)
				cy_  = new Blob<float>(batchSize_, input->channel, numLayer_, hiddenSize_);
			else
				cy_->Reset(batchSize_, input->channel, numLayer_, hiddenSize_);

			// initilaize dhx
			if (dhx_ == nullptr)
				dhx_  = new Blob<float>(batchSize_, input->channel, numLayer_, hiddenSize_);
			else
				dhx_->Reset(batchSize_, input->channel, numLayer_, hiddenSize_);

			// initilaize dhy
			if (dhy_ == nullptr)
				dhy_  = new Blob<float>(batchSize_, input->channel, numLayer_, hiddenSize_);
			else
				dhy_->Reset(batchSize_, input->channel, numLayer_, hiddenSize_);

			// initilaize dcx
			if (dcx_ == nullptr)
				dcx_  = new Blob<float>(batchSize_, input->channel, numLayer_, hiddenSize_);
			else
				dcx_->Reset(batchSize_, input->channel, numLayer_, hiddenSize_);

			// initilaize dcy
			if (dcy_ == nullptr)
				dcy_  = new Blob<float>(batchSize_, input->channel, numLayer_, hiddenSize_);
			else
				dcy_->Reset(batchSize_, input->channel, numLayer_, hiddenSize_);

			// initilaize output
			if (output_ == nullptr)
				output_  = new Blob<float>(batchSize_, input->channel, hiddenSize_, seqLen_);
			else
				output_->Reset(batchSize_, input->channel, hiddenSize_, seqLen_);

			// initialize workspace for cudnn
			SetWorkspace();
		}

		// initialize weights and bias
		if(weights_ == nullptr)
		{
			// initialize containers handles
			weights_ = new Blob<float>(1, input->channel, 1, weightSize / sizeof(float));
			biases_  = new Blob<float>(numLayer_, hiddenSize_);	// bias size

			// initialize weights
			if(loadPretrain && freeze_)
			{
				printf("loadPretrain Init\n");
				if(LoadParameter())
				{
					std::cout << "error occurred.." << std::endl;
					exit(-1);
				}
			}
			else if (!freeze_)
			{
				printf("InitWeightBias Init\n");
				InitWeightBias();
			}
			else
			{
				/* do nothing */
			}
		}

		CheckCudnnErrors(cudnnRNNForward(cuda->Cudnn(),
										 rnnDesc,
										 CUDNN_FWD_MODE_TRAINING,
										 devSeqLengthArray,
										 xDesc,
										 input_->Cuda(),
										 yDesc,
										 output_->Cuda(),
										 hDesc,
										 hx_->Cuda(),
										 hy_->Cuda(),
										 cDesc,
										 cx_->Cuda(),
										 cy_->Cuda(),
										 weightSize,
										 weights_->Cuda(),
										 workspaceSize,
										 dWorkspace,
										 reserveSize,
										 reserveSpace));

	#if (DEBUG_RNN & 0x01)
		input_->Print(layerName + "::input", true, input->num, 28);
		weights_->Print(layerName + "::weight", true);
		output_->Print(layerName + "::output", true);
	#endif

		return output_;
	}

	Blob<float> *RNN::Backward(Blob<float> *gradOutput)
	{
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

		CheckCudnnErrors(cudnnRNNBackwardData_v8(cuda->Cudnn(),
												 rnnDesc,
												 devSeqLengthArray,
												 yDesc,
												 output_->Cuda(),
												 gradOutput_->Cuda(),
												 xDesc,
												 gradInput_->Cuda(),
												 hDesc,
												 hx_->Cuda(),
												 dhy_->Cuda(),
												 dhx_->Cuda(),
												 cDesc,
												 cx_->Cuda(),
												 dcy_->Cuda(),
												 dcx_->Cuda(),
												 weightSize,
												 weights_->Cuda(),
												 workspaceSize,
												 dWorkspace,
												 reserveSize,
												 reserveSpace));

		CheckCudnnErrors(cudnnRNNBackwardWeights_v8(cuda->Cudnn(),
													rnnDesc,
													CUDNN_WGRAD_MODE_ADD,
													devSeqLengthArray,
													xDesc,
													input_->Cuda(),
													hDesc,
													hx_->Cuda(),
													yDesc,
													output_->Cuda(),
													weightSize,
													gradWeights_->Cuda(),
													workspaceSize,
													dWorkspace,
													reserveSize,
													reserveSpace));

	#if (DEBUG_RNN & 0x02)
		std::cout << layerName << "[BACKWARD]" << std::endl;
		gradOutput_->Print(layerName + "::gradients", true);
		gradWeights_->Print(layerName + "gfilter", true);
		if (!gradientStop)
			gradInput_->Print(layerName +"gdata", true);
	#endif

		return gradInput_;
	}
}