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
    Layer::Layer()
    {

    }

    Layer::~Layer()
    {

    #if (DEBUG_FORWARD > 0 || DEBUG_BACKWARD > 0)
	    std::cout << "Destroy Layer: " << layerName << std::endl;
    #endif

        if(output_       != nullptr)  delete output_;
	    if(gradInput_    != nullptr)  delete gradInput_;

	    if(weights_      != nullptr)  delete weights_;
	    if(biases_       != nullptr)  delete biases_;
	    if(gradWeights_  != nullptr)  delete gradWeights_;
	    if(gradBiases_   != nullptr)  delete gradBiases_;
    }

    void Layer::InitWeightBias(unsigned int seed)
    {
	    CheckCudaErrors(cudaDeviceSynchronize());

        if(weights_ == nullptr || biases_ == nullptr) return;

	    // Create random network
	    std::random_device rd;
	    std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

	    // He uniform distribution
	    float range = sqrt(6.f / input_->Size());	// He's initialization
	    std::uniform_real_distribution<> dis(-range, range);

	    for(int i = 0; i < weights_->Length(); i++)
		    weights_->Ptr()[i] = static_cast<float>(dis(gen));
	    for(int i = 0; i < biases_->Length(); i++)
		    biases_->Ptr()[i] = 0.f;

		printf("He uniform distribution\n");

	    // copy initialized value to the device
	    weights_->To(DEV_TYPE::CUDA);
	    biases_->To(DEV_TYPE::CUDA);

	    std::cout << ".. initialized " << layerName << " layer .." << std::endl;
    }

    void Layer::UpdateWeightsBiases(float learningRate)
    {
	    float eps = -1.f * learningRate;

	    if(weights_ != nullptr && gradWeights_ != nullptr)
	    {

        #if(DEBUG_UPDATE)
		    weights_->print(layerName + "::weights (before update)", true);
		    gradWeights_->print(layerName + "::gweights", true);
        #endif // DEBUG_UPDATE

		// w = w + eps * dw
		    CheckCublasErrors(cublasSaxpy(cuda->Cublas(), weights_->Length(), &eps, gradWeights_->Cuda(), 1, weights_->Cuda(), 1));

        #if(DEBUG_UPDATE)
		    weights->print(layerName + "weights (after update)", true);
		    // getchar();
        #endif // DEBUG_UPDATE
	    }

	    if(biases_ != nullptr && gradBiases_ != nullptr)
	    {

        #if(DEBUG_UPDATE)
		    biases_->print(layerName + "biases (before update)", true);
		    gradBiases_->print(layerName + "gbiases", true);
        #endif // DEBUG_UPDATE

		// b = b + eps * db
		CheckCublasErrors(cublasSaxpy(cuda->Cublas(), biases_->Length(), &eps, gradBiases_->Cuda(), 1, biases_->Cuda(), 1));

        #if (DEBUG_UPDATE)
		    biases_->print(layerName + "biases (after update)", true);
		    // getchar();
        #endif // DEBUG_UPDATE
	    }
    }

    float Layer::GetLoss(Blob<float> *target)
    {
	    assert("No Loss layer has no loss." && false);
	    return EXIT_FAILURE;
    }

    int Layer::GetAccuracy(Blob<float> *target)
    {
	    assert("No Loss layer cannot estimate accuracy." && false);
	    return EXIT_FAILURE;
    }

    int Layer::LoadParameter()
    {
	    std::stringstream filenameWeights, filenameBiases;

	    // load weights and biases pretrained parameters
	    filenameWeights << layerName << ".bin";
	    if(weights_->FileRead(filenameWeights.str())) return -1;

	    filenameBiases << layerName << ".bias.bin";
	    if(biases_->FileRead(filenameBiases.str())) return -2;

	    std::cout << ".. loaded " << layerName << " pretrain parameter.." << std::endl;

	    return 0;
    }

    int Layer::SaveParameter()
    {
	    std::stringstream filenameWeights, filenameBiases;

	    std::cout << ".. saving " << layerName << " parameter ..";
	
	    // Write weights file
	    if(weights_)
	    {
		    filenameWeights << layerName << ".bin";
		    if(weights_->FileWrite(filenameWeights.str())) return -1;
	    }
	
	    // Write bias file
	    if(biases_)
	    {
		    filenameBiases << layerName << ".bias.bin";
		    if(biases_->FileWrite(filenameBiases.str())) return -2;
	    }

	    std::cout << " done .." << std::endl;

	    return 0;
    }

	std::string Layer::GetName()
	{
		return layerName;
	}

	void Layer::SetCudaContext(CudaContext *context)
	{
		cuda = context;
	}

	void Layer::SetLoadPretrain()
	{ 
		loadPretrain = true;
	}

    void Layer::SetGradientStop()
	{
		gradientStop = true;
	}

    void Layer::Freeze()
	{
		freeze_ = true;
	}
    
	void Layer::UnFreeze()
	{
		freeze_ = false;
	}

    /****************************************************************
    * Dense Layer                                                  *
    ****************************************************************/

	Dense::Dense(std::string name, int outSize)
	{
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
		// initialize weights and biases
		if(weights_ == nullptr)
		{
			// setup parameter size information
			inputSize  = input->channel * input->height * input->width;
		
			// initialize weight, bias, and output
			weights_ = new Blob<float>(1, 1, inputSize, outputSize);
			biases_  = new Blob<float>(1, 1, outputSize);
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
			InitOneVec<<<(batchSize_ + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D, BLOCK_DIM_1D >>>(dOneVec, batchSize_);

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

	/****************************************************************
 	* Activation Layer                                             *
 	****************************************************************/

	Activation::Activation(std::string name, cudnnActivationMode_t mode, float coef)
	{
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

	/****************************************************************
 	 * Layer definition                                             *
 	****************************************************************/

	/**
 	* Convolutional layer with bias
 	*/
	Conv2D::Conv2D(std::string name, int outChannels, int kernelSize, int stride, int padding, int dilation) :  outChannels(outChannels),
																												kernelSize(kernelSize),
																												stride(stride),
																												padding(padding),
																												dilation(dilation)
	{
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
		// initialize weights and bias
		if(weights_ == nullptr)
		{
			// initialize containers handles
			CheckCudnnErrors(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outChannels, input->channel, kernelSize, kernelSize));
			weights_ = new Blob<float>(outChannels, input->channel, kernelSize, kernelSize);
			biases_  = new Blob<float>(1, outChannels);	// bias size
			biasDesc = biases_->Tensor();
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

	/****************************************************************
 	 * Layer definition                                             *
 	****************************************************************/
	Pooling::Pooling(std::string name, int kernelSize, int padding, int stride, cudnnPoolingMode_t mode) :  poolKernelSize(kernelSize),
																											poolPadding(padding),
																											poolStride(stride),
																											poolMode(mode)
	{
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

	/****************************************************************
 	 * Layer definition                                             *
 	****************************************************************/

	/**
 	* RNN layer with bias
 	*/
	RNN::RNN(std::string name, const int hiddenSize, const int numLayer, double dropout, cudnnDirectionMode_t bidirectional, cudnnRNNMode_t mode)
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
		if(input == nullptr || batchSize_ != input->num || hInput_ == nullptr)
		{
			// initialize input
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
		hInput_->Print(layerName + "::input", true, input->num, hiddenSize_);
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