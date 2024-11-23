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

        if(output       != nullptr)  delete output;
	    if(gradInput    != nullptr)  delete gradInput;

	    if(weights      != nullptr)  delete weights;
	    if(biases       != nullptr)  delete biases;
	    if(gradWeights  != nullptr)  delete gradWeights;
	    if(gradBiases   != nullptr)  delete gradBiases;
    }

    void Layer::InitWeightBias(unsigned int seed)
    {
	    CheckCudaErrors(cudaDeviceSynchronize());

        if(weights == nullptr || biases == nullptr) return;

	    // Create random network
	    std::random_device rd;
	    std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

	    // He uniform distribution
	    float range = sqrt(6.f / input->Size());	// He's initialization
	    std::uniform_real_distribution<> dis(-range, range);

	    for(int i = 0; i < weights->Length(); i++)
		    weights->Ptr()[i] = static_cast<float>(dis(gen));
	    for(int i = 0; i < biases->Length(); i++)
		    biases->Ptr()[i] = 0.f;

	    // copy initialized value to the device
	    weights->To(DEV_TYPE::CUDA);
	    biases->To(DEV_TYPE::CUDA);

	    std::cout << ".. initialized " << layerName << " layer .." << std::endl;
    }

    void Layer::UpdateWeightsBiases(float learningRate)
    {
	    float eps = -1.f * learningRate;

	    if(weights != nullptr && gradWeights != nullptr)
	    {

        #if(DEBUG_UPDATE)
		    weights->print(layerName + "::weights (before update)", true);
		    gradWeights->print(layerName + "::gweights", true);
        #endif // DEBUG_UPDATE

		// w = w + eps * dw
		    CheckCublasErrors(cublasSaxpy(cuda->Cublas(), weights->Length(), &eps, gradWeights->Cuda(), 1, weights->Cuda(), 1));

        #if(DEBUG_UPDATE)
		    weights->print(layerName + "weights (after update)", true);
		    // getchar();
        #endif // DEBUG_UPDATE
	    }

	    if(biases != nullptr && gradBiases != nullptr)
	    {

        #if(DEBUG_UPDATE)
		    biases->print(layerName + "biases (before update)", true);
		    gradBiases->print(layerName + "gbiases", true);
        #endif // DEBUG_UPDATE

		// b = b + eps * db
		CheckCublasErrors(cublasSaxpy(cuda->Cublas(), biases->Length(), &eps, gradBiases->Cuda(), 1, biases->Cuda(), 1));

        #if (DEBUG_UPDATE)
		    biases->print(layerName + "biases (after update)", true);
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
	    if(weights->FileRead(filenameWeights.str())) return -1;

	    filenameBiases << layerName << ".bias.bin";
	    if(biases->FileRead(filenameBiases.str())) return -2;

	    std::cout << ".. loaded " << layerName << " pretrain parameter.." << std::endl;

	    return 0;
    }

    int Layer::SaveParameter()
    {
	    std::stringstream filenameWeights, filenameBiases;

	    std::cout << ".. saving " << layerName << " parameter ..";
	
	    // Write weights file
	    if(weights)
	    {
		    filenameWeights << layerName << ".bin";
		    if(weights->FileWrite(filenameWeights.str())) return -1;
	    }
	
	    // Write bias file
	    if(biases)
	    {
		    filenameBiases << layerName << ".bias.bin";
		    if(biases->FileWrite(filenameBiases.str())) return -2;
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
		freeze = true;
	}
    
	void Layer::UnFreeze()
	{
		freeze = false;
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
		if(weights == nullptr)
		{
			// setup parameter size information
			inputSize  = input->channel * input->height * input->width;
		
			// initialize weight, bias, and output
			weights = new Blob<float>(1, 1, inputSize, outputSize);
			biases  = new Blob<float>(1, 1, outputSize);
		}

		// initilaize input and output
		if(input == nullptr || batchSize != input->num)
		{
			input = input;
			batchSize  = input->num;

			if(output == nullptr)
				output  = new Blob<float>(batchSize, outputSize);
			else
				output->Reset(batchSize, outputSize);
		
			output->Tensor();

			if(dOneVec != nullptr) cudaFree(dOneVec);

			CheckCudaErrors(cudaMalloc((void**)&dOneVec, sizeof(float) * batchSize));
			InitOneVec<<<(batchSize + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D, BLOCK_DIM_1D >>>(dOneVec, batchSize);

			// initialize weights and biases
			if(loadPretrain && !freeze)
			{
				if(LoadParameter())
				{
					std::cout << "error occurred.." << std::endl;
					exit(-1);
				}
			}
			else if(!freeze)
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
				outputSize, batchSize, inputSize,
				&cuda->one,  
				weights->Cuda(), inputSize, 
				input->Cuda(), inputSize,
				&cuda->zero, 
				output->Cuda(),  outputSize));

		// output += biases * dOneVec ^ T
		CheckCublasErrors(
			cublasSgemm(cuda->Cublas(),
				CUBLAS_OP_N, CUBLAS_OP_N, 
				outputSize, batchSize, 1,
				&cuda->one, 
				biases->Cuda(), outputSize, 
				dOneVec, 1, 
				&cuda->one, 
				output->Cuda(), outputSize));

	#if (DEBUG_DENSE & 0x01)
		input->Print(layerName + "::input",  true);
		weights->Print(layerName + "::weight", true);
		biases->Print(layerName + "::bias",   true);
		output->Print(layerName + "::output", true);
	#endif // DEBUG_DENSE

		return output;
	}

	Blob<float> *Dense::Backward(Blob<float> *gradOutput)
	{
		if(gradWeights == nullptr)
		{
			gradWeights = new Blob<float>(weights->Shape());
			gradBiases  = new Blob<float>(biases->Shape());
		}

		if(gradInput == nullptr || batchSize != gradOutput->num)
		{
			gradOutput  = gradOutput;

			if (gradInput == nullptr)
				gradInput   = new Blob<float>(input->Shape());
			else
				gradInput->Reset(input->Shape());
		}

		// db = (dy) * d_one_vec
		cublasSgemv(cuda->Cublas(),
				CUBLAS_OP_N,
				outputSize, batchSize,
				&cuda->one,
				gradOutput->Cuda(), outputSize,
				dOneVec, 1,
				&cuda->zero,
				gradBiases->Cuda(), 1);

		// dw = x * (dy)^T
		cublasSgemm(cuda->Cublas(),
			CUBLAS_OP_N, CUBLAS_OP_T,
			inputSize, outputSize, batchSize,
			&cuda->one,
			input->Cuda(),        inputSize,
			gradOutput->Cuda(),   outputSize,
			&cuda->zero,
			gradWeights->Cuda(),  inputSize);

		// dx = W * dy
		if (!gradientStop)
			cublasSgemm(cuda->Cublas(),
				CUBLAS_OP_N, CUBLAS_OP_N,
				inputSize, batchSize, outputSize,
				&cuda->one,
				weights->Cuda(),    inputSize,
				gradOutput->Cuda(), outputSize,
				&cuda->zero, 
				gradInput->Cuda(),  inputSize);

	#if (DEBUG_DENSE & 0x02)
		std::cout << layerName << "[BACKWARD]" << std::endl;
		gradOutput->Print(layerName + "::gradients", true, gradOutput->num);
		gradWeights->Print(layerName + "::gfilter", true);
		gradBiases->Print(layerName + "::gbias", true);
		if(!gradientStop) gradInput->Print(layerName + "::gdata", true);
	#endif // DEBUG_DENSE

		return gradInput;
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
		if(input == nullptr || batchSize != input->num)
		{
			input = input;
			inputDesc = input->Tensor();
			batchSize = input->num;

			if(output == nullptr)
				output = new Blob<float>(input->Shape());
			else
				output->Reset(input->Shape());

			outputDesc = output->Tensor();
		}

		cudnnActivationForward(cuda->Cudnn(),
			actDesc,
			&cuda->one,
			inputDesc,
			input->Cuda(),
			&cuda->zero,
			outputDesc,
			output->Cuda());

		return output;
	}

	Blob<float> *Activation::Backward(Blob<float> *gradOutput)
	{
		if (gradInput == nullptr || batchSize != gradOutput->num)
		{
			gradOutput = gradOutput;

			if (gradInput == nullptr)
				gradInput = new Blob<float>(input->Shape());
			else
				gradInput->Reset(input->Shape());		
		}

		cudnnActivationBackward(cuda->Cudnn(),
			actDesc,
			&cuda->one, 
			outputDesc, output->Cuda(),
			outputDesc, gradOutput->Cuda(), 
			inputDesc, input->Cuda(), 
			&cuda->zero, 
			inputDesc, gradInput->Cuda());

		return gradInput;
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
		if(input == nullptr || batchSize != input->num)
		{
			input = input;
			inputDesc = input->Tensor();
			batchSize = input->num;
		
			if(output == nullptr)
				output = new Blob<float>(input->Shape());
			else
				output->Reset(input->Shape());		

			outputDesc = output->Tensor();
		}

	#if (DEBUG_SOFTMAX & 0x01)
		std::cout << layerName << "[FORWARD]" << std::endl;
		input->Print(layerName + "::input", true, input->num);
	#endif

		CheckCudnnErrors(
			cudnnSoftmaxForward(cuda->Cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
				&cuda->one,  inputDesc,  input->Cuda(),
				&cuda->zero, outputDesc, output->Cuda()));

	#if (DEBUG_SOFTMAX & 0x01)
		output->Print(layerName + "::output", true, input->num);
	#endif

		return output;
	}

	Blob<float> *Softmax::Backward(Blob<float> *target)
	{
		CheckCudaErrors(cudaDeviceSynchronize());

		if(gradInput == nullptr || batchSize != target->num)
		{
			if (gradInput == nullptr)
				gradInput = new Blob<float>(input->Shape());
			else
		 		gradInput->Reset(input->Shape());
		}

		// set grad_input_ as predict
		CheckCudaErrors(cudaMemcpyAsync(gradInput->Cuda(), 
			output->Cuda(), output->BufSize(), 
			cudaMemcpyDeviceToDevice));

		// set gradInput = predict - target	
		CheckCublasErrors(
			cublasSaxpy(cuda->Cublas(), target->Length(),
				&cuda->minusOne, target->Cuda(), 1,
				gradInput->Cuda(), 1));

		// normalize the grad_output by the batch size
		int gradOutputSize = target->num * target->channel * target->height * target->width;
		float scale = 1.f / static_cast<float>(target->num);
		CheckCublasErrors(cublasSscal(cuda->Cublas(), gradOutputSize, &scale, gradInput->Cuda(), 1));

	#if (DEBUG_SOFTMAX & 0x02)
		std::cout << layerName << "[BACKWARD]" << std::endl;
		input->Print( layerName + "::input", true);
		output->Print(layerName + "::predict", true);
		target->Print( layerName + "::y", true, target->num);
		gradInput->Print(layerName + "::dx", true, target->num);
	#endif

		return gradInput;
	}

	float Softmax::getLoss(Blob<float> *target)
	{
		return loss.Loss(output, target);
	}

	int Softmax::getAccuracy(Blob<float> *target)
	{
		int batchSize = output->num;
		int outputSize = output->Size();

		assert(batchSize == target->num);
		assert(outputSize == target->Size());

		float *hOutput, *hTarget;
		int idxOutput, idxTarget;
		int hitCount = 0;

		// get predicts and targets
		hOutput = output->To(HOST);
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
		if(weights == nullptr)
		{
			// initialize containers handles
			CheckCudnnErrors(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outChannels, input->channel, kernelSize, kernelSize));
			weights = new Blob<float>(outChannels, input->channel, kernelSize, kernelSize);
			biases  = new Blob<float>(1, outChannels);	// bias size
			biasDesc = biases->Tensor();
		}
 
		// initilaize input and output
		if(input == nullptr || batchSize != input->num)
		{
			// initialize input
			input = input;
			inputDesc = input->Tensor();
			batchSize  = input->num;

			printf("batchSize = %d, Size 0 = %d, Size 1 = %d, Size 2 = %d, Size 3 = %d\n", batchSize, outputSize[0], outputSize[1], outputSize[2], outputSize[3]);

			// initilaize output
			CheckCudnnErrors(cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &outputSize[0], &outputSize[1], &outputSize[2], &outputSize[3]));

			if (output == nullptr)
				output  = new Blob<float>(outputSize);
			else
				output->Reset(outputSize);

			outputDesc = output->Tensor();

			// initialize workspace for cudnn
			SetWorkspace();

			// initialize weights
			if(loadPretrain && !freeze)
			{
				if(LoadParameter())
				{
					std::cout << "error occurred.." << std::endl;
					exit(-1);
				}
			}
			else if (!freeze)
			{
				InitWeightBias();
			}
			else
			{
				/* do nothing */
			}
		}

		CheckCudnnErrors(cudnnConvolutionForward(cuda->Cudnn(),
			&cuda->one,  inputDesc,  input->Cuda(),
			filterDesc, weights->Cuda(), convDesc, convFwdAlgo, dWorkspace,  workspaceSize,
			&cuda->zero, outputDesc, output->Cuda()));

		CheckCudnnErrors(cudnnAddTensor(cuda->Cudnn(), 
			&cuda->one, biasDesc, biases->Cuda(), 
			&cuda->one, outputDesc, output->Cuda()));

	#if (DEBUG_CONV & 0x01)
		input->Print(layerName + "::input", true, input->num, 28);
		weights->Print(layerName + "::weight", true);
		biases->Print(layerName + "::bias", true);
		output->Print(layerName + "::output", true);
	#endif

		return output;
	}

	Blob<float> *Conv2D::Backward(Blob<float> *gradOutput)
	{
		// initialize gradOutput back-propagation space
		if(gradInput == nullptr || batchSize != gradOutput->num)
		{
			gradOutput  = gradOutput;
			gradWeights = new Blob<float>(weights->Shape());
			gradBiases  = new Blob<float>(1, biases->channel);

			if(gradInput == nullptr)
				gradInput = new Blob<float>(input->Shape());
			else
				gradInput->Reset(input->Shape());
		}

		// gradients of biases
		CheckCudnnErrors(cudnnConvolutionBackwardBias(cuda->Cudnn(), &cuda->one, outputDesc, gradOutput->Cuda(), &cuda->zero, biasDesc, gradBiases->Cuda()));
	
		// gradients of weights 
		CheckCudnnErrors(
			cudnnConvolutionBackwardFilter(cuda->Cudnn(),
				&cuda->one, 
				inputDesc, input->Cuda(), 
				outputDesc, gradOutput->Cuda(),
				convDesc, convBwdFilterAlgo, dWorkspace, workspaceSize,
				&cuda->zero, 
				filterDesc, gradWeights->Cuda()));

		// gradients of input data
		if (!gradientStop)
			CheckCudnnErrors(
				cudnnConvolutionBackwardData(cuda->Cudnn(),
					&cuda->one, 
					filterDesc, weights->Cuda(), 
					outputDesc, gradOutput->Cuda(), 
					convDesc, convBwdDataAlgo, dWorkspace, workspaceSize,
					&cuda->zero, 
					inputDesc, gradInput->Cuda()));

	#if (DEBUG_CONV & 0x02)
		std::cout << layerName << "[BACKWARD]" << std::endl;
		gradOutput->Print(layerName + "::gradients", true);
		gradBiases->Print(layerName + "gbias", true);
		gradWeights->Print(layerName + "gfilter", true);
		if (!gradientStop)
			gradInput->Print(layerName +"gdata", true);
	#endif

	#if (DEBUG_CONV & 0x04)
		gradOutput->Print(layerName + "::gradients", true);
		gradBiases->Print(layerName + "::gbias", true);
	#endif

		return gradInput;
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
		if(input == nullptr || batchSize != input->num)
		{
			input = input;

			// resource initialize
			inputDesc = input->Tensor();
			batchSize = input->num;
		
			// setting output
			cudnnGetPooling2dForwardOutputDim(poolDesc, inputDesc, &outputSize[0], &outputSize[1], &outputSize[2], &outputSize[3]);
			if(output == nullptr)
				output = new Blob<float>(outputSize);
			else
				output->Reset(outputSize);
		
			outputDesc = output->Tensor();
		}

		cudnnPoolingForward(cuda->Cudnn(), poolDesc,
			&cuda->one,   inputDesc,  input->Cuda(),
			&cuda->zero,  outputDesc, output->Cuda());

		return output;
	}

	Blob<float> *Pooling::Backward(Blob<float> *gradOutput)
	{
		if (gradInput == nullptr || batchSize != gradOutput->num)
		{
			gradOutput = gradOutput;

			if (gradInput == nullptr)
				gradInput = new Blob<float>(input->Shape());
			else
				gradInput->Reset(input->Shape());
		}

		CheckCudnnErrors(
			cudnnPoolingBackward(cuda->Cudnn(), poolDesc,
				&cuda->one,  
				outputDesc, output->Cuda(),
				outputDesc, gradOutput->Cuda(), 
				inputDesc,  input->Cuda(), 
				&cuda->zero, 
				inputDesc,  gradInput->Cuda()));

		return gradInput;
	}
}