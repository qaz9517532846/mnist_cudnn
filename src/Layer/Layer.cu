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
	    std::cout << "Destroy Layer: " << name << std::endl;
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

	    std::cout << ".. initialized " << name << " layer .." << std::endl;
    }

    void Layer::UpdateWeightsBiases(float learningRate)
    {
	    float eps = -1.f * learningRate;

	    if(weights != nullptr && gradWeights != nullptr)
	    {

        #if(DEBUG_UPDATE)
		    weights->print(name + "::weights (before update)", true);
		    gradWeights->print(name + "::gweights", true);
        #endif // DEBUG_UPDATE

		// w = w + eps * dw
		    CheckCublasErrors(cublasSaxpy(cuda->Cublas(), weights->Length(), &eps, gradWeights->Cuda(), 1, weights->Cuda(), 1));

        #if(DEBUG_UPDATE)
		    weights->print(name + "weights (after update)", true);
		    // getchar();
        #endif // DEBUG_UPDATE
	    }

	    if(biases != nullptr && gradBiases != nullptr)
	    {

        #if(DEBUG_UPDATE)
		    biases->print(name + "biases (before update)", true);
		    gradBiases->print(name + "gbiases", true);
        #endif // DEBUG_UPDATE

		// b = b + eps * db
		CheckCublasErrors(cublasSaxpy(cuda->Cublas(), biases->Length(), &eps, gradBiases->Cuda(), 1, biases->Cuda(), 1));

        #if (DEBUG_UPDATE)
		    biases->print(name + "biases (after update)", true);
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
	    filenameWeights << name << ".bin";
	    if(weights->FileRead(filenameWeights.str())) return -1;

	    filenameBiases << name << ".bias.bin";
	    if(biases->FileRead(filenameBiases.str())) return -2;

	    std::cout << ".. loaded " << name << " pretrain parameter.." << std::endl;

	    return 0;
    }

    int Layer::SaveParameter()
    {
	    std::stringstream filenameWeights, filenameBiases;

	    std::cout << ".. saving " << name << " parameter ..";
	
	    // Write weights file
	    if(weights)
	    {
		    filenameWeights << name << ".bin";
		    if(weights->FileWrite(filenameWeights.str())) return -1;
	    }
	
	    // Write bias file
	    if(biases)
	    {
		    filenameBiases << name << ".bias.bin";
		    if(biases->FileWrite(filenameBiases.str())) return -2;
	    }

	    std::cout << " done .." << std::endl;

	    return 0;
    }

    /****************************************************************
    * Dense Layer                                                  *
    ****************************************************************/

	Dense::Dense(std::string name, int outSize)
	{
		name = name;
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
		input->Print(name + "::input",  true);
		weights->Print(name + "::weight", true);
		biases->Print(name + "::bias",   true);
		output->Print(name + "::output", true);
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
		std::cout << name << "[BACKWARD]" << std::endl;
		gradOutput->Print(name + "::gradients", true, gradOutput->num);
		gradWeights->Print(name + "::gfilter", true);
		gradBiases->Print(name + "::gbias", true);
		if(!gradientStop) gradInput->Print(name + "::gdata", true);
	#endif // DEBUG_DENSE

		return gradInput;
	}

	/****************************************************************
 	* Activation Layer                                             *
 	****************************************************************/


}