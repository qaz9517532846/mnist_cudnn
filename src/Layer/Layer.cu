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
}