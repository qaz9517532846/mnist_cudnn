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

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void Add(float *arrayA, float *arrayB, float *arrayC, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += step)
	{
		arrayC[i] = arrayA[i] + arrayB[i];
    }
}

__global__ void AdamUpdate(int N, int step, float *m, float *v, float *w, const float *g, const float beta1, const float beta2, const float eps_hat, const float lr)
{
    CUDA_1D_KERNEL_LOOP(i, N)
	{
        // Updating running mean and var.
        m[i] = m[i] * beta1 + g[i] * (1 - beta1);
        v[i] = v[i] * beta2 + g[i] * g[i] * (1 - beta2);
        float mi = m[i] / (1 - std::pow(beta1, step + 1));
        float vi = v[i] / (1 - std::pow(beta2, step + 1));
        // Update parameters.
        w[i] = w[i] - lr * mi / (std::sqrt(vi) + eps_hat);
    }
}

__global__ void RmspropUpdate(int N, float *m, float *w, const float *g, const float decay, const float eps_hat, const float lr)
{
    CUDA_1D_KERNEL_LOOP(i, N)
	{
        // Updating running mean and var.
        m[i] = m[i] + (1 - decay) * (g[i] * g[i] - m[i]);
        // Update parameters.
        w[i] = w[i] - lr * g[i] / std::sqrt(eps_hat + m[i]);
    }
}

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
	
	void Layer::UpdateWeightsBiasesWithAdam(float learningRate, float beta1, float beta2, float epsHat, int step)
	{
		if (weights_ != nullptr && gradWeights_ != nullptr)
		{
		#if (DEBUG_UPDATE)
			weights_->print(name_ + "::weights (before update)", true);
			grad_weights_->print(name_ + "::gweights", true);
		#endif // DEBUG_UPDATE
        	/*
			 * mt = b1*mt + (1-b1)*dw
        	 * vt = b2*vt + (1-b2)*dw*dw
        	 * mtt = mt/(1-(b1^(i+1)))
        	 * vtt = vt/(1-(b2^(i+1)))
        	 * w = w + eps * mtt/(std::sqrt(vtt) + e)
        	 */
			 
			AdamUpdate<<< 16, BLOCK_DIM_1D >>>(weights_->Length(),
												step,
												weightsM_->Cuda(),
												weightsV_->Cuda(),
												weights_->Cuda(),
												gradWeights_->Cuda(),
												beta1,
												beta2,
												epsHat,
												learningRate);
		#if (DEBUG_UPDATE)
			weights_->print(name_ + "weights (after update)", true);
			// getchar();
		#endif // DEBUG_UPDATE
		}
		
		if (biases_ != nullptr && gradBiases_ != nullptr)
		{
		#if (DEBUG_UPDATE)
			biases_->print(name_ + "biases (before update)", true);
			gradBiases_->print(name_ + "gbiases", true);
		#endif // DEBUG_UPDATE
		
			AdamUpdate<<< 16, BLOCK_DIM_1D >>>(biases_->Length(),
												step,
												biasesM_->Cuda(),
												biasesV_->Cuda(),
												biases_->Cuda(),
												gradBiases_->Cuda(),
												beta1,
												beta2,
												epsHat,
												learningRate);
												
		#if (DEBUG_UPDATE)
			biases_->print(name_ + "biases (after update)", true);
			// getchar();
		#endif // DEBUG_UPDATE
		}
	}

	void Layer::UpdateWeightsBiasesWithRmsprop(float learningRate, float decay, float epsHat)
	{
		if (weights_ != nullptr && gradWeights_ != nullptr)
		{
		#if (DEBUG_UPDATE)
        	weights_->print(name_ + "::weights (before update)", true);
        	gradWeights_->print(name_ + "::gweights", true);
		#endif // DEBUG_UPDATE

        	/*
        	 * mt = mt + (1 - decay) * (dw * dw - mt)
        	 * w = w - learning_rate * dw /std::sqrt(eps_hat + mt)
        	 */
        	RmspropUpdate<<< BLOCK_DIM, BLOCK_DIM_1D >>>(weights_->Length(),
														 weightsM_->Cuda(),
														 weights_->Cuda(),
														 gradWeights_->Cuda(),
														 decay,
														 epsHat,
														 learningRate);
		#if (DEBUG_UPDATE)
        	weights_->print(name_ + "weights (after update)", true);
        	// getchar();
		#endif // DEBUG_UPDATE
    	}

		if (biases_ != nullptr && gradBiases_ != nullptr)
		{
		#if (DEBUG_UPDATE)
        	biases_->print(name_ + "biases (before update)", true);
        	gradBiases_->print(name_ + "gbiases", true);
		#endif // DEBUG_UPDATE

        	RmspropUpdate<<< BLOCK_DIM, BLOCK_DIM_1D >>>(weights_->Length(),
														 weightsM_->Cuda(),
														 weights_->Cuda(),
														 gradWeights_->Cuda(),
														 decay,
														 epsHat,
														 learningRate);
		#if (DEBUG_UPDATE)
			biases_->print(name_ + "biases (after update)", true);
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
}