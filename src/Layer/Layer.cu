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

__global__ void PadForward(const int count, const float *in, float *out,
                           const int num, const int channel, const int heightIn, const int widthIn,
                           const int pad)
{
	CUDA_1D_KERNEL_LOOP(index, count)
	{
		int i = index;  // Preserve the original value
		int heightOut = heightIn + pad + pad;
		int widthOut = widthIn + pad + pad;
		int w = i % widthIn;
		i /= widthIn;
		int h = i % heightIn;
		i /= heightIn;
		int c = i % channel;
		i /= channel;
			
		out[((i * channel + c) * heightOut + h + pad) * widthOut + pad + w] = in[index];
		int w1 = index % widthOut;
		int h1 = (index / widthOut) % heightOut;
		if (h1 < pad || h1 > heightOut - 1 - pad || w1 < pad || w1 > widthOut - 1 - pad)
		{
			out[index] = 0.f;
		}
	}
}

__global__ void PadBackward(const int count, const float *in, float *out,
                            const int num, const int channel, const int heightIn, const int widthIn,
                            const int pad) 
{
    CUDA_1D_KERNEL_LOOP(index, count)
	{
        int i = index;  // Preserve original value
        int heightOut = heightIn + pad + pad;
        int widthOut = widthIn + pad + pad;
        int w = i % widthIn;
        i /= widthIn;
        int h = i % heightIn;
        i /= heightIn;
        int c = i % channel;
        i /= channel;
        out[index] = in[((i * channel + c) * heightOut + h + pad) * widthOut + pad + w];
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

__global__ void AddUpdate(float *arrayA, float *arrayB, float *arrayC, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += step)
	{
		arrayC[i] = arrayA[i] + arrayB[i];
    }
}

__global__ void MomentumUpdate(int N, float *m, float *w, const float *g, const float lr, const float momentum)
{
    CUDA_1D_KERNEL_LOOP(i, N)
	{
		m[i] = momentum * m[i] + lr * g[i];
		w[i] = w[i] - m[i];
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
		if(weightsM_     != nullptr)  delete weightsM_;
	    if(biasesM_      != nullptr)  delete biasesM_;
		if(weightsV_     != nullptr)  delete weightsV_;
	    if(biasesV_      != nullptr)  delete biasesV_;
	    if(gradWeights_  != nullptr)  delete gradWeights_;
	    if(gradBiases_   != nullptr)  delete gradBiases_;
    }

    void Layer::InitWeightBias(unsigned int seed)
    {
	    CheckCudaErrors(cudaDeviceSynchronize());

        if(weights_ == nullptr || biases_ == nullptr) return;

		printf("He uniform distribution 1\n");

	    // Create random network
	    std::random_device rd;
	    std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

		printf("He uniform distribution 2\n");

	    // He uniform distribution
	    float range = sqrt(6.f / input_->Size());	// He's initialization
	    std::uniform_real_distribution<> dis(-range, range);

		printf("He uniform distribution 3\n");

	    for(int i = 0; i < weights_->Length(); i++)
		    weights_->Ptr()[i] = static_cast<float>(dis(gen));
	    for(int i = 0; i < biases_->Length(); i++)
		    biases_->Ptr()[i] = 0.f;
		for(int i = 0; i < weightsM_->Length(); i++)
		    weightsM_->Ptr()[i] = 0.f;
	    for(int i = 0; i < biasesM_->Length(); i++)
		    biasesM_->Ptr()[i] = 0.f;
		for(int i = 0; i < weightsV_->Length(); i++)
		    weightsV_->Ptr()[i] = 0.f;
	    for(int i = 0; i < biasesV_->Length(); i++)
		    biasesV_->Ptr()[i] = 0.f;

		printf("He uniform distribution\n");

	    // copy initialized value to the device
		weights_->To(DEV_TYPE::CUDA);
		biases_->To(DEV_TYPE::CUDA);
		weightsM_->To(DEV_TYPE::CUDA);
		biasesM_->To(DEV_TYPE::CUDA);
		weightsV_->To(DEV_TYPE::CUDA);
		biasesV_->To(DEV_TYPE::CUDA);

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
			//config = GetGpuLaunchConfig(weights_->Length(), AdamUpdate, 0, 0);
			AdamUpdate<<<16, BLOCK_DIM_1D>>>(weights_->Length(),
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
			//config = GetGpuLaunchConfig(biases_->Length(), AdamUpdate, 0, 0);
			AdamUpdate<<<16, BLOCK_DIM_1D>>>(biases_->Length(),
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
			//config = GetGpuLaunchConfig(weights_->Length(), RmspropUpdate, 0, 0);
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
			//config = GetGpuLaunchConfig(biases_->Length(), RmspropUpdate, 0, 0);
        	RmspropUpdate<<< BLOCK_DIM, BLOCK_DIM_1D >>>(biases_->Length(),
														 biasesM_->Cuda(),
														 biases_->Cuda(),
														 gradBiases_->Cuda(),
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

	void Layer::SetOutputTo(Layer *layer)
	{
		if (outputTo_ == nullptr)
			outputTo_ = layer;
		else
			copyOutputTo_ = layer;
	}
	
	void Layer::SetLayerRelationship(Layer *input1From, Layer *input2From)
	{
		input1From_ = input1From;
		input2From_ = input2From;
		if (input1From_ != nullptr) input1From_->SetOutputTo(this);
		if (input2From_ != nullptr) input2From_->SetOutputTo(this);
	}
	
	Blob<float> *Layer::GetInput(Blob<float> *input)
	{
		if (input1From_ != nullptr) input   = input1From_->GetOutput();
		if (input2From_ != nullptr) input2_ = input2From_->GetOutput();
		
		return input;
	}
	
	Blob<float> *Layer::SumGradients(Blob<float> *grad)
	{
		Blob<float> *grad2;
		if (outputTo_ != nullptr) grad = outputTo_->GetGrad();
		if (copyOutputTo_ != nullptr)
		{
			grad2 = copyOutputTo_->GetGrad();
			// grad = grad + grad2
			//config = GetGpuLaunchConfig(grad2->Length(), AddUpdate, 0, 0);
			AddUpdate<<<BLOCK_DIM, BLOCK_DIM_1D>>>(grad->Cuda(), grad2->Cuda(), grad->Cuda(), grad->Length());
		}

		return grad;
	}

	std::string Layer::GetName()
	{
		return layerName;
	}

	void Layer::SetCudaContext(CudaContext *context)
	{
		cuda = context;
	}

	Blob<float> *Layer::GetOutput()
	{
		return output_;
	}
	
	Blob<float> *Layer::GetGrad()
	{
		return gradInput_;
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