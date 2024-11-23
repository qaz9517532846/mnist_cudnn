#include "mnist_cudnn/Network.h"

#include "mnist_cudnn/Helper.h"
#include "mnist_cudnn/Layer.h"

#include <iostream>
#include <iomanip>
#include <nvtx3/nvToolsExt.h>

namespace CUDA_NETWORK
{
    Network::Network()
    {
	    // nothing
    }

    Network::~Network()
    {
	    // destroy network
	    for(auto layer: layersVect)
		    delete layer;

	    // terminate CUDA context
	    if(cuda != nullptr) delete cuda;
    }

    void Network::AddLayer(Layer *layer)
    {
	    layersVect.push_back(layer);

	    // tagging layer to stop gradient if it is the first layer
	    if(layersVect.size() == 1) layersVect.at(0)->SetGradientStop();
    }

    Blob<float> *Network::Forward(Blob<float> *input)
    {
	    output = input;

	    nvtxRangePushA("Forward");
	    for(auto layer : layersVect)
	    {
		#if (DEBUG_FORWARD)
		    std::cout << "[[Forward ]][[ " << std::setw(7) << layer->GetName() << " ]]\t(" << output->num << ", " << output->channel << ", " << output->height << ", " << output->width << ")\t";
		#endif // DEBUG_FORWARD

		    output = layer->Forward(output);

		#if (DEBUG_FORWARD)
		    std::cout << "--> (" << output->num << ", " << output->channel << ", " << output->height << ", " << output->width << ")" << std::endl;
		    checkCudaErrors(cudaDeviceSynchronize());

		#if (DEBUG_FORWARD > 1)
			output->Print("output", true);

			if (phase == INTERFACE) GetChar();
		#endif
		#endif // DEBUG_FORWARD

		    // TEST
		    // CheckCudaErrors(cudaDeviceSynchronize());
	    }
	    nvtxRangePop();

	    return output;
    }

    void Network::Backward(Blob<float> *target)
    {
	    Blob<float> *gradient = target;

	    if (phase == INTERFACE) return;

	    nvtxRangePushA("Backward");
	    // back propagation.. update weights internally.....
	    for (auto layer = layersVect.rbegin(); layer != layersVect.rend(); layer++)
	    {
		    // getting back propagation status with gradient size

        #if (DEBUG_BACKWARD)
		    std::cout << "[[Backward]][[ " << std::setw(7) << (*layer)->GetName() << " ]]\t(" << gradient->num << ", " << gradient->channel << ", " << gradient->height << ", " << gradient->width << ")\t";
        #endif // DEBUG_BACKWARD

		    gradient = (*layer)->Backward(gradient);

        #if (DEBUG_BACKWARD)
		    // and the gradient result
		    std::cout << "--> (" << gradient->num << ", " << gradient->channel << ", " << gradient->height << ", " << gradient->width << ")" << std::endl;
		    checkCudaErrors(cudaDeviceSynchronize());

        #if (DEBUG_BACKWARD > 1)
		    gradient->Print((*layer)->GetName() + "::dx", true);
		    GetChar();
        #endif
        #endif // DEBUG_BACKWARD
	    }
	    nvtxRangePop();
    }

    void Network::Update(float learningRate)
    {
	    if(phase == INTERFACE)
		    return;

    #if (DEBUG_UPDATE)
	    std::cout << "Start update.. lr = " << learningRate << std::endl;
    #endif

	    nvtxRangePushA("Update");
	    for(auto layer : layersVect)
	    {
		    // if no parameters, then pass
		    if (layer->weights == nullptr || layer->gradWeights == nullptr ||
			    layer->biases == nullptr || layer->gradBiases == nullptr)
			    continue;

		    layer->UpdateWeightsBiases(learningRate);
	    }
	    nvtxRangePop();
    }

    int Network::WriteFile()
    {
	    std::cout << ".. store weights to the storage .." << std::endl;
	    for (auto layer : layersVect)
	    {
		    int err = layer->SaveParameter();
		
		    if (err != 0)
		    {
			    std::cout << "-> error code: " << err << std::endl;
			    exit(err);
		    }
	    }

	    return 0;
    }

    int Network::LoadPretrain()
    {
	    for (auto layer : layersVect)
	    {
		    layer->SetLoadPretrain();
	    }

	    return 0;
    }

    // 1. initialize cuda resource container
    // 2. register the resource container to all the layers
    void Network::Cuda()
    {
	    cuda = new CudaContext();

	    std::cout << ".. model Configuration .." << std::endl;
	    for (auto layer : layersVect)
        {
		    std::cout << "CUDA: " << layer->GetName() << std::endl;
		    layer->SetCudaContext(cuda);
	    }
    }

    // 
    void Network::Train()
    {
	    phase = TRAINING;

    	// unfreeze all layers
	    for (auto layer : layersVect)
	    {
		    layer->UnFreeze();
	    }
    }

    void Network::Test()
    {
	    phase = INTERFACE;

	    // freeze all layers
	    for (auto layer : layersVect)
	    {
		    layer->Freeze();
	    }
    }

    std::vector<Layer*> Network::Layers()
    {
	    return layersVect;
    }

    float Network::Loss(Blob<float> *target)
    {
	    Layer *layer = layersVect.back();
	    return layer->GetLoss(target);
    }

    int Network::GetAccuracy(Blob<float> *target)
    {
	    Layer *layer = layersVect.back();
	    return layer->GetAccuracy(target);
    }

    /*#if 0
        Blob<float> *predict = this->output_;
	    int batch_size = predict->n();
	    int output_size = predict->c();

    #if (DEBUG_ACCURACY)
	    std::cout << "[[ ACCURACY ]]" << std::endl;
	    predict->print("predict:", true);
	    target->print("target:", true);
    #endif // DEBUG_ACCURACY

	    float* h_predict = predict->to(host);
	    float* h_target  = target->to(host);
	    cudaDeviceSynchronize();
	    int result = 0;
	    for (int b = 0; b < batch_size; b++)
	    {
		    int idx_predict = 0;
		    int idx_target = 0;
		    for (int j = 0; j < output_size; j++)
            {
			    if (h_predict[b*output_size + j] > h_predict[idx_predict])
				    idx_predict = j;
			    // std::cout << "[" << j << "]" << h_target[b*output_size + j] << ", " << h_target[idx_predict] << std::endl;
			    if (h_target[b*output_size + j] > h_target[idx_target])
				    idx_target = j;
		    }
		
    #if (DEBUG_ACCURACY)
		    std::cout << "predict:: " << idx_predict << ", target::" << idx_target << std::endl;
    #endif // DEBUG_ACCURACY
		    //std::cout << "p: " << idx_predict << ", y: " << idx_target << std::endl;

		    if (idx_predict == idx_target)
			    result++;
	    }

    #if (DEBUG_ACCURACY)
	    getchar();
    #endif // DEBUG_ACCURACY*/
}