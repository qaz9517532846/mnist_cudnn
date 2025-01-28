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
 	 * Add definition                                             *
 	****************************************************************/
	Add::Add(std::string name, Layer *inputFrom, Layer *input2From)
	{
		SetLayerRelationship(inputFrom, input2From);

		layerName = name;
	}

	Add::~Add()
	{

	}

	Blob<float> *Add::Forward(Blob<float> *input)
	{
		input = GetInput(input);

        // initilaize input and output
        if (input_ == nullptr || batchSize_ != input->num)
        {
            input_ = input;
            batchSize_ = input->num;
            
            if (output_ == nullptr)
                output_ = new Blob<float>(input->Shape());
            else
                output_->Reset(input->Shape());
            //config = GetGpuLaunchConfig(input_->Length(), AddUpdate, 0, 0);
        }

        // y = x + x2
        AddUpdate<<<BLOCK_DIM_1D, BLOCK_DIM_1D>>>(input_->Cuda(), input2_->Cuda(), output_->Cuda(), input_->Length());

    #if (DEBUG_ADD & 0x01)
        std::cout << name_ << "[FORWARD]" << std::endl;
        input_->Print(name_ + "::input", true, input_->n());
        input2_->Print(name_ + "::input", true, input2_->n());
        output_->Print(name_ + "::output", true);
    #endif

        return output_;
	}

	Blob<float> *Add::Backward(Blob<float> *gradOutput)
	{
		gradOutput = SumGradients(gradOutput);

		if (gradInput_ == nullptr || batchSize_ != gradOutput->num)
		{
			gradOutput_ = gradOutput;

			if (gradInput_ == nullptr)
				gradInput_ = new Blob<float>(input_->Shape());
			else
				gradInput_->Reset(input_->Shape());
		}

        // y = x + x2
        // dy/dx = dy/dx2 = 1
        CheckCudaErrors(cudaMemcpy(gradInput_->Cuda(), gradOutput_->Cuda(), gradOutput_->BufSize(), cudaMemcpyDeviceToDevice));

    #if (DEBUG_ADD & 0x02)
        std::cout << name_ << "[BACKWARD]" << std::endl;
        gradOutput_->Print(name_ + "::gradients", true, gradOutput_->num);
        if (!gradientStop_)
            gradInput_->Print(name_ + "::gdata", true);
    #endif // DEBUG_PADDING

		return gradInput_;
	}
}