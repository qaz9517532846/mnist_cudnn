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
	Pad::Pad(std::string name, Layer *inputFrom, std::array<int, 8> paddings, int padValue)
	{
		SetLayerRelationship(inputFrom);

		layerName = name;
        paddings_ = paddings;
        padValue_ = padValue;
	}

	Pad::~Pad()
	{

	}

	Blob<float> *Pad::Forward(Blob<float> *input)
	{
		input = GetInput(input);

        // initilaize input and output
        if (input_ == nullptr || batchSize_ != input->num)
        {
            input_ = input;
            batchSize_ = input_->num;
            
            int n = input_->num + paddings_.at(0) + paddings_.at(1);
            int c = input_->channel + paddings_.at(2) + paddings_.at(3);
            int h = input_->height + paddings_.at(4) + paddings_.at(5);
            int w = input_->width + paddings_.at(6) + paddings_.at(7);
            if (output_ == nullptr)
                output_ = new Blob<float>(n, c, h, w);
            else
                output_->Reset(n, c, h, w);
        }
        
        // eigen implemented.
        Eigen::GpuStreamDevice stream;
        Eigen::GpuDevice gpuDevice(&stream);
        Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> gpuIn(input_->Cuda(), input_->num, input_->channel, input_->height, input_->width);
        Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> gpuOut(output_->Cuda(), output_->num, output_->channel, output_->height, output_->width);
        Eigen::array<std::pair<int, int>, 4> pads;
        pads[0] = std::make_pair(paddings_.at(0), paddings_.at(1));
        pads[1] = std::make_pair(paddings_.at(2), paddings_.at(3));
        pads[2] = std::make_pair(paddings_.at(4), paddings_.at(5));
        pads[3] = std::make_pair(paddings_.at(6), paddings_.at(7));
        gpuOut.device(gpuDevice) = gpuIn.pad(pads, padValue_);

        // cuda implemented.
        //config = GetGpuLaunchConfig(input_->Length(), PadForward, 0, 0);
        //PadForward<<<config.blockCnt, config.threadPerBlock>>>(input_->Length(), input_->Cuda(), output_->Cuda(), input_->num, input_->channel, input_->height, input_->width, paddings_.at(4));
        //cudaDeviceSynchronize();
        
    #if (DEBUG_PADDING & 0x01)
        std::cout << name_ << "[FORWARD]" << std::endl;
        input_->print(  name_ + "::input", true, input_->num, output_->height);
        output_->print( name_ + "::output", true, output_->num, output_->height);
    #endif
    
        return output_;
	}

	Blob<float> *Pad::Backward(Blob<float> *gradOutput)
	{
		gradOutput = SumGradients(gradOutput);

		if (gradInput_ == nullptr || batchSize_ != gradOutput->num)
		{
			gradOutput_ = gradOutput;
            batchSize_ = gradOutput_->num;

            int n = gradOutput_->num - paddings_.at(0) - paddings_.at(1);
            int c = gradOutput_->channel - paddings_.at(2) - paddings_.at(3);
            int h = gradOutput_->height - paddings_.at(4) - paddings_.at(5);
            int w = gradOutput_->width - paddings_.at(6) - paddings_.at(7);

			if (gradInput_ == nullptr)
				gradInput_ = new Blob<float>(n, c, h, w);
			else
				gradInput_->Reset(n, c, h, w);
            
            //config = GetGpuLaunchConfig(gradInput_->Length(), PadBackward, 0, 0);
		}
        
        if (!gradientStop)
        {
            // eigen implemented.
            Eigen::GpuStreamDevice stream;
            Eigen::GpuDevice gpuDevice(&stream);
            Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> gpuIn(gradOutput_->Cuda(), gradOutput_->num, gradOutput_->channel, gradOutput_->height, gradOutput_->width);
            Eigen::TensorMap <Eigen::Tensor<float, 4, Eigen::RowMajor>> gpuOut(gradInput_->Cuda(), gradInput_->num, gradInput_->channel, gradInput_->height, gradInput_->width);
            Eigen::array<int, 4> offsets = {paddings_.at(0), paddings_.at(2), paddings_.at(4), paddings_.at(6)};
            Eigen::array<int, 4> extents = {gradInput_->num, gradInput_->channel, gradInput_->height, gradInput_->width};
            gpuOut.device(gpuDevice) = gpuIn.slice(offsets, extents);

            // cuda implemented.
            //PadBackward <<<config.block_count, config.thread_per_block>>>(gradInput_->len(), gradOutput_->Cuda(), gradInput_->Cuda(), gradOutput_->n(), gradOutput_->c(), gradOutput_->h(), gradOutput_->w(), paddings_.at(4));
            //cudaDeviceSynchronize();
        }

    #if (DEBUG_PADDING & 0x02)
        std::cout << name_ << "[BACKWARD]" << std::endl;
        gradOutput_->Print(name_ + "::gradients", true, gradOutput_->num);
        if (!gradientStop_) gradInput_->Print(name_ + "::gdata", true);
    #endif // DEBUG_PADDING

		return gradInput_;
	}
}