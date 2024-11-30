#ifndef IMAGE_PRO_H
#define IMAGE_PRO_H

#include <string>
#include <fstream>
#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <assert.h>

#include "mnist_cudnn/Blob.h"
#include "mnist_cudnn/Helper.h"
#include "mnist_cudnn/Layer.h"
#include "mnist_cudnn/Network.h"

#define MNIST_CLASS 10

namespace CUDA_NETWORK
{
    class MNIST
    {
        public:
            MNIST();
            ~MNIST();

            // load train dataset
            void Train(std::string filePath, int batchSize = 1, bool shuffle = false);
            
            // load test dataset
            void Test(std::string imgFile);
            
            // update shared batch data buffer at current step index
            void GetBatch();
            void GetTestBatch();
            
            // increase current step index
            // optionally it updates shared buffer if input parameter is true.
            int  Next();
            
            // returns a pointer which has input batch data
            Blob<float>* GetData()   { return data;  }
            // returns a pointer which has target batch data
            Blob<float>* GetTarget() { return target; }

        private:
            // container
            std::vector<float> ImageProcess(std::string imgFile);
            std::vector<std::vector<float>> dataPool;
            std::vector<std::array<float, MNIST_CLASS>> targetPool;
            Blob<float>* data = nullptr;
            Blob<float>* target = nullptr;

            // data loader initialization
            void LoadTrainData(std::string dataPath);
            
            int ToInt(uint8_t *ptr);
            
            std::string filePath;
            // data loader control
            int  step_       = -1;
            bool shuffle;
            int  batchSize_  = 1;
            int  channels_   = 1;
            int  height_     = 1;
            int  width_      = 1;
            int  numClasses_ = 10;
            int  numSteps_   = 0;
            
            void CreateSharedSpace();
            void ShuffleDataset();
    };
}
#endif