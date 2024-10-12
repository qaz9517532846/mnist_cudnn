#ifndef NETWORK_H
#define NETWORK_H

#include <string>
#include <vector>

#include <cudnn.h>

#include "mnist_cudnn/Helper.h"
#include "mnist_cudnn/Loss.h"
#include "mnist_cudnn/Layer.h"

namespace CUDA_NETWORK
{
    typedef enum
    {
        TRAINING,
        INTERFACE
    } WORD_LOAD_TYPE;

    class Network
    {
        public:
            Network();
            ~Network();

            void AddLayer(Layer *layer);

            Blob<float> *Forward(Blob<float> *input);
            void Backward(Blob<float> *input = nullptr);
            void Update(float learningRate = 0.02f);

            int LoadPretrain();
            int WriteFile();
            
            float Loss(Blob<float> *target);
            int GetAccuracy(Blob<float> *target);
            
            void Cuda();
            void Train();
            void Test();
            
            Blob<float> *output;
            std::vector<Layer*> Layers();

        private:
            std::vector<Layer*> layersVect;
            CudaContext *cuda = nullptr;
            WORD_LOAD_TYPE phase = INTERFACE;
    };
}

#endif