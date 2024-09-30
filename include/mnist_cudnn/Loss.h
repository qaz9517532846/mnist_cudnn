#ifndef LOSS_H
#define LOSS_H

#include "mnist_cudnn/Blob.h"

namespace CUDA_NETWORK
{
    class CrossEntropyLoss
    {
        public:
            CrossEntropyLoss();
            ~CrossEntropyLoss();

            float Loss(Blob<float> *predict, Blob<float> *target);
            //float Accuracy(Blob<float> *predict, Blob<float> *target);

        private:
            // reduced loss
            float hLoss = 0.f;
            float *dLoss = nullptr;

            float *dWorkspace = nullptr;
            void InitWorkspace(int batchSize);
    };
}

#endif // LOSS_H