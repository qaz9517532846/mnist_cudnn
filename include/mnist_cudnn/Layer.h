#ifndef LAYER_H
#define LAYER_H

#include "mnist_cudnn/Blob.h"
#include "mnist_cudnn/Loss.h"
#include "mnist_cudnn/Helper.h"

namespace CUDA_NETWORK
{
    class Layer
    {
        public:
            Layer();
            ~Layer();

            virtual Blob<float> *Forward(Blob<float> *input) = 0;
            virtual Blob<float> *Backward(Blob<float> *gradInput) = 0;

            std::string getName();

            virtual float getLoss(Blob<float> *target);
            virtual int   getAccuracy(Blob<float> *target);

            void SetCudaContext(CudaContext *context);

            void SetLoadPretrain();
            void SetGradientStop();

            /* Weight Freeze or Unfreeze */
            void Freeze();
            void UnFreeze();

        protected:
            // name of layer
            std::string name;

            // tensor descriptor for the input/output tensors
            cudnnTensorDescriptor_t inputDesc;
            cudnnTensorDescriptor_t outputDesc;

            // weight/bias descriptor
            cudnnFilterDescriptor_t filterDesc;
            cudnnTensorDescriptor_t biasDesc;
    
            // output memory
            Blob<float> *input        = nullptr;    /* x  */
            Blob<float> *output       = nullptr;    /* y  */
            Blob<float> *gradInput    = nullptr;    /* dx */
            Blob<float> *gradOutput   = nullptr;    /* dy */

            // master weights & bias
            bool freeze               = false;     /* control parameter updates */
            Blob<float> *weights      = nullptr;   /* w */
            Blob<float> *biases       = nullptr;   /* b */
            Blob<float> *gradWeights  = nullptr;   /* dw */
            Blob<float> *gradBiases   = nullptr;   /* db */

            int batchSize = 0;  // mini-batch size
    
            // initialize weights along with the input size
            void InitWeightBias(unsigned int seed = 0);
            void UpdateWeightsBiases(float learningRate);

            // cuda handle container
            CudaContext *cuda = nullptr;

            // pretrain parameters
            bool loadPretrain = false;
            int LoadParameter();
            int SaveParameter();

            // gradient stop tagging
            bool gradientStop = false;

            friend class Network;
    };

    class Dense: public Layer
    {
        public:
            Dense(std::string name, int outSize);
            ~Dense();

            Blob<float> *Forward(Blob<float> *input);
            Blob<float> *Backward(Blob<float> *gradInput);

        private: 
            int inputSize = 0;
            int outputSize = 0;

            float *dOneVec = nullptr;
    };

    class Activation: public Layer
    {
        public:
            Activation(std::string name, cudnnActivationMode_t mode, float coef = 0.f);
            ~Activation();

            Blob<float> *Forward(Blob<float> *input);
            Blob<float> *Backward(Blob<float> *gradInput);

        private:
            cudnnActivationDescriptor_t actDesc;
            cudnnActivationMode_t mode;
            float coef;
    };

}

#endif