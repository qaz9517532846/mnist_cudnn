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

            virtual float GetLoss(Blob<float> *target);
            virtual int   GetAccuracy(Blob<float> *target);

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

    class Softmax: public Layer
    {
        public:
            Softmax(std::string name);
            ~Softmax();

            Blob<float> *Forward(Blob<float> *input);
            Blob<float> *Backward(Blob<float> *gradInput);

            float getLoss(Blob<float> *target);
            int   getAccuracy(Blob<float> *target);

        private:
            CrossEntropyLoss loss;
    };

    class Conv2D: public Layer
    {
        public:
            Conv2D(std::string name, int out_channels, int kernel_size, int stride = 1, int padding = 0, int dilation = 1);
            ~Conv2D();

            Blob<float> *Forward(Blob<float> *input);
            Blob<float> *Backward(Blob<float> *gradOutput);

        private:
            int outChannels;
            int kernelSize;
            int stride;
            int padding;
            int dilation;
    
            std::array<int, 4> outputSize;

            // convolution
            cudnnConvolutionDescriptor_t    convDesc;
    
            cudnnConvolutionFwdAlgo_t       convFwdAlgo;
            cudnnConvolutionBwdDataAlgo_t   convBwdDataAlgo;
            cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;

            size_t workspaceSize = 0;
            void** dWorkspace = nullptr;
            void SetWorkspace();
    };

    class Pooling: public Layer
    {
        public: 
            Pooling(std::string name, int kernelSize, int padding, int stride, cudnnPoolingMode_t mode);
            ~Pooling();

            Blob<float> *Forward(Blob<float> *input);
            Blob<float> *Backward(Blob<float> *gradOutput);

        private:
            int kernelSize;
            int padding;
            int stride;
            cudnnPoolingMode_t mode;

            std::array<int, 4> outputSize;
            cudnnPoolingDescriptor_t poolDesc;
    };
}

#endif