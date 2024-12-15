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

            std::string GetName();

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
            std::string layerName;

            // tensor descriptor for the input/output tensors
            cudnnTensorDescriptor_t inputDesc;
            cudnnTensorDescriptor_t outputDesc;

            // weight/bias descriptor
            cudnnFilterDescriptor_t filterDesc;
            cudnnTensorDescriptor_t weightDesc;
            cudnnTensorDescriptor_t biasDesc;


    
            // output memory
            Blob<float> *input_        = nullptr;    /* x  */
            Blob<float> *output_       = nullptr;    /* y  */
            Blob<float> *hInput_       = nullptr;    /* hInput  */
            Blob<float> *gradInput_    = nullptr;    /* dx */
            Blob<float> *gradOutput_   = nullptr;    /* dy */

            // master weights & bias
            bool freeze_               = false;     /* control parameter updates */
            Blob<float> *weights_      = nullptr;   /* w */
            Blob<float> *biases_       = nullptr;   /* b */
            Blob<float> *gradWeights_  = nullptr;   /* dw */
            Blob<float> *gradBiases_   = nullptr;   /* db */

            int batchSize_ = 0;  // mini-batch size
    
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
            cudnnActivationMode_t actMode;
            float actCoef;
    };

    class Softmax: public Layer
    {
        public:
            Softmax(std::string name);
            ~Softmax();

            Blob<float> *Forward(Blob<float> *input);
            Blob<float> *Backward(Blob<float> *gradInput);

            float GetLoss(Blob<float> *target);
            int   GetAccuracy(Blob<float> *target);

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
            int poolKernelSize;
            int poolPadding;
            int poolStride;
            cudnnPoolingMode_t poolMode;

            std::array<int, 4> outputSize;
            cudnnPoolingDescriptor_t poolDesc;
    };

    class RNN: public Layer
    {
        public:
            RNN(std::string name, const int hiddenSize, const int numLayer, double dropout, cudnnDirectionMode_t bidirectional, cudnnRNNMode_t mode);
            ~RNN();

            Blob<float> *Forward(Blob<float> *input);
            Blob<float> *Backward(Blob<float> *gradOutput);

        private:
            int hiddenSize_;
            int numLayer_;
            int inputSize_;
            int seqLen_;
            double dropout_;
            int dimHidden[3];
            int strideHidden[3];
            int *seqLengthArray;
            int *devSeqLengthArray;

            // output memory
            Blob<float> *hx_        = nullptr;    /* hx  */
            Blob<float> *hy_        = nullptr;    /* hy  */
            Blob<float> *cx_        = nullptr;    /* cx  */
            Blob<float> *cy_        = nullptr;    /* cy  */
            Blob<float> *dhx_       = nullptr;    /* dhx  */
            Blob<float> *dhy_       = nullptr;    /* dhy  */
            Blob<float> *dcx_       = nullptr;    /* dcx  */
            Blob<float> *dcy_       = nullptr;    /* dcy  */


            cudnnRNNMode_t mode_;
            cudnnDirectionMode_t bidirectional_;
            cudnnRNNDescriptor_t rnnDesc;
            cudnnDropoutDescriptor_t dropoutDesc;
            cudnnRNNDataDescriptor_t xDesc;
            cudnnRNNDataDescriptor_t yDesc;
            cudnnTensorDescriptor_t  hDesc;
            cudnnTensorDescriptor_t  cDesc;
            // Initialize dropout descriptor
            
            void *dropoutStates;
            size_t dropoutStatesSize = 0;
            size_t weightSize;
            size_t workspaceSize = 0;
            void** dWorkspace = nullptr;
            size_t reserveSize = 0;
            void** reserveSpace = nullptr;
            void** weightSpace = nullptr;
            void SetWorkspace();
    };
}

#endif