#include <mnist_cudnn/Blob.h>
#include <mnist_cudnn/Helper.h>
#include <mnist_cudnn/Loss.h>
#include <mnist_cudnn/Mnist.h>
#include <mnist_cudnn/Network.h>
#include <mnist_cudnn/Layer.h>

#include <iomanip>
#include <nvtx3/nvToolsExt.h>


using namespace CUDA_NETWORK;

int main(int argc, char** argv)
{
    /* configure the network */
    int batchSizeTrain = 256;
    int numStepsTrain = 2400;
    int monitoringStep = 200;

    float learningRate = 0.00001f;
    float decay = 0.9f;
    float epsHat = 0.0000000001f;

    bool loadPretrain = false;
    bool fileSave = true;

    MNIST mnist;

    /* Welcome Message */
    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    // phase 1. training
    std::cout << "[TRAIN]" << std::endl;
    mnist.Train("/home/zmtech/data/train/", "train.txt", batchSizeTrain, true);

    // step 2. model initialization
    Layer *mainline = nullptr;
    Network model;
    model.AddLayer(new Conv2D("vgg_conv_1_1", mainline, 64, 3, 1, 1)); //[1,1,28,28] -> [1,64,28,28]
    //model.AddLayer(new FusedBatchNormalization("vgg_bn_1_1", mainline, CUDNN_BATCHNORM_SPATIAL));//[1,64,28,28] -> [1,64,28,28]
    model.AddLayer(new Activation("vgg_relu_1_1", mainline, CUDNN_ACTIVATION_RELU)); //[1,64,28,28] -> [1,64,28,28]
    model.AddLayer(new Conv2D("vgg_conv_1_2", mainline, 64, 3, 1, 1)); //[1,64,28,28] -> [1,64,28,28]
    //model.AddLayer(new FusedBatchNormalization("vgg_bn_1_2", mainline, CUDNN_BATCHNORM_SPATIAL));//[1,64,28,28] -> [1,64,28,28]
    model.AddLayer(new Activation("vgg_relu_1_2", mainline, CUDNN_ACTIVATION_RELU)); //[1,64,28,28] -> [1,64,28,28]
    model.AddLayer(new Pooling("vgg_pool_1_1", mainline, 2, 0, 2, CUDNN_POOLING_MAX)); //[1,64,28,28] -> [1,64,14,14]

    model.AddLayer(new Conv2D("vgg_conv_2_1", mainline, 128, 3, 1, 1)); //[1,64,14,14] -> [1,128,14,14]
    //model.AddLayer(new FusedBatchNormalization("vgg_bn_2_1", mainline, CUDNN_BATCHNORM_SPATIAL));//[1,128,14,14] -> [1,128,14,14]
    model.AddLayer(new Activation("vgg_relu_2_1", mainline, CUDNN_ACTIVATION_RELU)); //[1,128,14,14] -> [1,128,14,14]
    model.AddLayer(new Conv2D("vgg_conv_2_2", mainline, 128, 3, 1, 1)); //[1,128,14,14] -> [1,128,14,14]
    //model.AddLayer(new FusedBatchNormalization("vgg_bn_2_2", mainline, CUDNN_BATCHNORM_SPATIAL));//[1,128,14,14] -> [1,128,14,14]
    model.AddLayer(new Activation("vgg_relu_2_2", mainline, CUDNN_ACTIVATION_RELU)); //[1,128,14,14] -> [1,128,14,14]
    model.AddLayer(new Pooling("vgg_pool_2_1", mainline, 2, 0, 2, CUDNN_POOLING_MAX)); //[1,128,14,14] -> [1,128,7,7]

    model.AddLayer(new Conv2D("vgg_conv_3_1", mainline, 256, 3, 1, 1)); //[1,128,7,7] -> [1,256,7,7]
    //model.AddLayer(new FusedBatchNormalization("vgg_bn_3_1", mainline, CUDNN_BATCHNORM_SPATIAL));//[1,256,7,7] -> [1,256,7,7]
    model.AddLayer(new Activation("vgg_relu_3_1", mainline, CUDNN_ACTIVATION_RELU)); //[1,256,7,7] -> [1,256,7,7]
    model.AddLayer(new Conv2D("vgg_conv_3_2", mainline, 256, 3, 1, 1)); //[1,256,7,7] -> [1,256,7,7]
    //model.AddLayer(new FusedBatchNormalization("vgg_bn_3_2", mainline, CUDNN_BATCHNORM_SPATIAL));//[1,256,7,7] -> [1,256,7,7]
    model.AddLayer(new Activation("vgg_relu_3_2", mainline, CUDNN_ACTIVATION_RELU)); //[1,256,7,7] -> [1,256,7,7]
    model.AddLayer(new Pooling("vgg_pool_3_1", mainline, 2, 0, 2, CUDNN_POOLING_MAX)); //[1,256,7,7] -> [1,256,3,3]

    model.AddLayer(new Conv2D("vgg_conv_4_1", mainline, 512, 3, 1, 1)); //[1,256,3,3] -> [1,512,3,3]
    //model.AddLayer(new FusedBatchNormalization("vgg_bn_4_1", mainline, CUDNN_BATCHNORM_SPATIAL));//[1,512,3,3] -> [1,512,3,3]
    model.AddLayer(new Activation("vgg_relu_4_1", mainline, CUDNN_ACTIVATION_RELU)); //[1,512,3,3] -> [1,512,3,3]
    model.AddLayer(new Conv2D("vgg_conv_4_2", mainline, 512, 3, 1, 1)); //[1,512,3,3] -> [1,512,3,3]
    //model.AddLayer(new FusedBatchNormalization("vgg_bn_4_2", mainline, CUDNN_BATCHNORM_SPATIAL));//[1,512,3,3] -> [1,512,3,3]
    model.AddLayer(new Activation("vgg_relu_4_2", mainline, CUDNN_ACTIVATION_RELU)); //[1,512,3,3] -> [1,512,3,3]
    model.AddLayer(new Pooling("vgg_pool_4_1", mainline, 2, 0, 2, CUDNN_POOLING_MAX)); //[1,512,3,3] -> [1,512,1,1]

    model.AddLayer(new Conv2D("vgg_conv_5_1", mainline, 512, 3, 1, 1)); //[1,512,1,1] -> [1,512,1,1]
    //model.AddLayer(new FusedBatchNormalization("vgg_bn_5_1", mainline, CUDNN_BATCHNORM_SPATIAL));//[1,512,1,1] -> [1,512,1,1]
    model.AddLayer(new Activation("vgg_relu_5_1", mainline, CUDNN_ACTIVATION_RELU)); //[1,512,1,1] -> [1,512,1,1]
    model.AddLayer(new Conv2D("vgg_conv_5_2", mainline, 512, 3, 1, 1)); //[1,512,1,1] -> [1,512,1,1]
    //model.AddLayer(new FusedBatchNormalization("vgg_bn_5_2", mainline, CUDNN_BATCHNORM_SPATIAL));//[1,512,1,1] -> [1,512,1,1]
    model.AddLayer(new Activation("vgg_relu_5_2", mainline, CUDNN_ACTIVATION_RELU)); //[1,512,1,1] -> [1,512,1,1]
    // model.AddLayer(new Pooling("vgg_pool_5_2", mainline, 2, 0, 2, CUDNN_POOLING_MAX)); //[1,512,1,1] -> [1,512,1,1]

    model.AddLayer(new Dense("vgg_dense_6_1", mainline, 256)); //[1,512,1,1] -> [1,256,1,1]
    model.AddLayer(new Activation("vgg_relu_6_1", mainline, CUDNN_ACTIVATION_RELU)); //[1,256,1,1] -> [1,256,1,1]
    model.AddLayer(new Dropout("vgg_dropout_6_1", mainline, 0.5)); //[1,256,1,1] -> [1,256,1,1]
    model.AddLayer(new Dense("vgg_dense_6_2", mainline, 256)); //[1,256,1,1] -> [1,256,1,1]
    model.AddLayer(new Activation("vgg_relu_6_2", mainline, CUDNN_ACTIVATION_RELU)); //[1,256,1,1] -> [1,256,1,1]
    model.AddLayer(new Dropout("vgg_dropout_6_2", mainline, 0.5)); //[1,256,1,1] -> [1,256,1,1]
    model.AddLayer(new Dense("vgg_dense_6_3", mainline, 10)); //[1,256,1,1] -> [1,10,1,1]
    model.AddLayer(new Softmax("vgg_softmax_6_3", mainline)); //[1,10,1,1]->[1,10,1,1]

    model.Cuda();

    if(loadPretrain) model.LoadPretrain();
    model.Train();

    // step 3. train
    int step = 0;
    Blob<float> *trainData = mnist.GetData();
    Blob<float> *trainTarget = mnist.GetTarget();
    mnist.GetBatch();

    int tpCount = 0;

    while (step < numStepsTrain)
    {
        // nvtx profiling start
        std::string nvtxMessage = std::string("step" + std::to_string(step));
        nvtxRangePushA(nvtxMessage.c_str());

        // update shared buffer contents
        trainData->To(CUDA);
        trainTarget->To(CUDA);
        
        // forward
        model.Forward(trainData);
        tpCount += model.GetAccuracy(trainTarget);

        // back-propagation
        model.Backward(trainTarget);

        // update parameter
        model.UpdateRmsprop(learningRate, decay, epsHat);

        // fetch next data
        step = mnist.Next();

        // nvtx profiling end
        nvtxRangePop();

        // calculation softmax loss
        if (step % monitoringStep == 0)
        {
            float loss = model.Loss(trainTarget);
            float accuracy =  100.f * tpCount / monitoringStep / batchSizeTrain;
            
            std::cout << "step: " << std::right << std::setw(4) << step << \
                         ", loss: " << std::left << std::setw(5) << std::fixed << std::setprecision(3) << loss << \
                         ", accuracy: " << accuracy << "%" << std::endl;

            tpCount = 0;
        }
    }

    // trained parameter save
    if (fileSave) model.WriteFile();

    return 0;
}