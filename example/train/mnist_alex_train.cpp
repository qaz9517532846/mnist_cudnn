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

    double learningRate = 0.02f;
    double lrDecay = 0.000005f;

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
    //first conv layer, pooling layer, and normalization layer
    model.AddLayer(new Conv2D("alexnet_conv1", mainline, 96, 11, 1, 5)); //[1,1,28,28] -> [1,96,28,28]
    model.AddLayer(new Activation("alexnet_relu1", mainline, CUDNN_ACTIVATION_RELU)); //[1,96,28,28] -> [1,96,28,28]
    model.AddLayer(new Pooling("alexnet_pool1", mainline, 2, 0, 2, CUDNN_POOLING_MAX)); //[1,96,28,28] -> [1,96,14,14]
    model.AddLayer(new LRN("alexnet_lrn1", mainline));//[1,96,14,14] -> [1,96,14,14]

    //second conv layer
    model.AddLayer(new Conv2D("alexnet_conv2", mainline, 256, 5, 1, 2)); //[1,96,14,14] -> [1,256,14,14]
    model.AddLayer(new Activation("alexnet_relu2", mainline, CUDNN_ACTIVATION_RELU)); //[1,256,14,14] -> [1,256,14,14]
    model.AddLayer(new Pooling("alexnet_pool2", mainline, 2, 0, 2, CUDNN_POOLING_MAX)); //[1,256,14,14] -> [1,256,7,7]
    model.AddLayer(new LRN("alexnet_lrn2", mainline));//[1,256,7,7] -> [1,256,7,7]

    //3rd conv layer
    model.AddLayer(new Conv2D("alexnet_conv3", mainline, 384, 3, 1, 1)); //[1,256,7,7] -> [1,384,7,7]
    model.AddLayer(new Activation("alexnet_relu3", mainline, CUDNN_ACTIVATION_RELU)); //[1,384,7,7] -> [1,384,7,7]

    //4th conv layer
    model.AddLayer(new Conv2D("alexnet_conv4", mainline, 384, 3, 1, 1)); //[1,256,7,7] -> [1,384,7,7]
    model.AddLayer(new Activation("alexnet_relu4", mainline, CUDNN_ACTIVATION_RELU)); //[1,384,7,7] -> [1,384,7,7]

    //5th conv layer
    model.AddLayer(new Conv2D("alexnet_conv5", mainline, 256, 3, 1, 1)); //[1,384,7,7] -> [1,256,7,7]
    model.AddLayer(new Activation("alexnet_relu5", mainline, CUDNN_ACTIVATION_RELU)); //[1,256,7,7] -> [1,256,7,7]
    model.AddLayer(new Pooling("alexnet_pool5", mainline, 2, 0, 2, CUDNN_POOLING_MAX)); //[1,256,7,7] -> [1,256,4,4]
    model.AddLayer(new LRN("alexnet_lrn5", mainline));//[1,256,4,4] -> [1,256,4,4]

    //1st fully connected layer
    model.AddLayer(new Dense("alexnet_dense1", mainline, 4096)); //[1,256,4,4] -> [1,4096,1,1]
    model.AddLayer(new Activation("alexnet_relu6", mainline, CUDNN_ACTIVATION_RELU)); //[1,4096,1,1] -> [1,4096,1,1]
    model.AddLayer(new Dropout("alexnet_dropout1", mainline, 0.5)); //[1,4096,1,1] -> [1,4096,1,1]

    //2nd fully connected layer
    model.AddLayer(new Dense("alexnet_dense2", mainline, 4096)); //[1,4096,1,1] -> [1,4096,1,1]
    model.AddLayer(new Activation("alexnet_relu7", mainline, CUDNN_ACTIVATION_RELU)); //[1,4096,1,1] -> [1,4096,1,1]
    model.AddLayer(new Dropout("alexnet_dropout2", mainline, 0.5)); ////[1,4096,1,1] -> [1,4096,1,1]

    //3rd fully connected layer
    model.AddLayer(new Dense("alexnet_dense3", mainline, 10)); //[1,4096,1,1]->[1,10,1,1]
    model.AddLayer(new Softmax("alexnet_softmax", mainline)); //[1,10,1,1]->[1,10,1,1]
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
        // we will use learning rate decay to the learning rate
        learningRate *= 1.f / (1.f + lrDecay * step);
        model.Update(learningRate);

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