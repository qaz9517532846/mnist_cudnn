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
    int tpCount, step;
    int batchSize = 1;
    int numSteps = 1;
    bool loadPretrain = true;

     // phase 2. inferencing
    // step 1. load test set
    std::cout << "[INFERENCE]" << std::endl;
    MNIST mnist;
    mnist.Test("/home/zmtech/data/test/3/mnist_test_200.png");

    // step 2. model initialization

    Network model;
    Layer *mainline = nullptr;
    model.AddLayer(new Conv2D("conv1", mainline, 20, 5));
    model.AddLayer(new Pooling("pool", mainline, 2, 0, 2, CUDNN_POOLING_MAX));
    model.AddLayer(new Conv2D("conv2", mainline, 50, 5));
    model.AddLayer(new Pooling("pool", mainline, 2, 0, 2, CUDNN_POOLING_MAX));
    model.AddLayer(new Dense("dense1", mainline, 500));
    model.AddLayer(new Activation("relu", mainline, CUDNN_ACTIVATION_RELU));
    model.AddLayer(new Dense("dense2", mainline, 10));
    model.AddLayer(new Softmax("softmax", mainline));
    model.Cuda();

    if(loadPretrain) model.LoadPretrain();
    model.Test();
    
    // step 3. iterates the testing loop
    Blob<float> *testData = mnist.GetData();
    mnist.GetTestBatch();


    // nvtx profiling start
    std::string nvtx_message = std::string("step" + std::to_string(step));
    nvtxRangePushA(nvtx_message.c_str());

    // update shared buffer contents
    testData->To(CUDA);

    // forward
    model.Forward(testData);

    // nvtx profiling stop
    nvtxRangePop();

    printf("Result = %d\n", model.Result());

    return 0;
}