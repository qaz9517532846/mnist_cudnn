#include <mnist_cudnn/Blob.h>
#include <mnist_cudnn/Helper.h>
#include <mnist_cudnn/Loss.h>
#include <mnist_cudnn/Mnist.h>
#include <mnist_cudnn/Network.h>
#include <mnist_cudnn/Layer.h>

#include <iomanip>
#include <map>
#include <nvtx3/nvToolsExt.h>

using namespace CUDA_NETWORK;

Layer *ResidualBlock(Network &model, Layer *mainline, int outChannels, int repetitions, int blockId, int &convName, int &fbnName, int &reluName, int &addName)
{
    Layer *shortcut = nullptr;
    int initStride;
    int initPadding;
    for (int i = 0; i < repetitions; i++)
    {
        initStride = 1;
        initPadding = 1;
        shortcut = mainline;

        if (i == 0)
        {
            if (blockId) initStride = 2;
            shortcut = model.AddLayer(new Conv2D("conv2d_" + std::to_string(++convName), shortcut, outChannels * 4, 1, initStride));
            shortcut = model.AddLayer(new FusedBatchNormalization("fbn_" +  std::to_string(++fbnName), shortcut, CUDNN_BATCHNORM_SPATIAL));
        }

        mainline = model.AddLayer(new Conv2D("conv2d_" + std::to_string(++convName), mainline, outChannels, 1, 1));
        mainline = model.AddLayer(new FusedBatchNormalization("fbn_" + std::to_string(++fbnName), mainline, CUDNN_BATCHNORM_SPATIAL));
        mainline = model.AddLayer(new Activation("relu_" + std::to_string(++reluName), mainline, CUDNN_ACTIVATION_RELU));

        if (i == 0 && blockId)
        {
            initPadding = 0;
            mainline = model.AddLayer(new Pad("pad_", mainline, {0, 0, 0, 0, 1, 1, 1, 1}, 0));
        }

        mainline = model.AddLayer(new Conv2D("conv2d_" + std::to_string(++convName), mainline, outChannels, 3, initStride, initPadding));
        mainline = model.AddLayer(new FusedBatchNormalization("fbn_" + std::to_string(++fbnName), mainline, CUDNN_BATCHNORM_SPATIAL));
        mainline = model.AddLayer(new Activation("relu_" + std::to_string(++reluName), mainline, CUDNN_ACTIVATION_RELU));
        mainline = model.AddLayer(new Conv2D("conv2d_" + std::to_string(++convName), mainline, outChannels * 4, 1, 1));
        mainline = model.AddLayer(new FusedBatchNormalization("fbn_" + std::to_string(++fbnName), mainline, CUDNN_BATCHNORM_SPATIAL));

        mainline = model.AddLayer(new Add("add_" + std::to_string(++addName), mainline, shortcut));
        mainline = model.AddLayer(new Activation("relu_" + std::to_string(++reluName), mainline, CUDNN_ACTIVATION_RELU));
    }

    return mainline;
}

int main(int argc, char** argv)
{
    int tpCount, step;
    int batchSize = 1;
    int numSteps = 1;
    bool loadPretrain = true;

    int resnetSize = 18;
    int outChannels = 64;
    int convName = 0;
    int fbnName = 0;
    int addName = 0;
    int reluName = 0;

    std::map<int, std::array<int, 4>> blockSize;

     // phase 2. inferencing
    // step 1. load test set
    std::cout << "[INFERENCE]" << std::endl;
    MNIST mnist;
    mnist.Test("/home/zmtech/data/test/3/mnist_test_200.png");

    // step 2. model initialization

    Network model;
    Layer *mainline = nullptr;
    mainline = model.AddLayer(new Pad("pad", mainline, {0, 0, 0, 0, 3, 3, 3, 3}, 0)); //[1,1,28,28] -> [1,1,34,34]
    mainline = model.AddLayer(new Conv2D("conv2d", mainline, 64, 7, 2)); //[1,1,34,34] -> [1,64,14,14]
    mainline = model.AddLayer(new FusedBatchNormalization("fbn", mainline, CUDNN_BATCHNORM_SPATIAL));
    mainline = model.AddLayer(new Activation("relu", mainline, CUDNN_ACTIVATION_RELU)); //[1,64,14,14] -> [1,64,14,14]
    mainline = model.AddLayer(new Pooling("pool", mainline, 3, 1, 2, CUDNN_POOLING_MAX)); //[1,64,14,14] -> [1,64,7,7]
    
    switch (resnetSize)
    {
        case 18:
            blockSize[18] = {2, 2, 2, 2};
            break;
        case 50:
            blockSize[50] = {3, 6, 4, 3};
            break;
        case 101:
            blockSize[101] = {3, 6, 23, 3};
        break;
    }


    for (int blockId = 0; blockId < 4; blockId++)
    {
        mainline = ResidualBlock(model, mainline, outChannels, blockSize[resnetSize].at(blockId), blockId, convName, fbnName, reluName, addName);
        outChannels *= 2;
    }

    mainline = model.AddLayer(new Dense("dense", mainline, 10)); //[1,500,1,1] -> [1,10,1,1]
    mainline = model.AddLayer(new Softmax("softmax", mainline));//[1,10,1,1] -> [1,10,1,1]
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