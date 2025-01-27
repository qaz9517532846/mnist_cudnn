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
    int numStepsTrain = 500;
    int monitoringStep = 100;
    int hiddenSize = 128;
    int numLayer = 4;

    double learningRate = 0.02;
    double dropOut = 0.1;

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
    model.AddLayer(new RNN("rnn1", mainline, hiddenSize, numLayer, dropOut, CUDNN_UNIDIRECTIONAL, CUDNN_RNN_RELU));
    model.AddLayer(new Dense("dense1", mainline, 10));
    model.AddLayer(new Activation("relu", mainline, CUDNN_ACTIVATION_RELU));
    model.AddLayer(new Softmax("softmax", mainline));
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