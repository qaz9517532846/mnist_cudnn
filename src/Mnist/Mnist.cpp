#include "mnist_cudnn/Mnist.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <string>
 

namespace CUDA_NETWORK
{
    MNIST::MNIST()
    {
    }

    MNIST::~MNIST()
    {
        delete data;
        delete target;
    }
    
    void MNIST::CreateSharedSpace()
    {
        // create blobs with batch size and sample size
        data = new Blob<float>(batchSize_, channels_, height_, width_);
        printf("Data Finished\n");
        data->Tensor();
        printf("Data Tensor\n");
        target = new Blob<float>(batchSize_, numClasses_);
        printf("Data target\n");
    }

    std::vector<float> MNIST::ImageProcess(std::string imgFile)
    {
        cv::Mat colorImage = cv::imread(imgFile, cv::IMREAD_COLOR);
        cv::Mat grayImage, resizedImage, normalizedImage;

        cv::cvtColor(colorImage, grayImage, cv::COLOR_BGR2GRAY);
        cv::resize(grayImage, resizedImage, cv::Size(28, 28));
        height_ = 28;
        width_ = 28;
        channels_ = 1;
        cv::normalize(resizedImage, normalizedImage, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

        std::vector<float> img = std::vector<float>(normalizedImage.rows * normalizedImage.cols);

        int idx = 0;
        for (int y = 0; y < normalizedImage.rows; y++)
        {
            for (int x = 0; x < normalizedImage.cols; x++)
            {
                img[idx] = normalizedImage.at<float>(y, x);
                idx++;
            }
        }

        return img;
    }

    void MNIST::LoadTrainData(std::string filePath)
    {
        printf("Read File = %s\n", filePath.c_str());
        std::string dataDir = "/home/zmtech/catkin2_ws/src/mnist_cudnn/data/train/";
        std::ifstream file(filePath);
        
        // check file exist??
        if (!file.is_open())
        {
            std::cerr << "Not Open file." << std::endl;
            return;
        }
        
        std::string line, imgFile;
        int label;
        int numData_ = 0;
        numClasses_ = MNIST_CLASS;
        while (std::getline(file, line))
        {
            std::string imgPath;
            std::cout << line << std::endl;  // 輸出每一行內容
            std::istringstream iss(line);
            iss >> imgFile >> label;
            printf("Image File Name = %s, Lable = %d\n", imgFile.c_str(), label);

            imgPath = dataDir + std::to_string(label) + "/" + imgFile;
            std::vector<float> imageRawData = ImageProcess(imgPath);

            std::array<float, MNIST_CLASS> targetBatch;
            std::fill(targetBatch.begin(), targetBatch.end(), 0.f);

            targetBatch[static_cast<int>(label)] = 1.f;

            dataPool.push_back(imageRawData);
            targetPool.push_back(targetBatch);
            numData_++;
        }

        numSteps_ = numData_ / batchSize_;
        // close file
        file.close();
    }

    void MNIST::ShuffleDataset()
    {
        std::random_device rd;
        std::mt19937 gData(rd());
        auto gTarget = gData;
        
        std::shuffle(std::begin(dataPool), std::end(dataPool), gData);
        std::shuffle(std::begin(targetPool), std::end(targetPool), gTarget);
    }

    int MNIST::ToInt(uint8_t *ptr)
    {
        return ((ptr[0] & 0xFF) << 24 | (ptr[1] & 0xFF) << 16 |
                (ptr[2] & 0xFF) << 8 | (ptr[3] & 0xFF) << 0);
    }

    void MNIST::Train(std::string filePath, int batchSize, bool shuffle)
    {
        if (batchSize < 1)
        {
            std::cout << "batch size should be greater than 1." << std::endl;
            return;
        }
        
        batchSize_ = batchSize;
        shuffle = shuffle;

        LoadTrainData(filePath);
        
        if (shuffle)
            ShuffleDataset();
        
        CreateSharedSpace();
        
        step_ = 0;
    }
    
    void MNIST::Test(std::string imgFile)
    {
        batchSize_ = numSteps_ = 1;
        numClasses_ = MNIST_CLASS;
        std::vector<float> imageRawData = ImageProcess(imgFile);
        dataPool.push_back(imageRawData);

        CreateSharedSpace();

        step_ = 0;
        printf("Test Finished");
    }

    void MNIST::GetBatch()
    {
        if (step_ < 0)
        {
            std::cout << "You must initialize dataset first.." << std::endl;
            exit (-1);
        }
        
        // index clipping
        int dataIdx = step_ % numSteps_ * batchSize_;

        // prepare data blob
        int dataSize = channels_ * width_ * height_;
        
        // copy data
        for (int i = 0; i < batchSize_; i++)
            std::copy(dataPool[dataIdx + i].data(), &dataPool[dataIdx + i].data()[dataSize], &data->Ptr()[dataSize * i]);

        // copy target with one-hot encoded
        for (int i = 0; i < batchSize_; i++)
            std::copy(targetPool[dataIdx + i].data(), &targetPool[dataIdx + i].data()[MNIST_CLASS], &target->Ptr()[MNIST_CLASS * i]);
    }

    void MNIST::GetTestBatch()
    {
        if (step_ < 0)
        {
            std::cout << "You must initialize dataset first.." << std::endl;
            exit (-1);
        }
        
        // index clipping
        int dataIdx = step_ % numSteps_ * batchSize_;

        // prepare data blob
        int dataSize = channels_ * width_ * height_;
        
        // copy data
        for (int i = 0; i < batchSize_; i++)
            std::copy(dataPool[dataIdx + i].data(), &dataPool[dataIdx + i].data()[dataSize], &data->Ptr()[dataSize * i]);
    }

    int MNIST::Next()
    {
        step_++;
        GetBatch();

        return step_;
    }
}