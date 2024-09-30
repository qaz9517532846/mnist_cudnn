#ifndef BLOB_H
#define BLOB_H

#include <array>
#include <string>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include <cudnn.h>

namespace CUDA_NETWORK
{
    typedef enum
    {
        HOST,
        CUDA
    } DEV_TYPE;

    template <typename ftype>
    class Blob
    {
        public:
            Blob(int n = 1, int ch = 1, int h = 1, int w = 1);
            Blob(std::array<int, 4> size);
            ~Blob();
            void Reset(int n = 1, int c = 1, int h = 1, int w = 1);
            void Reset(std::array<int, 4> size);
            std::array<int, 4> Shape();
            int Size();

            int Length();
            int BufSize();
            cudnnTensorDescriptor_t tensorDesc;
            cudnnTensorDescriptor_t Tensor();
            ftype *Ptr();
            ftype *Cuda();
            ftype *To(DEV_TYPE target);
            void Print(std::string name, bool viewParam = false, int numBatch = 1, int width = 16);
            int FileRead(std::string filename);
            int FileWrite(std::string filename);

            int size;
            int lenght;
            int bufSize;
            int num;
            int channel;
            int height;
            int width;
            bool isTensor = false;
        private:
            ftype *hPtr = nullptr;
            ftype *dPtr = nullptr;
            
    };
} //namespace CUDA_NETWORK

#endif