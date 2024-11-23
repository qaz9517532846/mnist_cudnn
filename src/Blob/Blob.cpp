#include "mnist_cudnn/Blob.h"

namespace CUDA_NETWORK
{
    template <typename ftype>
    inline Blob<ftype>::Blob(int n, int c, int h, int w): num(n), channel(c), height(h), width(w) 
    {
        hPtr = new float[num * channel * height * width];
    }

    template <typename ftype>
    inline Blob<ftype>::Blob(std::array<int, 4> size): num(size[0]), channel(size[1]), height(size[2]), width(size[3]) 
    {
        hPtr = new float[num * channel * height * width];
    }

    template <typename ftype>
    inline Blob<ftype>::~Blob() 
    { 
        if (hPtr != nullptr) delete [] hPtr; 
	    if (dPtr != nullptr) cudaFree(dPtr);
        if (isTensor) cudnnDestroyTensorDescriptor(tensorDesc);
    }

    // reset the current blob with the new size information
    template <typename ftype>
    void Blob<ftype>::Reset(int n, int c, int h, int w)
    {
        // update size information
        num = n;
        channel = c;
        height = h;
        width = w;

        // terminate current buffers
        if(hPtr != nullptr)
        {
            delete [] hPtr;
            hPtr = nullptr;
        }

        if(dPtr != nullptr)
        {
            cudaFree(dPtr);
            dPtr = nullptr;
        }

        // create new buffer
        hPtr = new float[num * channel * height * width];
        Cuda();

        // reset tensor descriptor if it was tensor
        if(isTensor)
        {
            cudnnDestroyTensorDescriptor(tensorDesc);
            isTensor = false;
        }
    }

    template <typename ftype>
    void Blob<ftype>::Reset(std::array<int, 4> size)
    {
        Reset(size[0], size[1], size[2], size[3]);
    }

    // returns array of tensor shape
    template <typename ftype>
    std::array<int, 4> Blob<ftype>::Shape()
    {
        return std::array<int, 4>({num, channel, height, width});
    }

    // returns number of elements for 1 batch
    template <typename ftype>
    int Blob<ftype>::Size()
    {
        return channel * height * width;
    }

    // returns number of total elements in blob including batch
    template <typename ftype>
    int Blob<ftype>::Length()
    {
        return num * channel * height * width;
    }

    // returns size of allocated memory
    template<typename ftype>
    int Blob<ftype>::BufSize()
    {
        return sizeof(ftype) * Length();
    }

    /* Tensor Control */
    template <typename ftype>
    cudnnTensorDescriptor_t Blob<ftype>::Tensor()
    {
        if (isTensor) return tensorDesc;
        
        cudnnCreateTensorDescriptor(&tensorDesc);
        cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num, channel, height, width);
        isTensor = true;

        return tensorDesc;
    }

    /* Memory Control */
    // get specified memory pointer
    template<typename ftype>
    ftype* Blob<ftype>::Ptr()
    { 
        return hPtr;
    }


    // get cuda memory
    template<typename ftype>
    ftype* Blob<ftype>::Cuda()
    {
        if(dPtr == nullptr)  cudaMalloc((void**)&dPtr, sizeof(ftype) * Length());
    
        return dPtr;
    }

    // transfer data between memory
    template<typename ftype>
    ftype* Blob<ftype>::To(DEV_TYPE target)
    {
        ftype *ptr = nullptr;
        if(target == HOST)
        {
            cudaMemcpy(hPtr, Cuda(), sizeof(ftype) * Length(), cudaMemcpyDeviceToHost);
            ptr = hPtr;
        }
        else // DeviceType::cuda
        {
            cudaMemcpy(Cuda(), hPtr, sizeof(ftype) * Length(), cudaMemcpyHostToDevice);
            ptr = dPtr;
        }
    
        return ptr;
    }

    template <typename ftype>
    void Blob<ftype>::Print(std::string name, bool viewParam, int numBatch, int width)
    {
        To(HOST);
        std::cout << "**" << name << "\t: (" << Size() << ")\t";
        std::cout << ".n: " << num << ", .c: " << channel << ", .h: " << height << ", .w: " << width;
        std::cout << std::hex << "\t(h:" << hPtr << ", d:" << dPtr << ")" << std::dec << std::endl;

        if(viewParam)
        {
            std::cout << std::fixed;
            std::cout.precision(6);
            
            int maxPrintLine = 4;
            if (width == 28)
            {
                std::cout.precision(3);
                maxPrintLine = 28;
            }

            int offset = 0;

            for(int n = 0; n < numBatch; n++)
            {
                if(numBatch > 1) std::cout << "<--- batch[" << n << "] --->" << std::endl;
                int count = 0;
                int printLineCount = 0;
                while (count < Size() && printLineCount < maxPrintLine)
                {
                    std::cout << "\t";
                    for(int s = 0; s < width && count < Size(); s++)
                    {
                        std::cout << hPtr[Size() * n + count + offset] << "\t";
                        count++;
                    }

                    std::cout << std::endl;
                    printLineCount++;
                }
            }

            std::cout.unsetf(std::ios::fixed);
        }
    }

    /* pretrained parameter load and save */
    template <typename ftype>
    int Blob<ftype>::FileRead(std::string filename)
    {
        std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
        if (!file.is_open())
        {
            std::cout << "Fail to access " << filename << std::endl;
            return -1;
        }

        file.read((char*)hPtr, sizeof(float) * this->Length());
        this->To(CUDA);
        file.close();

        return 0;
    }

    template <typename ftype>
    int Blob<ftype>::FileWrite(std::string filename)
    {
        std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary );
        if (!file.is_open())
        {
            std::cout << "Fail to write " << filename << std::endl;
            return -1;
        }

        file.write((char*)this->To(HOST), sizeof(float) * this->Length());
        file.close();

        return 0;
    }

    template class Blob<float>;
}