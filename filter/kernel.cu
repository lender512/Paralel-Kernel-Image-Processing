#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

/* HARDWARE CONFIGURATION */
#define BLOCK_SIZE 16



/* KERNEL FILTER CONFIGURATION */

#define kernelDim 3

#define kernelDivisor 1

__constant__ float kernel[kernelDim*kernelDim] =

//{
//    0, 0, 1, 2, 1, 0, 0,
//    0, 3, 13, 22, 13, 3, 0,
//    1, 13, 59, 97, 59, 13, 1,
//    2, 22, 97, 159, 97, 22, 2,
//    1, 13, 59, 97, 59, 13, 1,
//    0, 3, 13, 22, 13, 3, 0,
//    0, 0, 1, 2, 1, 0, 0
//};

/*{
    1, 4, 6, 4, 1,
    4, 16, 24, 16, 4,
    2, 24, 36, 24, 2,
    4, 16, 24, 16, 4,
    1, 4, 6, 4, 1
 };*/

/*{
    1, 4, 6, 4, 1,
    4, 16, 24, 16, 4,
    2, 24, -476, 24, 2,
    4, 16, 24, 16, 4,
    1, 4, 6, 4, 1
 };*/

/*{ 
    -1, 0, 1,
    -1, 0, 1,
    -1, 0, 1 
};*/

{
    -1, 0, 1,
    -1, 0, 1,
    -1, 0, 1
};




/* MACRO UTILITY */
#define from2Dto1D(_cols, _i, _j) (((_cols) * _i) + (_j))
#define bound(_var, _lower, _upper) _var = (_var < _lower) ? _lower : (_var > _upper) ? _upper : _var


/* CUDA PROCEDURE */
__global__ void applyFilterCuda(uchar4* dst, const uchar4* src, const size_t rows, const size_t cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int dstPixel = from2Dto1D( cols, i, j );
    float dstValue[3] = {0, 0, 0};


    if ((i < rows) && (j < cols)) {
        for (int ki = 0; ki < kernelDim; ++ki) {
            for (int kj = 0; kj < kernelDim; ++kj) {

                const float kernelValue = kernel[from2Dto1D(kernelDim, ki, kj)];
                const int srcI = i + (ki - kernelDim / 2);
                const int srcJ = j + (kj - kernelDim / 2);
                const int srcPixel = from2Dto1D(cols, srcI, srcJ);

                if (srcI < 0 || srcI > rows || srcJ  < 0 || srcJ > cols) continue;

                dstValue[0] += kernelValue * src[srcPixel].x;
                dstValue[1] += kernelValue * src[srcPixel].y;
                dstValue[2] += kernelValue * src[srcPixel].z;
            }
        }

        dstValue[0] /= kernelDivisor;
        dstValue[1] /= kernelDivisor;
        dstValue[2] /= kernelDivisor;

        bound(dstValue[0], 0, 255);
        bound(dstValue[1], 0, 255); 
        bound(dstValue[2], 0, 255);

        dst[dstPixel].x = static_cast<uchar>( dstValue[0] );
        dst[dstPixel].y = static_cast<uchar>( dstValue[1] );
        dst[dstPixel].z = static_cast<uchar>( dstValue[2] );
    }
}

/* CONVERSION 8UC4 PROCEDURE */
void convert_8UC4(cv::Mat& image) {

    if (image.channels() == 3) cvtColor(image, image, cv::COLOR_BGR2BGRA);
    else if (image.channels() == 1) cvtColor(image, image, cv::COLOR_GRAY2BGRA);

    if (image.depth() != CV_8U) image.convertTo(image, CV_8U);

    if (image.channels() != 4 || image.depth() != CV_8U) {
        printf("\n\nERROR: Could not convert the image to 8UC4 !\n\n");
        exit(-1);
    }
}

/* INTERFACE FILTER PROCEDURE */
void applyFilter(std::string const& input, std::string const& output) {
    cv::Mat srcImg = cv::imread(input, cv::IMREAD_UNCHANGED);
    convert_8UC4(srcImg);
    cv::Mat dstImg(srcImg.size(), srcImg.type());

    const size_t rows = srcImg.rows;
    const size_t cols = srcImg.cols;
    const size_t bytes = sizeof(uchar4) * rows * cols;

    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 gridSize((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);


    uchar4* srcCuda, * dstCuda;


    cudaMalloc((void**)&srcCuda, bytes);
    cudaMalloc((void**)&dstCuda, bytes);

    cudaMemcpy(srcCuda, srcImg.ptr<uchar>(0), bytes, cudaMemcpyHostToDevice);
    cudaMemset(dstCuda, 255, bytes);


    /*float elapsed = 0;
    cudaEvent_t start, stop;
    (cudaEventCreate(&start));
    (cudaEventCreate(&stop));
    (cudaEventRecord(start, 0));*/

    applyFilterCuda << <gridSize, blockSize >> > (dstCuda, srcCuda, rows, cols);

    /*(cudaEventRecord(stop, 0));
    (cudaEventSynchronize(stop));
    (cudaEventElapsedTime(&elapsed, start, stop));
    (cudaEventDestroy(start));
    (cudaEventDestroy(stop));
    printf("The elapsed time in gpu was %.2f ms\n", elapsed);*/


    cudaMemcpy(dstImg.ptr<uchar>(0), dstCuda, bytes, cudaMemcpyDeviceToHost);

    cudaFree(srcCuda);
    cudaFree(dstCuda);


    cv::imwrite(output, dstImg);
}

/* MAIN EXAMPLE (COULD PASS FILE PATHS BY COMMAND LINE ARGUMENTS) */
int main() {
    const std::string folder = "C:/Users/INTEL/Downloads/";
    applyFilter(folder + "input.jpg", folder + "output.jpg");
    return 0;
}
