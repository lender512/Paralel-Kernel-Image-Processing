﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>



#define from2Dto1D(i, j) (((j) * gridDim.x) + (i))
__global__ void betterCombKernel(const unsigned char* source, unsigned char* target, const int* kernel, int kernelDim, int divisor)
{

    //int kernelDim = 5;
    int x = blockIdx.x + kernelDim / 2;
    int y = blockIdx.y + kernelDim / 2;

    
    int idx = x + y * gridDim.x;
    int value = 0;

    for (int i = 0; i < kernelDim; ++i)
    {
        for (int j = 0; j < kernelDim; ++j) {
            value += kernel[j * kernelDim + i] * source[from2Dto1D(x + (i - kernelDim / 2), y + (j - kernelDim / 2))];
        }
    }

    value = value / (divisor);
    if (value < 255) {
        if (value < 0) {
            target[idx] = 0;
        }
        else {
            target[idx] = value;
        }
    }
    else {
        target[idx] = 255;
    }
    
}


int main() {

    std::string folder = "C:/Users/luis/Desktop/source/";

    cv::Mat img = cv::imread(folder + "input.tif");



    cv::Mat filterImgMat(img);
    unsigned char* sourceR;
    unsigned char* sourceG;
    unsigned char* sourceB;
    unsigned char* targetR;
    unsigned char* targetG;
    unsigned char* targetB;
    int* kernel;
    
    const int height = img.size().height;
    const int width = img.size().width;
    dim3 grid(width, height);


    /*int kernelH[] = { 1, 4, 6, 4, 1,
                     4, 16, 24, 16, 4,
                     2, 24, 36, 24, 2,
                     4, 16, 24, 16, 4,
                     1, 4, 6, 4, 1};*/

    /*int kernelH[] = { 1, 4, 6, 4, 1,
                    4, 16, 24, 16, 4,
                    2, 24, -476, 24, 2,
                    4, 16, 24, 16, 4,
                    1, 4, 6, 4, 1};*/

    int kernelH[] = {
        0, 0, 1, 2, 1, 0, 0,
        0, 3, 13, 22, 13, 3, 0,
        1, 13, 59, 97, 59, 13, 1,
        2, 22, 97, 159, 97, 22, 2,
        1, 13, 59, 97, 59, 13, 1,
        0, 3, 13, 22, 13, 3, 0,
        0, 0, 1, 2, 1, 0, 0,
    };


    

   /* int kernelH[] = { -1, 0, 1,
                      -1, 0, 1,
                      -1, 0, 1 };*/
    int divisor = 1003;


    int kernelDim = sqrt(sizeof(kernelH)/sizeof(int));

    cudaMalloc((void**)&kernel, kernelDim * kernelDim * sizeof(int));


    cudaMalloc((void**)&sourceR, width * height * sizeof(char));
    cudaMalloc((void**)&sourceG, width * height * sizeof(char));
    cudaMalloc((void**)&sourceB, width * height * sizeof(char));
    cudaMalloc((void**)&targetR, width * height * sizeof(char));
    cudaMalloc((void**)&targetG, width * height * sizeof(char));
    cudaMalloc((void**)&targetB, width * height * sizeof(char));



    uchar* imgMatrixR = new uchar[width * height];
    uchar* imgMatrixG = new uchar[width * height];
    uchar* imgMatrixB = new uchar[width * height];

    for (int col = 0; col < height; ++col) {
        for (int row = 0; row < width; ++row) {
            auto v = img.at<cv::Vec3b>(col, row);
            imgMatrixR[width * col + row] = v[0];
            imgMatrixG[width * col + row] = v[1];
            imgMatrixB[width * col + row] = v[2];
        }
    }

    cudaMemcpy(kernel, kernelH, kernelDim * kernelDim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sourceR, imgMatrixR, width * height * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(sourceG, imgMatrixG, width * height * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(sourceB, imgMatrixB, width * height * sizeof(char), cudaMemcpyHostToDevice);
    
 

    betterCombKernel << < grid, 1 >> > (sourceR, targetR, kernel, kernelDim, divisor);
    betterCombKernel << < grid, 1 >> > (sourceG, targetG, kernel, kernelDim, divisor);
    betterCombKernel << < grid, 1 >> > (sourceB, targetB, kernel, kernelDim, divisor);


    cudaMemcpy(imgMatrixR, targetR, width * height * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(imgMatrixG, targetG, width * height * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(imgMatrixB, targetB, width * height * sizeof(char), cudaMemcpyDeviceToHost);

    for (int col = kernelDim/2; col < height; ++col) {
        for (int row = kernelDim/2; row < width; ++row) {
            cv::Vec3b v = filterImgMat.at<cv::Vec3b>(col, row);
            v[0] = imgMatrixR[width * col + row];
            v[1] = imgMatrixG[width * col + row];
            v[2] = imgMatrixB[width * col + row];
            filterImgMat.at<cv::Vec3b>(col, row) = v;
        }
    }

    cv::imwrite(folder + "output5.jpg", filterImgMat);
    
    cudaFree(sourceR);
    cudaFree(sourceG);
    cudaFree(sourceB);
    cudaFree(targetR);
    cudaFree(targetG);
    cudaFree(targetB);
    cudaFree(kernel);
    delete [] imgMatrixR;
    delete [] imgMatrixG;
    delete [] imgMatrixB;

}