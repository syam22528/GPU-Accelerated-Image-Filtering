#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

// CUDA Kernel for Grayscale Conversion
__global__ void grayscaleKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels; // Input pixel index
        int grayIdx = y * width + x;         // Output pixel index
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        output[grayIdx] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

void gpuGrayscale(const Mat& inputImage, Mat& outputImage) {
    // Image dimensions
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    // Allocate device memory
    unsigned char *d_input, *d_output;
    size_t inputSize = width * height * channels * sizeof(unsigned char);
    size_t outputSize = width * height * sizeof(unsigned char);
    cudaMalloc((void**)&d_input, inputSize);
    cudaMalloc((void**)&d_output, outputSize);

    // Copy input image to device
    cudaMemcpy(d_input, inputImage.data, inputSize, cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    grayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete

    // Copy result back to host
    cudaMemcpy(outputImage.data, d_output, outputSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    // Load input image
    Mat inputImage = imread("input.jpg");
    if (inputImage.empty()) {
        cerr << "Error loading image!" << endl;
        return -1;
    }

    // Prepare output image
    Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);

    // Measure GPU execution time
    auto gpuStart = chrono::high_resolution_clock::now();
    gpuGrayscale(inputImage, outputImage);
    auto gpuEnd = chrono::high_resolution_clock::now();
    chrono::duration<double> gpuTime = gpuEnd - gpuStart;
    cout << "GPU Execution Time: " << gpuTime.count() << " seconds" << endl;

    // Save and display the results
    imwrite("output_gpu.jpg", outputImage);
    cout << "Output saved as output_gpu.jpg" << endl;

    return 0;
}
