#include "cudnn_image_processor.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>

namespace {

constexpr int kChannels = 1;
constexpr int kFilterSize = 3;

void checkCuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

void checkCudnn(cudnnStatus_t result, const char* operation) {
  if (result != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudnnGetErrorString(result));
  }
}

struct CudnnHandleDeleter {
  void operator()(cudnnContext* handle) const {
    if (handle != nullptr) {
      cudnnDestroy(handle);
    }
  }
};

struct TensorDescriptorDeleter {
  void operator()(cudnnTensorStruct* descriptor) const {
    if (descriptor != nullptr) {
      cudnnDestroyTensorDescriptor(descriptor);
    }
  }
};

struct FilterDescriptorDeleter {
  void operator()(cudnnFilterStruct* descriptor) const {
    if (descriptor != nullptr) {
      cudnnDestroyFilterDescriptor(descriptor);
    }
  }
};

struct ConvolutionDescriptorDeleter {
  void operator()(cudnnConvolutionStruct* descriptor) const {
    if (descriptor != nullptr) {
      cudnnDestroyConvolutionDescriptor(descriptor);
    }
  }
};

using CudnnHandle = std::unique_ptr<cudnnContext, CudnnHandleDeleter>;
using TensorDescriptor = std::unique_ptr<cudnnTensorStruct, TensorDescriptorDeleter>;
using FilterDescriptor = std::unique_ptr<cudnnFilterStruct, FilterDescriptorDeleter>;
using ConvolutionDescriptor =
    std::unique_ptr<cudnnConvolutionStruct, ConvolutionDescriptorDeleter>;

}  // namespace

ProcessingResult processImagesWithCudnn(const std::vector<float>& input_images,
                                        const Options& options) {
  checkCuda(cudaSetDevice(0), "cudaSetDevice");
  cudaDeviceProp device_properties{};
  checkCuda(cudaGetDeviceProperties(&device_properties, 0),
            "cudaGetDeviceProperties");

  const std::size_t pixels_per_image =
      static_cast<std::size_t>(options.image_width) * options.image_height;
  if (input_images.size() != options.image_count * pixels_per_image) {
    throw std::runtime_error("Input image data does not match the dimensions.");
  }

  ProcessingResult result;
  result.convolution_output_images.resize(input_images.size());
  result.gpu_name = device_properties.name;
  result.cudnn_version = std::to_string(cudnnGetVersion());

  cudnnHandle_t raw_handle = nullptr;
  checkCudnn(cudnnCreate(&raw_handle), "cudnnCreate");
  CudnnHandle handle(raw_handle);

  cudnnFilterDescriptor_t raw_filter_descriptor = nullptr;
  checkCudnn(cudnnCreateFilterDescriptor(&raw_filter_descriptor),
             "cudnnCreateFilterDescriptor");
  FilterDescriptor filter_descriptor(raw_filter_descriptor);
  checkCudnn(cudnnSetFilter4dDescriptor(filter_descriptor.get(), CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW, kChannels, kChannels,
                                        kFilterSize, kFilterSize),
             "cudnnSetFilter4dDescriptor");

  cudnnConvolutionDescriptor_t raw_convolution_descriptor = nullptr;
  checkCudnn(cudnnCreateConvolutionDescriptor(&raw_convolution_descriptor),
             "cudnnCreateConvolutionDescriptor");
  ConvolutionDescriptor convolution_descriptor(raw_convolution_descriptor);
  checkCudnn(cudnnSetConvolution2dDescriptor(
                 convolution_descriptor.get(), 1, 1, 1, 1, 1, 1,
                 CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
             "cudnnSetConvolution2dDescriptor");

  // This normalized 3x3 Gaussian kernel makes the cuDNN convolution an
  // easy-to-inspect denoising image-processing operation.
  const float gaussian_filter[kFilterSize * kFilterSize] = {
      1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
      2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
      1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f};
  float* device_filter = nullptr;
  checkCuda(cudaMalloc(&device_filter, sizeof(gaussian_filter)),
            "cudaMalloc filter");
  checkCuda(cudaMemcpy(device_filter, gaussian_filter, sizeof(gaussian_filter),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy filter");

  const int images_per_stream =
      (options.image_count + options.stream_count - 1) / options.stream_count;
  const std::size_t chunk_bytes =
      static_cast<std::size_t>(images_per_stream) * pixels_per_image * sizeof(float);
  std::vector<cudaStream_t> streams(options.stream_count);
  std::vector<float*> device_inputs(options.stream_count);
  std::vector<float*> device_outputs(options.stream_count);
  std::vector<TensorDescriptor> input_descriptors(options.stream_count);
  std::vector<TensorDescriptor> output_descriptors(options.stream_count);

  for (int stream_index = 0; stream_index < options.stream_count;
       ++stream_index) {
    checkCuda(cudaStreamCreateWithFlags(&streams[stream_index],
                                        cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags");
    checkCuda(cudaMalloc(&device_inputs[stream_index], chunk_bytes),
              "cudaMalloc input images");
    checkCuda(cudaMalloc(&device_outputs[stream_index], chunk_bytes),
              "cudaMalloc output images");

    cudnnTensorDescriptor_t raw_input_descriptor = nullptr;
    cudnnTensorDescriptor_t raw_output_descriptor = nullptr;
    checkCudnn(cudnnCreateTensorDescriptor(&raw_input_descriptor),
               "cudnnCreateTensorDescriptor input");
    checkCudnn(cudnnCreateTensorDescriptor(&raw_output_descriptor),
               "cudnnCreateTensorDescriptor output");
    input_descriptors[stream_index].reset(raw_input_descriptor);
    output_descriptors[stream_index].reset(raw_output_descriptor);
  }

  cudaEvent_t start_event;
  cudaEvent_t end_event;
  checkCuda(cudaEventCreate(&start_event), "cudaEventCreate start");
  checkCuda(cudaEventCreate(&end_event), "cudaEventCreate end");
  checkCuda(cudaEventRecord(start_event), "cudaEventRecord start");

  constexpr float kAlpha = 1.0f;
  constexpr float kBeta = 0.0f;
  for (int stream_index = 0; stream_index < options.stream_count;
       ++stream_index) {
    const int first_image = stream_index * images_per_stream;
    const int batch_size = std::min(images_per_stream,
                                    options.image_count - first_image);
    if (batch_size <= 0) {
      continue;
    }
    const std::size_t transfer_bytes = batch_size * pixels_per_image * sizeof(float);
    checkCudnn(cudnnSetTensor4dDescriptor(
                   input_descriptors[stream_index].get(), CUDNN_TENSOR_NCHW,
                   CUDNN_DATA_FLOAT, batch_size, kChannels, options.image_height,
                   options.image_width),
               "cudnnSetTensor4dDescriptor input");
    checkCudnn(cudnnSetTensor4dDescriptor(
                   output_descriptors[stream_index].get(), CUDNN_TENSOR_NCHW,
                   CUDNN_DATA_FLOAT, batch_size, kChannels, options.image_height,
                   options.image_width),
               "cudnnSetTensor4dDescriptor output");
    checkCuda(cudaMemcpyAsync(
                  device_inputs[stream_index],
                  input_images.data() + first_image * pixels_per_image,
                  transfer_bytes, cudaMemcpyHostToDevice, streams[stream_index]),
              "cudaMemcpyAsync host-to-device");
    checkCudnn(cudnnSetStream(handle.get(), streams[stream_index]),
               "cudnnSetStream");

    // One GPU operation per batch: cuDNN convolution with a Gaussian filter.
    checkCudnn(cudnnConvolutionForward(
                   handle.get(), &kAlpha, input_descriptors[stream_index].get(),
                   device_inputs[stream_index], filter_descriptor.get(),
                   device_filter, convolution_descriptor.get(),
                   CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &kBeta,
                   output_descriptors[stream_index].get(),
                   device_outputs[stream_index]),
               "cudnnConvolutionForward");
    checkCuda(cudaMemcpyAsync(result.convolution_output_images.data() +
                                  first_image * pixels_per_image,
                              device_outputs[stream_index], transfer_bytes,
                              cudaMemcpyDeviceToHost, streams[stream_index]),
              "cudaMemcpyAsync device-to-host");
  }

  for (cudaStream_t stream : streams) {
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }
  checkCuda(cudaEventRecord(end_event), "cudaEventRecord end");
  checkCuda(cudaEventSynchronize(end_event), "cudaEventSynchronize end");
  checkCuda(cudaEventElapsedTime(&result.elapsed_milliseconds, start_event,
                                 end_event),
            "cudaEventElapsedTime");

  for (int stream_index = 0; stream_index < options.stream_count;
       ++stream_index) {
    cudaFree(device_inputs[stream_index]);
    cudaFree(device_outputs[stream_index]);
    cudaStreamDestroy(streams[stream_index]);
  }
  cudaFree(device_filter);
  cudaEventDestroy(start_event);
  cudaEventDestroy(end_event);
  return result;
}
