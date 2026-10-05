#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cudnn_image_processor.h"
#include "image_data.h"
#include "options.h"

namespace {

constexpr char kOutputDirectory[] = "data/results";

std::vector<std::uint8_t> convertToPgmPixels(const std::vector<float>& image) {
  std::vector<std::uint8_t> pixels(image.size());
  for (std::size_t index = 0; index < image.size(); ++index) {
    const float clamped = std::max(0.0f, std::min(1.0f, image[index]));
    pixels[index] = static_cast<std::uint8_t>(clamped * 255.0f + 0.5f);
  }
  return pixels;
}

void writePgmImage(const std::string& filename,
                   const std::vector<std::uint8_t>& pixels, int image_width,
                   int image_height) {
  std::ofstream image_file(filename, std::ios::binary);
  if (!image_file) {
    throw std::runtime_error("Could not open image output file.");
  }
  image_file << "P5\n" << image_width << ' ' << image_height << "\n255\n";
  image_file.write(reinterpret_cast<const char*>(pixels.data()),
                   static_cast<std::streamsize>(pixels.size()));
}

void writeResults(const std::vector<float>& input_images,
                  const ProcessingResult& result, const Options& options) {
  std::filesystem::create_directories(kOutputDirectory);
  const std::size_t pixels_per_image =
      static_cast<std::size_t>(options.image_width) * options.image_height;
  std::vector<float> input_image(input_images.begin(),
                                 input_images.begin() + pixels_per_image);
  std::vector<float> convolution_result(
      result.convolution_output_images.begin(),
      result.convolution_output_images.begin() + pixels_per_image);
  writePgmImage(std::string(kOutputDirectory) + "/input_0000.pgm",
                convertToPgmPixels(input_image), options.image_width,
                options.image_height);
  writePgmImage(std::string(kOutputDirectory) +
                    "/cudnn_convolution_result_0000.pgm",
                convertToPgmPixels(convolution_result), options.image_width,
                options.image_height);
}

}  // namespace

int main(int argument_count, char** arguments) {
  try {
    const Options options = parseOptions(argument_count, arguments);
    const std::vector<float> input_images = generateImages(
        options.image_count, options.image_width, options.image_height);
    const ProcessingResult result = processImagesWithCudnn(input_images, options);
    writeResults(input_images, result, options);

    std::cout << std::fixed << std::setprecision(3)
              << "GPU: " << result.gpu_name << "\n"
              << "cuDNN version: " << result.cudnn_version << "\n"
              << "Dataset: " << options.image_count << " grayscale images x "
              << options.image_width << " x " << options.image_height << " pixels\n"
              << "CUDA streams: " << options.stream_count << "\n"
              << "GPU operation: cuDNN 3x3 Gaussian convolution\n"
              << "GPU pipeline time: " << result.elapsed_milliseconds << " ms\n"
              << "Wrote data/results/input_0000.pgm and "
                 "data/results/cudnn_convolution_result_0000.pgm\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
