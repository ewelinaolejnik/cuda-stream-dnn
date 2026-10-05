#include "options.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr int kMinimumImageDimension = 3;

}  // namespace

Options parseOptions(int argument_count, char** arguments) {
  Options options;
  for (int argument_index = 1; argument_index < argument_count;
       ++argument_index) {
    const std::string argument = arguments[argument_index];
    if (argument == "--help") {
      std::cout << "Usage: cuda_stream_dnn [--images N] [--width N] "
                   "[--height N] [--streams N]\n";
      std::exit(0);
    }
    if (++argument_index >= argument_count) {
      throw std::runtime_error("Missing value after " + argument);
    }
    if (argument == "--images") {
      options.image_count = std::stoi(arguments[argument_index]);
    } else if (argument == "--width") {
      options.image_width = std::stoi(arguments[argument_index]);
    } else if (argument == "--height") {
      options.image_height = std::stoi(arguments[argument_index]);
    } else if (argument == "--streams") {
      options.stream_count = std::stoi(arguments[argument_index]);
    } else {
      throw std::runtime_error("Unknown option: " + argument);
    }
  }
  if (options.image_count < 1 ||
      options.image_width < kMinimumImageDimension ||
      options.image_height < kMinimumImageDimension || options.stream_count < 1) {
    throw std::runtime_error("Invalid image dimensions or stream count.");
  }
  return options;
}
