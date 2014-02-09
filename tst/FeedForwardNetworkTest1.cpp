//------------------------------------------------------------------------------
/*
*  
*  The MIT License (MIT)
* 
*  Copyright (C) 2014 Cody Griffin (cody.m.griffin@gmail.com)
* 
*  Permission is hereby granted, free of charge, to any person obtaining a copy
*  of this software and associated documentation files (the "Software"), to deal
*  in the Software without restriction, including without limitation the rights
*  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*  copies of the Software, and to permit persons to whom the Software is
*  furnished to do so, subject to the following conditions:
* 
*  The above copyright notice and this permission notice shall be included in all
*  copies or substantial portions of the Software.
* 
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*  SOFTWARE.
*/

#include "FeedForwardNetwork.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cstdint>
#include <tuple>
#include <fstream>
#include <chrono>

//------------------------------------------------------------------------------
/*
 * This is not a test yet, per se.  Really just an example.  Perhaps we just put
 * a lower bound on the accepted test error on MNIST (say, 6%).  
 *
 * This is pretty much the "Hello World" of machine learning - MNIST digits using
 * sigmoid neurons and backprop.  
 *
 */

// Some byte swapping (MNIST data is big endian)
#define SWAP_UINT16(x) (((x) >> 8) | ((x) << 8))
#define SWAP_UINT32(x) (((x) >> 24) \
                     | (((x) & 0x00FF0000) >> 8) \
                     | (((x) & 0x0000FF00) << 8) \
                     |  ((x) << 24))

//------------------------------------------------------------------------------
// For some primitive performance analysis
template <typename Resolution = std::chrono::nanoseconds>
struct Stopwatch {
  typedef std::chrono::high_resolution_clock  Clock;
  static uint64_t clock(std::function<void (void)> func) {
    auto start = Clock::now();
    func();
    auto end  = Clock::now();
    return std::chrono::duration_cast<Resolution>(end - start).count();
  }
};

//------------------------------------------------------------------------------
// Helper class for loading data from the MNIST files
struct MnistData {
  typedef std::vector<uint8_t> Image;
  typedef uint8_t              Label;

  MnistData(const std::string& imageFile, const std::string& labelFile) {
    std::ifstream images, labels;
    images.open(imageFile, std::ios::binary);
    labels.open(labelFile, std::ios::binary);

    uint32_t magic, numLabels;
    images.read(reinterpret_cast<char*>(&magic),      sizeof(uint32_t));
    images.read(reinterpret_cast<char*>(&numImages_), sizeof(uint32_t));
    images.read(reinterpret_cast<char*>(&numRows_),   sizeof(uint32_t));
    images.read(reinterpret_cast<char*>(&numCols_),   sizeof(uint32_t));

    labels.read(reinterpret_cast<char*>(&magic),      sizeof(uint32_t));
    labels.read(reinterpret_cast<char*>(&numLabels),  sizeof(uint32_t));

    numImages_ = SWAP_UINT32(numImages_);
    numLabels  = SWAP_UINT32(numLabels);
    numRows_   = SWAP_UINT32(numRows_);
    numCols_   = SWAP_UINT32(numCols_);

    uint8_t             pixel, label;
    Image               image(numRows_ * numCols_);
    imageData_.resize(numImages_);
    for (int c = 0; c < numImages_; c++) { 
      for (int y = 0; y < numRows_; y++) { 
        for (int x = 0; x < numCols_; x++) { 
          images.read(reinterpret_cast<char*>(&pixel), sizeof(uint8_t));
          image[(y*numCols_ + x)] = pixel; 
        }
      }

      labels.read(reinterpret_cast<char*>(&label), sizeof(uint8_t));
      imageData_[c] = std::make_tuple(image,label);
    }
  }
  
  // Do something for each image and label
  void each(std::function<void (const Image&, const Label&)> f) { 
    for (auto image : imageData_) {
      f(std::get<0>(image), std::get<1>(image));
    }
  }
  
  uint32_t                                numRows_;
  uint32_t                                numCols_;
  uint32_t                                numImages_;
private:
  std::vector<std::tuple<Image, Label>>   imageData_;
};

//------------------------------------------------------------------------------
// Some typedefs for convenience
// TODO Layer should be run-time parameterized
// which means that Matrix should be run-time
// parameterized 
typedef rook::Layer<784, 350> InputLayer;
typedef rook::Layer<350,  10> OutputLayer;

// Some helper functions for get MNIST into our net
// Note that dimensions must be the same - this is 
// big mismatch between compile-time and run-time 
// parameterization
InputLayer::Input encodeImage(const MnistData::Image& image) {
  InputLayer::Input input;
  for (int x = 0; x < image.size(); x++) { 
    input.at(x) = image[x]/255.0f; 
  }
  return input;
}

OutputLayer::Output encodeLabel(const MnistData::Label& label) {
  OutputLayer::Output output;
  return output.vapply([&](size_t i) -> float {
    return (i == label)?1.0f:0.0f; 
  }); 
}

MnistData::Label decodeOutput(const OutputLayer::Output& output) {
  int guess = 0;
  for (int i = 0; i < 10; i++) {
    if (output.at(i) > output.at(guess)) {
      guess = i;
    }    
  }
  return guess;
}

//------------------------------------------------------------------------------

int main () {
  // Define a feedfoward network using our layers from above
  // The benefit of compile-time parameterization is clear
  // here: you know immediately if you mismatched layers
  rook::FeedForwardNetwork<
    InputLayer, 
    OutputLayer
  > mnist;

  // Load our MNIST data
  MnistData trainingData("data/train-images-idx3-ubyte",
                         "data/train-labels-idx1-ubyte");
  MnistData     testData("data/t10k-images-idx3-ubyte",
                         "data/t10k-labels-idx1-ubyte");

  // Some helpful temporaries
  unsigned            correct = 0;
  uint8_t             guess;
  InputLayer::Input   digit;
  OutputLayer::Output output;

  //----------------------------------------------------------------------------
  // Training Time
  trainingData.each([&](const MnistData::Image& image, const MnistData::Label& label) {
    digit  = encodeImage(image);
    output = encodeLabel(label);
  
    auto nanos = Stopwatch<std::chrono::microseconds>::clock([&] {
      mnist.learn(digit, output);
    });

    std::cout << "Took " << nanos << "µs.  Learned a " << (unsigned)label << std::endl;
  });

  //----------------------------------------------------------------------------
  // Test Time
  testData.each([&](const MnistData::Image& image, const MnistData::Label& label) {
    digit = encodeImage(image);

    auto nanos = Stopwatch<std::chrono::microseconds>::clock([&] {
      output = mnist(digit);
    });

    guess = decodeOutput(output);
    if (guess == label) correct++;

    std::cout << "Took " << nanos << "µs.  Guessed a " << (unsigned)guess 
              << " (should be " << (unsigned)label << ")" << std::endl;
  });
  
  std::cout << "Number of images: " << testData.numImages_ << std::endl;
  std::cout << "Number correct: "   << correct   << std::endl;
  std::cout << "Test Error: " << std::setprecision(2) << std::fixed 
            << (1.0f - (float)correct/(float)testData.numImages_) * 100.0f << "%" << std::endl;
}

//------------------------------------------------------------------------------
