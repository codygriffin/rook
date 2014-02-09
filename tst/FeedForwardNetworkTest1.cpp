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
#include <array>
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

int main () {
  srand (time(NULL));
  
  typedef rook::Layer<784, 350> InputLayer;
  typedef rook::Layer<350,  10> OutputLayer;
  rook::FeedForwardNetwork<
    InputLayer, 
    OutputLayer
  > net;

  std::ifstream images;
  std::ifstream labels;
  images.open("data/train-images-idx3-ubyte", std::ios::binary);
  labels.open("data/train-labels-idx1-ubyte", std::ios::binary);

  uint32_t magic, numImages, numLabels, numRows, numCols;
  images.read(reinterpret_cast<char*>(&magic),     sizeof(uint32_t));
  images.read(reinterpret_cast<char*>(&numImages), sizeof(uint32_t));
  images.read(reinterpret_cast<char*>(&numRows),   sizeof(uint32_t));
  images.read(reinterpret_cast<char*>(&numCols),   sizeof(uint32_t));

  labels.read(reinterpret_cast<char*>(&magic),     sizeof(uint32_t));
  labels.read(reinterpret_cast<char*>(&numLabels), sizeof(uint32_t));

  numImages = SWAP_UINT32(numImages);
  numRows   = SWAP_UINT32(numRows);
  numCols   = SWAP_UINT32(numCols);

  uint8_t             pixel;
  uint8_t             label;
  InputLayer::Input   digit;
  OutputLayer::Output output;
  for (int c = 0; c < numImages; c++) { 
    for (int y = 0; y < numRows; y++) { 
      for (int x = 0; x < numCols; x++) { 
        images.read(reinterpret_cast<char*>(&pixel), sizeof(uint8_t));
        digit.at(y*28 + x) = pixel/255.0f; 
      }
    }
    labels.read(reinterpret_cast<char*>(&label), sizeof(uint8_t));

    output = output.vapply([&](size_t i) -> float {
      return (i == label)?1.0f:0.0f; 
    }); 

    auto nanos = Stopwatch<std::chrono::microseconds>::clock([&] {
      net.learn(digit, output);
    });

    std::cout << "Took " << nanos << "µs.  Learned a " << (unsigned)label << std::endl;
  }

  images.close();
  labels.close();
  images.open("data/t10k-images-idx3-ubyte", std::ios::binary);
  labels.open("data/t10k-labels-idx1-ubyte", std::ios::binary);

  images.read(reinterpret_cast<char*>(&magic),     sizeof(uint32_t));
  images.read(reinterpret_cast<char*>(&numImages), sizeof(uint32_t));
  images.read(reinterpret_cast<char*>(&numRows),   sizeof(uint32_t));
  images.read(reinterpret_cast<char*>(&numCols),   sizeof(uint32_t));

  labels.read(reinterpret_cast<char*>(&magic),     sizeof(uint32_t));
  labels.read(reinterpret_cast<char*>(&numLabels), sizeof(uint32_t));

  numImages = SWAP_UINT32(numImages);
  numRows   = SWAP_UINT32(numRows);
  numCols   = SWAP_UINT32(numCols);

  unsigned correct = 0;
  for (int c = 0; c < numImages; c++) { 
    for (int y = 0; y < numRows; y++) { 
      for (int x = 0; x < numCols; x++) { 
        images.read(reinterpret_cast<char*>(&pixel), sizeof(uint8_t));
        digit.at(y*28 + x) = pixel/255.0f; 
      }
    }
    labels.read(reinterpret_cast<char*>(&label), sizeof(uint8_t));

    auto nanos = Stopwatch<std::chrono::microseconds>::clock([&] {
      output = net.infer(digit);
    });

    int guess = 0;
    for (int i = 0; i < 10; i++) {
      if (output.at(i) > output.at(guess)) {
        guess = i;
      }    
    }
    if (guess == label) correct++;

    std::cout << "Took " << nanos << "µs.  Guessed a " << (unsigned)guess 
              << " (should be " << (unsigned)label << ")" << std::endl;
  }
  
  std::cout << "Number of images: " << numImages << std::endl;
  std::cout << "Number correct: "   << correct   << std::endl;
  std::cout << "Test Error: " << std::setprecision(2) << std::fixed 
            << (1.0f - (float)correct/(float)numImages) * 100.0f << "%" << std::endl;
}

//------------------------------------------------------------------------------
