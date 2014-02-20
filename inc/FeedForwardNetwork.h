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

#ifndef INCLUDED_FEEDFORWARDNETWORK_H
#define INCLUDED_FEEDFORWARDNETWORK_H

#ifndef INCLUDED_LAYER_H
#include "Layer.h"
#endif

#ifndef INCLUDED_MEMORY
#include <memory>
#define INCLUDED_MEMORY
#endif

//------------------------------------------------------------------------------

namespace rook { 

//------------------------------------------------------------------------------

template <typename InputLayer, typename...HiddenLayers>
struct FeedForwardNetwork;

template <typename InputLayer, typename...HiddenLayers>
struct FeedForwardNetwork {
  // Recursively find the total output size of the network
  typedef typename FeedForwardNetwork<HiddenLayers...>::Output Output;
  typedef typename InputLayer::Input                           Input;

  // Initialize network (default)
  FeedForwardNetwork()
  : pHiddenLayers_(new FeedForwardNetwork<HiddenLayers...>()) {
  }

  // Do a forward pass through the net 
  Output
  infer(const Input& input) const {
    // Update this layer, then update the hidden layers
    auto   next = inputLayer_.infer(input);
    return pHiddenLayers_->infer(next);
  }

  std::tuple<Input, Output>
  learn(const Input& input, const Output& target, float learningRate = 0.1f) {
    // Calculate the output of this layer
    const auto next   = inputLayer_.infer(input);
    const auto herror = pHiddenLayers_->learn(next, target);

    // This is turd
    const auto error  = inputLayer_.correct(input, next, std::get<0>(herror));
    return std::make_tuple(std::get<0>(error), std::get<1>(herror));
  }

  InputLayer& getLayer() {
    return inputLayer_;
  }

  FeedForwardNetwork<HiddenLayers...>& getRemainNetwork() {
    return *pHiddenLayers_;
  }

private:
  std::unique_ptr<FeedForwardNetwork<HiddenLayers...>> pHiddenLayers_;
  InputLayer                                           inputLayer_;
};

template <typename OutputLayer>
struct FeedForwardNetwork<OutputLayer> {
  // Terminal case - this must be the output layer
  typedef typename OutputLayer::Output Output;
  typedef typename OutputLayer::Input  Input;

  Output
  infer(const Input& input) const {
    return outputLayer_.infer(input);
  }

  std::tuple<Input, Output>
  learn(const Input& input, const Output& target, float learningRate = 0.1f) {
    // The input to the final layer is input, 
    // and our overall target is target
    // we generate a predicted output
    const auto prediction = infer(input);

    // Update our output weights and get our errors
    const auto error      = outputLayer_.learn(input, prediction, target);

    // Back propagate our error
    return error;
  }

  OutputLayer& getLayer() { 
    return outputLayer_;
  }

private:
  OutputLayer outputLayer_;
};

//------------------------------------------------------------------------------

} // namespace rook

//------------------------------------------------------------------------------

#endif
