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

#ifndef INCLUDED_AUTOENCODER_H
#define INCLUDED_AUTOENCODER_H

#ifndef INCLUDED_LAYER_H
#include "Layer.h"
#define INCLUDED_LAYER_H
#endif

//------------------------------------------------------------------------------

namespace rook { 

//------------------------------------------------------------------------------

template <size_t X, size_t Y>
struct Autoencoder {
  // Use regular feed-forward layers
  typedef rook::Layer<Y, X, Sigmoid>   Decoder;
  typedef rook::Layer<X, Y, Sigmoid>   Encoder;

  typedef typename Encoder::Input  Input;
  typedef typename Encoder::Output Code;

  // Encoding is just a forward pass through the encoder layer
  // (constrained to be the transpose of the decoding layer)
  Code
  encode(Input const& input) const {
    return encoder.infer(input);
  }

  // Decoding is just a forward pass through the decoder layer
  Input
  decode(Code const& code) const {
    return decoder.infer(code);
  }

  // We can reconstruct an input by doing an encode and a 
  // decode
  Input
  reconstruct(Input  const& input) {
    return decode(encode(input));
  }

  // For learning, we constrain the encoder and decoder to share
  // weights (but not biases)
  // TODO only learn decoder and encoder's bias - better sharing
  // no need for copy or transpose
  Input
  learn(Input  const& input, float learningRate = 0.1f) {
    // Reconstruct the input
    auto corrupted = input.apply([](float a) -> float {
      return (rand()%100<60) ? a : 0.0f;
    });
    auto code   = encode(corrupted);
    auto recon  = decode(code);

    // Update the weights and biases
    auto decoderError  = decoder.learn(code, recon, input, learningRate);
    auto encoderError  = encoder.learn(input, code, std::get<0>(decoderError), learningRate);

    // Tie our weights (but not biases)
    encoder.getWeightMatrix() = decoder.getWeightMatrix().transpose();

    return std::get<1>(decoderError);
  }

  typename Decoder::WeightMatrix&
  getWeightMatrix() {
    return decoder.getWeightMatrix();
  }

  Decoder  decoder;
  Encoder  encoder;
};

//------------------------------------------------------------------------------

} // namespace rook

//------------------------------------------------------------------------------

#endif
