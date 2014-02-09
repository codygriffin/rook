# rook
###### Simple C++ Machine Learning

The only thing this does now is a simple 2-layer feed-forward network for 
training MNIST.  With a single pass through the training data, this can
achieve around a 4.0% test error.  

Because this was a learning exercise for me, the code makes some heavy use
of templates.  This was great for compile-time error checking.  

For now, there is a single "test" file in tst/FeedForwardNetworkTest1.cpp. 
This has all of the MNIST loading and uses the included Matrix, Layer and
FeedFowardNetwork classes to learn some digits.  

## Goals

* No dependencies
* Readability over performance
* Simple

## Build

Pretty standard fare.  Requires a recent version of g++ or clang (C++11 support),
along with GNU Make and curl (for retrieving MNIST).

```
git clone https://github.com/codygriffin/rook.git
cd rook
make
```

## Future Plans
Autoencoders, regularization options, RBMs.  
I also want to get away from compile-time parameterization.

