# Parallelizing Neural Networks

## Development Environment

C++11

CUDA 9.0

## Usage

Compile and run the train.cu file:

```
nvcc --std=c++11 train.cu -o train
./train
```

To execute only on the CPU, change the last argument of the `Train()` function in train.cu to `true`.

