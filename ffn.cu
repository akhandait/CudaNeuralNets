#pragma once

#include <vector>
#include "layers/layer.hpp"
#include "layers/cross_entropy.cu"

/**
 * Implementation of a standard feed forward network.
 */
class FFN
{
 public:
  // Constructor.
  FFN(float lr)
  {
    this->lr = lr;
  }

  // Destructor to release allocated memory.
  ~FFN()
  {
    for (auto layer : network)
    {
      delete layer;
    }
  }

  void Add(Layer* layer)
  {
    this->network.push_back(layer);
  }

  Matrix Forward(Matrix X)
  {
    Matrix Z = X;

    for (int i = 0; i < network.size(); i++)
    {
      Z = network[i]->Forward(Z);
    }

    output = Z;
    return output;
  }

  void Backward(Matrix output, Matrix labels)
  {
    // dOutput.allocateMemoryIfNotAllocated(predictions.shape);
    // float loss = CELoss.Forward(output, labels);
    dOutput = CELoss.Backward(output, labels);

    for (int i = network.size() - 1; i >= 0; i--)
    {
      dOutput = network[i]->Backward(dOutput, this->lr);
    }

  cudaDeviceSynchronize();
  }

  void Train();

  std::vector<Layer*> Network() const {return network;}

 private:
  Matrix output;
  Matrix dOutput;
  CrossEntropy CELoss;

  std::vector<Layer*> network;

  float lr;
};