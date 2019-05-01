#pragma once
#include "../matrix.cu"

__global__ void ForwardCrossEntropy(float *output, float *labels,
    int nColsOutput, float *loss)
{
  int col = blockIdx.x;

  atomicAdd(loss, -logf(output[(int)labels[col] * nColsOutput + col]));
}

__global__ void BackwardCrossEntropy(float *output, float *labels,
    int nColsOutput, float *dOutput)
{
  int row = threadIdx.x;
  int col = blockIdx.x;

  if (row == labels[col])
    dOutput[row * nColsOutput + col] = -1 / (output[row * nColsOutput + col]);
  else
    dOutput[row * nColsOutput + col] = 0.0;
}

class CrossEntropy
{
 public:
  CrossEntropy()
  {
    /* Nothing to do here. */
  }

  ~CrossEntropy()
  {
    /* Nothing to do here. */
  }

  float Forward(Matrix output, Matrix labels)
  {
    if (output.nCols != labels.nCols)
    {
      std::cerr << "ERROR: Number of columns in the output matrix should " <<
          "be equal to the number of colmns of the labels matrix." << std::endl;
    }

    float* loss;
    CheckErrors(cudaMallocManaged(&loss, sizeof(float)),
        "CrossEntropy::Forward() cudaMalloc : loss");
    *loss = 0.0f;

    ForwardCrossEntropy<<<output.nCols, 1>>>(output.deviceMat.get(),
        labels.deviceMat.get(), output.nCols, loss);

    cudaDeviceSynchronize();
    // https://stackoverflow.com/questions/19193468/why-do-we-need-
    // cudadevicesynchronize-in-kernels-with-device-printf
    CheckErrors(cudaGetLastError(),
        "CrossEntropy:: Kernel invocation: ForwardCrossEntropy");

    lossReturn = *loss;
    CheckErrors(cudaFree(loss), "CrossEntropy::Forward() Cuda free : loss");

    return lossReturn / output.nCols;
  }

  Matrix& Backward(Matrix output, Matrix labels)
  {
    if (output.nCols != labels.nCols)
    {
      std::cerr << "ERROR: Number of columns in the output matrix should " <<
          "be equal to the number of colmns of the labels matrix." << std::endl;
    }

    dOutput.AllocateMemory(output.nRows, output.nCols);

    BackwardCrossEntropy<<<output.nCols, output.nRows>>>(output.deviceMat.get(),
        labels.deviceMat.get(), output.nCols, dOutput.deviceMat.get());
    CheckErrors(cudaGetLastError(),
        "CrossEntropy:: Kernel invocation: BackwardCrossEntropy");dOutput.CopyDeviceToHost();

    return dOutput;
  }

 private:
  Matrix dOutput;

  float lossReturn;
};
