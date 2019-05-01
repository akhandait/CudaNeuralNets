#pragma once

#include "matrix.cu"

#include <vector>

class Dataset
{
 public:
  Dataset(int batchSize, int nPoints)
  {
    this->batchSize = batchSize;
    int nBatches = nPoints / batchSize;

    for (int i = 0; i < nBatches; i++)
    {
      dataBatches.push_back(Matrix(2, batchSize));
      labelBatches.push_back(Matrix(1, batchSize));

      for (int j = 0; j < batchSize; j++)
      {
        dataBatches[i][j] = (static_cast<float>(rand()) / RAND_MAX - 0.5) * 2;
        dataBatches[i][batchSize + j] = (static_cast<float>(rand()) / RAND_MAX - 0.5) * 2;

        if (dataBatches[i][j] >= 0 && dataBatches[i][batchSize + j] >= 0)
          labelBatches[i][j] = 0;
        else if (dataBatches[i][j] < 0 && dataBatches[i][batchSize + j] >= 0)
          labelBatches[i][j] = 1;
        else if (dataBatches[i][j] < 0 && dataBatches[i][batchSize + j] < 0)
          labelBatches[i][j] = 2;
        else
          labelBatches[i][j] = 3;
      }

      dataBatches[i].CopyHostToDevice();
      labelBatches[i].CopyHostToDevice();
    }
  }

  std::vector<Matrix>& DataBatches()
  {
    return dataBatches;
  }

  std::vector<Matrix>& LabelBatches()
  {
    return labelBatches;
  }

 private:
  int batchSize;

  std::vector<Matrix> dataBatches;
  std::vector<Matrix> labelBatches;
};
