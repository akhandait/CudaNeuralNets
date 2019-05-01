#include "ffn.cu"
#include "layers/linear.cu"
#include "layers/layer.hpp"
#include "layers/relu.cu"
#include "layers/softmax.cu"
#include "layers/cross_entropy.cu"
#include "dataset.cu"

int main()
{
  Dataset dataset(64, 64 * 200);

  FFN network(0.01);
  network.Add(new Linear(2, 40));
  network.Add(new ReLU());
  network.Add(new Linear(40, 40));
  network.Add(new ReLU());
  network.Add(new Linear(40, 4));
  network.Add(new Softmax());

  CrossEntropy crossEntropy;
  for (int epoch = 0; epoch < 10; epoch++)
  {
    float loss = 0;
    Matrix output;
    for (int batch = 0; batch < 200; batch++)
    {
      output = network.Forward(dataset.DataBatches().at(batch));
      network.Backward(output, dataset.LabelBatches().at(batch));
      loss += crossEntropy.Forward(output, dataset.LabelBatches().at(batch));
    }

    std::cout << "Epoch: " << epoch << std::endl;
    std::cout << "Loss: " << loss / 200 << std::endl;
  }

  // Matrix inp(784, 4);
  // for (int i = 0; i < inp.nRows; i++)
  // {
  //   for (int j = 0; j < inp.nCols; j++)
  //   {
  //       inp(i, j) = 1;
  //   }
  // }
  // inp.CopyHostToDevice();

  // FFN net(0.01);

  // net.Add(new Linear(784, 500));
  // net.Add(new ReLU());
  // net.Add(new Linear(500, 200));
  // net.Add(new ReLU());
  // net.Add(new Linear(200, 10));
  // net.Add(new Softmax());

  // Matrix out;
  // out = net.Forward(inp);
  // out.CopyDeviceToHost();

  // for (int i = 0; i < out.nRows; i++)
  // {
  //   for (int j = 0; j < out.nCols; j++)
  //   {
  //     std::cout << out(i, j) << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // Matrix labels(1, 4);
  // labels(0, 0) = 6;
  // labels(0, 1) = 2;
  // labels(0, 2) = 9;
  // labels(0, 3) = 4;
  // labels.CopyHostToDevice();
  // net.Backward(out, labels);
}
