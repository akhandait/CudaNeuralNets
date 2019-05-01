#include "matrix.cu"
// #include <iostream>
// #include "linear.cu"
// #include "relu.cu"
#include "dataset.cu"
#include "layers/cross_entropy.cu"

using namespace std;

int main()
{
  // Dataset d(64, 64 * 10);
  // Matrix a = d.LabelBatches().at(5);
  // for (int i = 0; i < a.nRows; i++)
  // {
  //   for (int j = 0; j < a.nCols; j++)
  //   {
  //     cout << a(i, j) << " ";
  //   }
  //   cout << endl;
  // }
  CrossEntropy ce;
  Matrix a(3, 3);
  a(0, 0) = 0.3;
  a(1, 0) = 0.25;
  a(2, 0) = 0.45;
  a(0, 1) = 0.3;
  a(1, 1) = 0.35;
  a(2, 1) = 0.35;
  a(0, 2) = 0.2;
  a(1, 2) = 0.4;
  a(2, 2) = 0.4;

  Matrix labels(1, 3);
  labels(0, 0) = 2;
  labels(0, 1) = 1;
  labels(0, 2) = 0;
  a.CopyHostToDevice();labels.CopyHostToDevice();

  float f = ce.Forward(a, labels);
  Matrix g = ce.Backward(a, labels);g.CopyHostToDevice();
  std::cout << f << std::endl;
  for (int i = 0; i < g.nRows; i++)
  {
    for (int j = 0; j < g.nCols; j++)
    {
      cout << g(i, j) << " ";
    }
    cout << endl;
  }
}
