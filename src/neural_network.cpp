#include "neural_network.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace ANN
{
NeuralNetwork::NeuralNetwork()
    : m(num_hidden_),
      num_hidden_(0),
      num_parameter_(0),
      weights_(),
      x1_log_(false),
      x2_log_(false),
      y_log_(true),
      Ao_(nullptr),
      bo_(nullptr),
      Ai_(nullptr),
      bi_(nullptr) {}

bool NeuralNetwork::Init(const int &molecule_flag, const int &mode_index, const std::string &species_name)
{
  std::string mode_name;
  switch (molecule_flag)
  {
    case 0:  // Atoms
      mode_name = "E";
      break;

    case 1:  // Diatomic molecules
      switch (mode_index)
      {
        case 0:
          mode_name = "R";
          break;
        case 1:
          mode_name = "V";
          break;
        case 2:
          mode_name = "E";
          break;
      }

    default:
      break;
  }

  const std::string filename = "./model/" + species_name + "/" + species_name + mode_name + ".dat";

  std::ifstream fin(filename);
  if (fin.is_open() == false) return false;

  fin >> num_hidden_ >> num_parameter_;
  fin >> x1_log_ >> x2_log_ >> y_log_;

  weights_.resize(num_parameter_);
  Ao_ = &weights_[0];
  bo_ = &weights_[m];
  Ai_ = &weights_[m + 1];
  bi_ = &weights_[3 * m + 1];

  for (int i = 0; i < num_parameter_; i++) fin >> weights_[i];
  fin.close();
  return true;
}
void NeuralNetwork::Pred(const double *x, double *f) const
{
  *f = bo_[0];
  for (int i = 0; i < m; i++)
  {
    const double yi = Ai_[i * 2] * x[0] + Ai_[i * 2 + 1] * x[1] + bi_[i];
    *f += (Ao_[i] * Transfer(yi));
  }
}
void NeuralNetwork::Derivative(const double *x, double *dfdx) const
{
  dfdx[0] = dfdx[1] = 0.0;
  for (int i = 0; i < m; i++)
  {
    const double temp = DiffTransfer(Ai_[i * 2] * x[0] + Ai_[i * 2 + 1] * x[1] + bi_[i]) * Ao_[i];
    dfdx[0] += (temp * Ai_[i * 2]);
    dfdx[1] += (temp * Ai_[i * 2 + 1]);
  }
}
inline double NeuralNetwork::Transfer(const double input) const
{
  return 2.0 / (1.0 + std::exp(-2.0 * input)) - 1.0;
}
inline double NeuralNetwork::DiffTransfer(const double input) const
{
  const double s = Transfer(input);
  return 1.0 - s * s;
}
}  // namespace ANN