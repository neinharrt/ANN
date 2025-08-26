#include "neural_network.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace ANN
{

NeuralNetworkAtom::NeuralNetworkAtom()
{
  num_input_ = 1;
  x_log_     = false;
  y_log_     = true;
  Ao_        = nullptr;
  bo_        = nullptr;
  Ai_        = nullptr;
  bi_        = nullptr;
}

bool NeuralNetworkAtom::Init(const int &mode_index, const std::string &species_name)
{
  std::string mode_name = "E";

  const std::string filename = "./model/" + species_name + "/" + species_name + mode_name + ".dat";

  std::ifstream fin(filename);
  if (fin.is_open() == false) return false;

  fin >> num_hidden_ >> num_parameter_;
  fin >> x_log_ >> y_log_;

  weights_.resize(num_parameter_);
  Ao_ = &weights_[0];
  bo_ = &weights_[num_hidden_];
  Ai_ = &weights_[num_hidden_ + 1];
  bi_ = &weights_[num_hidden_ * (num_input_ + 1) + 1];

  for (int i = 0; i < num_parameter_; i++) fin >> weights_[i];
  fin.close();
  return true;
}
void NeuralNetworkAtom::Pred(const double x, double *f) const
{
  *f = bo_[0];
  for (int i = 0; i < num_hidden_; i++)
  {
    const double yi = Ai_[i] * x + bi_[i];
    *f += (Ao_[i] * Transfer(yi));
  }
}
void NeuralNetworkAtom::Derivative(const double x, double *dfdx) const
{
  dfdx[0] = dfdx[1] = dfdx[2] = dfdx[3] = 0.0;
  for (int i = 0; i < num_hidden_; i++)
  {
    const double temp = DiffTransfer(Ai_[i] * x + bi_[i]) * Ao_[i];
    dfdx[3] += (temp * Ai_[i]);
  }
}

NeuralNetworkDiatomic::NeuralNetworkDiatomic()
{
  num_input_ = 3;
  x1_log_    = false;
  x2_log_    = false;
  x3_log_    = false;
  y_log_     = true;
  Ao_        = nullptr;
  bo_        = nullptr;
  Ai_        = nullptr;
  bi_        = nullptr;
}
bool NeuralNetworkDiatomic::Init(const int &mode_index, const std::string &species_name)
{
  std::string mode_name;
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

  const std::string filename = "./model/" + species_name + "/" + species_name + mode_name + ".dat";

  std::ifstream fin(filename);
  if (fin.is_open() == false) return false;

  fin >> num_hidden_ >> num_parameter_;
  fin >> x1_log_ >> x2_log_ >> x3_log_ >> y_log_;

  weights_.resize(num_parameter_);
  Ao_ = &weights_[0];
  bo_ = &weights_[num_hidden_];
  Ai_ = &weights_[num_hidden_ + 1];
  bi_ = &weights_[num_hidden_ * (num_input_ + 1) + 1];

  for (int i = 0; i < num_parameter_; i++) fin >> weights_[i];
  fin.close();
  return true;
}
void NeuralNetworkDiatomic::Pred(const double *x, double *f) const
{
  *f = bo_[0];
  for (int i = 0; i < num_hidden_; i++)
  {
    const double yi = Ai_[i * 3 + 0] * x[0] + Ai_[i * 3 + 1] * x[1] + Ai_[i * 3 + 2] * x[2] + bi_[i];
    *f += (Ao_[i] * Transfer(yi));
  }
}
void NeuralNetworkDiatomic::Derivative(const double *x, double *dfdx) const
{
  dfdx[0] = dfdx[1] = dfdx[2] = dfdx[3] = 0.0;
  for (int i = 0; i < num_hidden_; i++)
  {
    const double temp =
        DiffTransfer(Ai_[i * 3 + 0] * x[0] + Ai_[i * 3 + 1] * x[1] + Ai_[i * 3 + 2] * x[2] + bi_[i]) * Ao_[i];
    dfdx[1] += (temp * (Ai_[i * 3 + 0]));
    dfdx[2] += (temp * (Ai_[i * 3 + 1]));
    dfdx[3] += (temp * (Ai_[i * 3 + 2]));
  }
}

inline double NeuralNetwork::Transfer(const double input) const { return 2.0 / (1.0 + std::exp(-2.0 * input)) - 1.0; }
inline double NeuralNetwork::DiffTransfer(const double input) const
{
  const double s = Transfer(input);
  return 1.0 - s * s;
}
}  // namespace ANN