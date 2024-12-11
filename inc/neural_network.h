#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace ANN
{
class NeuralNetwork
{
  private:
    const int &m;                  // reserved for num_hidden_
    int        num_hidden_;        // m
    int        num_parameter_;     // 4*m+1

    std::vector<double> weights_;  // (VECTOR) size: 4*m+1 --> | bo_ | Ao_ | bi_ | Ai_ |

    bool x1_log_;
    bool x2_log_;
    bool y_log_;

    char *species_;

    double *Ao_;  // (VECTOR) size: 1 * m
    double *bo_;  // (SCALAR) size: 1 * 1
    double *Ai_;  // (MATRIX) size: m * 2
    double *bi_;  // (VECTOR) size: m * 1

    // f = Ao*xo + bo
    // xo = Transfer(yi)
    // yi = Ai*x + bi

  public:
    NeuralNetwork();
    ~NeuralNetwork() {};

    bool Init(const int &molecule_flag, const int &mode_index, const std::string &species_name);
    void Pred(const double *x, double *f) const;
    void Derivative(const double *x, double *dfdx) const;

  private:
    inline double Transfer(const double input) const;
    inline double DiffTransfer(const double input) const;
};

const double erg2J = 1.0E-04;                    // Convert erg/g to J/kg

const double boltz = 1.380649E-23;               // Boltzmann constant (J/K)

extern std::vector<std::vector<double>> thetv;   // Vibrational characteristic temperature (K)

extern std::vector<std::vector<double>> ge;      // Electronic multiplicity of ground state

extern std::vector<std::vector<double>> thetel;  // Electronic characteristic temperature (K)
}  // namespace ANN