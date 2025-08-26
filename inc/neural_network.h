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
  protected:

    int num_input_;      // k
    int num_hidden_;     // m
    int num_parameter_;  // m * (k + 2) + 1

    std::vector<double> weights_;  // (VECTOR) size: m * (k + 2) + 1 --> | bo_ | Ao_ | bi_ | Ai_ |

    bool y_log_;

    char *species_;

    double *Ao_;  // (VECTOR) size: 1 * m
    double *bo_;  // (SCALAR) size: 1 * 1
    double *Ai_;  // (MATRIX) size: m * k
    double *bi_;  // (VECTOR) size: m * 1

    // f = Ao*xo + bo
    // xo = Transfer(yi)
    // yi = Ai*x + bi

  public:

    NeuralNetwork();
    ~NeuralNetwork() {};

    virtual bool Init(const int &mode_index, const std::string &species_name);
    virtual void Pred(const double x, double *f) const;
    virtual void Pred(const double *x, double *f) const;
    virtual void Derivative(const double x, double *dfdx) const;
    virtual void Derivative(const double *x, double *dfdx) const;

  protected:

    inline double Transfer(const double input) const;
    inline double DiffTransfer(const double input) const;
};

class NeuralNetworkAtom : public NeuralNetwork
{
  private:

    bool x_log_;

  public:

    NeuralNetworkAtom();
    ~NeuralNetworkAtom() {};

    bool Init(const int &mode_index, const std::string &species_name);
    void Pred(const double x, double *f) const;
    void Derivative(const double x, double *dfdx) const;
};

class NeuralNetworkDiatomic : public NeuralNetwork
{
  private:

    bool x1_log_;
    bool x2_log_;
    bool x3_log_;

  public:

    NeuralNetworkDiatomic();
    ~NeuralNetworkDiatomic() {};

    bool Init(const int &mode_index, const std::string &species_name);
    void Pred(const double *x, double *f) const;
    void Derivative(const double *x, double *dfdx) const;
};

const double erg2J = 1.0E-04;  // Convert erg/g to J/kg

const double boltz = 1.380649E-23;  // Boltzmann constant (J/K)

extern std::vector<std::vector<double>> thetv;  // Vibrational characteristic temperature (K)

extern std::vector<std::vector<double>> ge;  // Electronic multiplicity of ground state

extern std::vector<std::vector<double>> thetel;  // Electronic characteristic temperature (K)

}  // namespace ANN