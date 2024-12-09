#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <algorithm>
#include "neural_network.h"

namespace ANN {
  class Model {
    private:
      std::vector<NeuralNetwork> networks_;

    public:
      Model();
      ~Model(){};

      bool Init(const std::string& filename);

    public:
      double ComputeTranslationalEnergy(const double Ttr, const double Tve);
      double ComputeRotationalEnergy(const double Ttr, const double Tve);
      double ComputeVibrationalEnergy(const double Ttr, const double Tve);
      double ComputeElectronicEnergy(const double Ttr, const double Tve);
  };

  // extern std::vector<NeuralNetwork> networks;
}
