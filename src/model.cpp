#include "model.h"

#include <iostream>

namespace ANN
{
Model::Model() {}

bool Model::Init(const char *species_name)
{
  bool flag = false;
  for (int i = 0; i < 37; i++)
  {
    if (std::string(species_name).compare(species_pack_[i]) == 0)
    {
      species_name_   = std::string(species_name);
      species_weight_ = weight_pack_[i];
      molecule_flag_  = flag_pack_[i];
      species_index_  = i;
      flag            = true;
      break;
    }
  }
  switch (molecule_flag_)
  {
    case 0:  // Atoms
    {
      ComputeTE = &Model::ComputeTEAtom;
      ComputeRE = &Model::ComputeREAtom;
      ComputeVE = &Model::ComputeVEAtom;
      ComputeEE = &Model::ComputeEEAtom;
      networks_.resize(1);
      for (int i = 0; i < 1; i++)
      {
        networks_[i] = std::make_shared<NeuralNetwork>();
        networks_[i]->Init(molecule_flag_, i, species_name_);
      }
      break;
    }
    case 1:  // Diatomic molecules
      ComputeTE = &Model::ComputeTEDiatomic;
      ComputeRE = &Model::ComputeREDiatomic;
      ComputeVE = &Model::ComputeVEDiatomic;
      ComputeEE = &Model::ComputeEEDiatomic;
      networks_.resize(3);
      for (int i = 0; i < 3; i++)
      {
        networks_[i] = std::make_shared<NeuralNetwork>();
        networks_[i]->Init(molecule_flag_, i, species_name_);
      }
      break;
    case 2:  // Polyatomic molecules
      ComputeTE = &Model::ComputeTEPolyatomic;
      ComputeRE = &Model::ComputeREPolyatomic;
      ComputeVE = &Model::ComputeVEPolyatomic;
      ComputeEE = &Model::ComputeEEPolyatomic;
      networks_.resize(0);
      break;
    case 3:  // Electron
      ComputeTE = &Model::ComputeTEElectron;
      ComputeRE = &Model::ComputeREElectron;
      ComputeVE = &Model::ComputeVEElectron;
      ComputeEE = &Model::ComputeEEElectron;
      networks_.resize(0);
      break;
    default:  // Other species (error)
      break;
  }
  return flag;
}

double Model::ComputeTEAtom(const double &Ttr, const double &Tve)
{
  return 1.5 * R_ * Ttr / species_weight_;
}
double Model::ComputeREAtom(const double &Ttr, const double &Tve)
{
  return 0.0;
}
double Model::ComputeVEAtom(const double &Ttr, const double &Tve)
{
  return 0.0;
}
double Model::ComputeEEAtom(const double &Ttr, const double &Tve)
{
  const double x[2] = {Ttr, Tve};
  double       lnE;
  networks_[0]->Pred(x, &lnE);
  return erg2J * exp(lnE);
}

double Model::ComputeTEDiatomic(const double &Ttr, const double &Tve)
{
  return 1.5 * R_ * Ttr / species_weight_;
}
double Model::ComputeREDiatomic(const double &Ttr, const double &Tve)
{
  const double x[2] = {Ttr, Tve};
  double       lnE;
  networks_[0]->Pred(x, &lnE);
  return erg2J * exp(lnE);
}
double Model::ComputeVEDiatomic(const double &Ttr, const double &Tve)
{
  const double x[2] = {Ttr, Tve};
  double       lnE;
  networks_[1]->Pred(x, &lnE);
  return erg2J * exp(lnE);
}
double Model::ComputeEEDiatomic(const double &Ttr, const double &Tve)
{
  const double x[2] = {Ttr, Tve};
  double       lnE;
  networks_[2]->Pred(x, &lnE);
  return erg2J * exp(lnE);
}

double Model::ComputeTEPolyatomic(const double &Ttr, const double &Tve)
{
  return 1.5 * R_ * Ttr / species_weight_;
}
double Model::ComputeREPolyatomic(const double &Ttr, const double &Tve)
{
  return R_ * Ttr / species_weight_;
}
double Model::ComputeVEPolyatomic(const double &Ttr, const double &Tve)
{
  double       E       = 0.0;
  const double Rs      = R_ / species_weight_;
  const double tve_inv = 1.0 / Tve;
  for (int ivib = 0; ivib < thetv_[species_index_].size(); ivib++)
  {
    E += Rs * thetv_[species_index_][ivib] / (exp(thetv_[species_index_][ivib] * tve_inv) - 1.0);
  }
  return E;
}
double Model::ComputeEEPolyatomic(const double &Ttr, const double &Tve)
{
  double num = 0.0;
  double den = 0.0;
  den += ge_[species_index_][0] * exp(-thetel_[species_index_][0] / Tve);
  for (int iele = 1; iele < thetel_[species_index_].size(); iele++)
  {
    num += ge_[species_index_][iele] * thetel_[species_index_][iele] * exp(-thetel_[species_index_][iele] / Tve);
    den += ge_[species_index_][iele] * exp(-thetel_[species_index_][iele] / Tve);
  }
  const double E = R_ / species_weight_ * num / den;
  return E;
}

double Model::ComputeTEElectron(const double &Ttr, const double &Tve)
{
  return 1.5 * R_ * Tve / species_weight_;
}
double Model::ComputeREElectron(const double &Ttr, const double &Tve)
{
  return 0.0;
}
double Model::ComputeVEElectron(const double &Ttr, const double &Tve)
{
  return 0.0;
}
double Model::ComputeEEElectron(const double &Ttr, const double &Tve)
{
  return 0.0;
}
}  // namespace ANN