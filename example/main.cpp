#include <iostream>

#include "model.h"

int main()
{
  const int                                nspecies = 4;
  std::vector<std::shared_ptr<ANN::Model>> models;
  std::vector<std::string>                 species_name = {"N2", "N2p", "N", "Np"};
  models.resize(nspecies);
  for (int i = 0; i < nspecies; i++)
  {
    models[i]      = std::make_shared<ANN::Model>();
    bool init_flag = models[i]->Init(species_name[i].c_str());
    if (!init_flag) std::cerr << "ANN model for species " << species_name[i] << " cannot be initiated\n";
  }

  const double ttr = 500.0;
  const double tve = 500.0;
  for (int ispecies = 0; ispecies < nspecies; ispecies++)
  {
    const double est = models[ispecies]->ComputeTranslationalEnergy(ttr, tve);
    const double esr = models[ispecies]->ComputeRotationalEnergy(ttr, tve);
    const double esv = models[ispecies]->ComputeVibrationalEnergy(ttr, tve);
    const double ese = models[ispecies]->ComputeElectronicEnergy(ttr, tve);

    double cvt[2], cvr[2], cvv[2], cve[2];
    models[ispecies]->ComputeTranslationalCv(&cvt[0], ttr, tve);
    models[ispecies]->ComputeRotationalCv(&cvr[0], ttr, tve);
    models[ispecies]->ComputeVibrationalCv(&cvv[0], ttr, tve);
    models[ispecies]->ComputeElectronicCv(&cve[0], ttr, tve);
  }
  return 0;
}