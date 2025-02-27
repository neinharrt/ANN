#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "neural_network.h"

namespace ANN
{
class Model
{
  private:

    std::vector<std::shared_ptr<NeuralNetwork>> networks_;

    std::string species_name_;
    double      species_weight_;
    int         molecule_flag_;
    int         species_index_;

  public:

    Model() {};
    ~Model() {};

    bool Init(const char *species_name);

  public:

    double ComputeTranslationalEnergy(const double &tt, const double &tr, const double &tv, const double &te)
    {
      return (this->*ComputeTE)(tt, tr, tv, te);
    };
    double ComputeRotationalEnergy(const double &tt, const double &tr, const double &tv, const double &te)
    {
      return (this->*ComputeRE)(tt, tr, tv, te);
    };
    double ComputeVibrationalEnergy(const double &tt, const double &tr, const double &tv, const double &te)
    {
      return (this->*ComputeVE)(tt, tr, tv, te);
    };
    double ComputeElectronicEnergy(const double &tt, const double &tr, const double &tv, const double &te)
    {
      return (this->*ComputeEE)(tt, tr, tv, te);
    };

    void ComputeTranslationalCv(double *cv, const double &tt, const double &tr, const double &tv, const double &te)
    {
      return (this->*ComputeTC)(&cv[0], tt, tr, tv, te);
    };
    void ComputeRotationalCv(double *cv, const double &tt, const double &tr, const double &tv, const double &te)
    {
      return (this->*ComputeRC)(&cv[0], tt, tr, tv, te);
    };
    void ComputeVibrationalCv(double *cv, const double &tt, const double &tr, const double &tv, const double &te)
    {
      return (this->*ComputeVC)(&cv[0], tt, tr, tv, te);
    };
    void ComputeElectronicCv(double *cv, const double &tt, const double &tr, const double &tv, const double &te)
    {
      return (this->*ComputeEC)(&cv[0], tt, tr, tv, te);
    };

    void ComputeTranslationalCv2(double *cv, const double &tt, const double &tr, const double &tv, const double &te)
    {
      return (this->*ComputeTC2)(&cv[0], tt, tr, tv, te);
    };
    void ComputeRotationalCv2(double *cv, const double &tt, const double &tr, const double &tv, const double &te)
    {
      return (this->*ComputeRC2)(&cv[0], tt, tr, tv, te);
    };
    void ComputeVibrationalCv2(double *cv, const double &tt, const double &tr, const double &tv, const double &te)
    {
      return (this->*ComputeVC2)(&cv[0], tt, tr, tv, te);
    };
    void ComputeElectronicCv2(double *cv, const double &tt, const double &tr, const double &tv, const double &te)
    {
      return (this->*ComputeEC2)(&cv[0], tt, tr, tv, te);
    };

  protected:

    double (Model::*ComputeTE)(const double &tt, const double &tr, const double &tv, const double &te);
    double (Model::*ComputeRE)(const double &tt, const double &tr, const double &tv, const double &te);
    double (Model::*ComputeVE)(const double &tt, const double &tr, const double &tv, const double &te);
    double (Model::*ComputeEE)(const double &tt, const double &tr, const double &tv, const double &te);

    void (Model::*ComputeTC)(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void (Model::*ComputeRC)(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void (Model::*ComputeVC)(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void (Model::*ComputeEC)(double *cv, const double &tt, const double &tr, const double &tv, const double &te);

    void (Model::*ComputeTC2)(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void (Model::*ComputeRC2)(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void (Model::*ComputeVC2)(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void (Model::*ComputeEC2)(double *cv, const double &tt, const double &tr, const double &tv, const double &te);

    double ComputeTEAtom(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeTCAtom(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeTCAtom2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    double ComputeREAtom(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeRCAtom(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeRCAtom2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    double ComputeVEAtom(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeVCAtom(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeVCAtom2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    double ComputeEEAtom(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeECAtom(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeECAtom2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);

    double ComputeTEDiatomic(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeTCDiatomic(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeTCDiatomic2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    double ComputeREDiatomic(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeRCDiatomic(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeRCDiatomic2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    double ComputeVEDiatomic(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeVCDiatomic(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeVCDiatomic2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    double ComputeEEDiatomic(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeECDiatomic(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeECDiatomic2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);

    double ComputeTEPolyatomic(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeTCPolyatomic(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeTCPolyatomic2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    double ComputeREPolyatomic(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeRCPolyatomic(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeRCPolyatomic2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    double ComputeVEPolyatomic(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeVCPolyatomic(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeVCPolyatomic2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    double ComputeEEPolyatomic(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeECPolyatomic(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeECPolyatomic2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);

    double ComputeTEElectron(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeTCElectron(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeTCElectron2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    double ComputeREElectron(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeRCElectron(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeRCElectron2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    double ComputeVEElectron(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeVCElectron(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeVCElectron2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    double ComputeEEElectron(const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeECElectron(double *cv, const double &tt, const double &tr, const double &tv, const double &te);
    void   ComputeECElectron2(double *cv, const double &tt, const double &tr, const double &tv, const double &te);

  protected:

    const double R_ = 8.31446261815324;  // Universal gas constant (J/K-mol)

    const std::vector<std::string> species_pack_ = {
        "N", "O", "C", "H", "Ar", "Np", "Op", "Cp", "Hp", "Arp",
        "N2", "O2", "C2", "H2", "NO", "NH", "OH", "CN", "CO", "CH",
        "SiO", "N2p", "O2p", "NOp", "CNp", "COp", "C3", "CO2", "C2H", "CH2",
        "H2O", "HCN", "CH3", "CH4", "C2H2", "H2O2", "e"};

    const std::vector<double> weight_pack_ = {
        // N, O, C, H, Ar
        1.400670E-02, 1.599900E-02, 1.201100E-02, 1.008000E-03, 3.994800E-02,
        // Np, Op, Cp, Hp, Arp
        1.400670E-02, 1.599940E-02, 1.201100E-02, 1.008000E-03, 3.994800E-02,
        // N2, O2, C2, H2, NO
        2.801340E-02, 3.199800E-02, 2.402200E-02, 2.015880E-03, 3.000610E-02,
        // NH, OH, CN, CO, CH
        1.501468E-02, 1.700740E-02, 2.601744E-02, 2.801010E-02, 1.302000E-02,
        // SiO, N2p, O2p, NOp, CNp
        4.408400E-02, 2.801000E-02, 3.199880E-02, 3.000610E-02, 2.616890E-02,
        // COp, C3, CO2, C2H, CH2
        2.800960E-02, 3.603300E-02, 4.400900E-02, 2.502930E-02, 1.402658E-02,
        // H2O, HCN, CH3, CH4, C2H2
        1.801528E-02, 2.702538E-02, 1.503452E-02, 1.604246E-02, 2.604000E-02,
        // H2O2, e
        3.401400E-02, 5.485790E-07};

    const std::vector<int> flag_pack_ = {
        // N, O, C, H, Ar, Np, Op, Cp, Hp, Arp
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // N2, O2, C2, H2, NO, NH, OH, CN, CO, CH
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        // SiO, N2p, O2p, NOp, CNp, COp, C3, CO2, C2H, CH2
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
        // H2O, HCN, CH3, CH4, C2H2, H2O2, e
        2, 2, 2, 2, 2, 2, 3};

    const std::vector<std::vector<double>> thetv_ = {
        // N, O, C, H, Ar
        {0.000000E+00},
        {0.000000E+00},
        {0.000000E+00},
        {0.000000E+00},
        {0.000000E+00},
        // Np, Op, Cp, Hp, Arp
        {0.000000E+00},
        {0.000000E+00},
        {0.000000E+00},
        {0.000000E+00},
        {0.000000E+00},
        // N2, O2, C2, H2, NO
        {3.393500E+03},
        {2.273576E+03},
        {2.668550E+03},
        {6.332449E+03},
        {2.739662E+03},
        // NH, OH, CN, CO, CH
        {4.722518E+03},
        {5.377918E+03},
        {2.976280E+03},
        {3.121919E+03},
        {4.116036E+03},
        // SiO, N2p, O2p, NOp, CNp
        {1.786348E+03},
        {3.175740E+03},
        {2.740576E+03},
        {3.419184E+03},
        {2.925145E+03},
        // COp, C3, CO2, C2H, CH2
        {3.185783E+03},
        {3.058622E+02, 3.058622E+02, 2.479591E+03, 4.302663E+03},
        {1.341115E+03, 1.341115E+03, 2.753973E+03, 4.854217E+03},
        {5.838549E+02, 5.944933E+02, 4.178792E+03, 6.938147E+03},
        {2.089006E+03, 6.250888E+03, 6.740219E+03},
        // H2O, HCN, CH3, CH4, C2H2
        {3.213415E+03, 7.657207E+03, 7.868867E+03},
        {1.537370E+03, 1.537370E+03, 4.404282E+03, 6.925168E+03},
        {1.078861E+03, 2.813475E+03, 2.813475E+03, 6.221624E+03, 6.582388E+03, 6.582388E+03},
        {2.688448E+03, 2.688448E+03, 2.688448E+03, 3.126155E+03, 3.126155E+03, 6.065759E+03, 6.277572E+03, 6.277572E+03, 6.277572E+03},
        {1.298803E+03, 1.298803E+03, 1.550028E+03, 1.550028E+03, 4.136480E+03, 6.859573E+03, 7.065823E+03},
        // H2O2, e
        {4.050701E+02, 6.583442E+02, 9.295760E+02, 1.580336E+03, 1.945645E+03, 2.467823E+03, 3.641613E+03, 6.467357E+03, 6.478553E+03},
        {0.000000E+00}};

    const std::vector<std::vector<double>> ge_ = {
        {1.0},                                               // N
        {1.0},                                               // O
        {1.0},                                               // C
        {1.0},                                               // H
        {1.0},                                               // Ar
        {1.0},                                               // Np
        {1.0},                                               // Op
        {1.0},                                               // Cp
        {1.0},                                               // Hp
        {1.0},                                               // Arp
        {1.0},                                               // N2
        {1.0},                                               // O2
        {1.0},                                               // C2
        {1.0},                                               // H2
        {1.0},                                               // NO
        {1.0},                                               // NH
        {1.0},                                               // OH
        {1.0},                                               // CN
        {1.0},                                               // CO
        {1.0},                                               // CH
        {1.0},                                               // SiO
        {1.0},                                               // N2p
        {1.0},                                               // O2p
        {1.0},                                               // NOp
        {1.0},                                               // CNp
        {1.0},                                               // COp
        {1.0, 6.0, 6.0, 3.0, 2.0, 6.0, 3.0, 1.0, 2.0, 2.0},  // C3
        {1.0, 3.0, 6.0, 3.0, 2.0},                           // CO2
        {2.0, 4.0},                                          // C2H
        {3.0, 1.0},                                          // CH2
        {1.0},                                               // H2O
        {1.0, 1.0},                                          // HCN
        {2.0, 2.0},                                          // CH3
        {1.0, 1.0},                                          // CH4
        {2.0, 3.0, 6.0, 1.0, 3.0, 1.0},                      // C2H2
        {1.0},                                               // H2O2
        {1.0}                                                // e
    };

    const std::vector<std::vector<double>> thetel_ = {
        {0.0000000000000E+00},                                                                                                                                                                                               // N
        {0.0000000000000E+00},                                                                                                                                                                                               // O
        {0.0000000000000E+00},                                                                                                                                                                                               // C
        {0.0000000000000E+00},                                                                                                                                                                                               // H
        {0.0000000000000E+00},                                                                                                                                                                                               // Ar
        {0.0000000000000E+00},                                                                                                                                                                                               // Np
        {0.0000000000000E+00},                                                                                                                                                                                               // Op
        {0.0000000000000E+00},                                                                                                                                                                                               // Cp
        {0.0000000000000E+00},                                                                                                                                                                                               // Hp
        {0.0000000000000E+00},                                                                                                                                                                                               // Arp
        {0.0000000000000E+00},                                                                                                                                                                                               // N2
        {0.0000000000000E+00},                                                                                                                                                                                               // O2
        {0.0000000000000E+00},                                                                                                                                                                                               // C2
        {0.0000000000000E+00},                                                                                                                                                                                               // H2
        {0.0000000000000E+00},                                                                                                                                                                                               // NO
        {0.0000000000000E+00},                                                                                                                                                                                               // NH
        {0.0000000000000E+00},                                                                                                                                                                                               // OH
        {0.0000000000000E+00},                                                                                                                                                                                               // CN
        {0.0000000000000E+00},                                                                                                                                                                                               // CO
        {0.0000000000000E+00},                                                                                                                                                                                               // CH
        {0.0000000000000E+00},                                                                                                                                                                                               // SiO
        {0.0000000000000E+00},                                                                                                                                                                                               // N2p
        {0.0000000000000E+00},                                                                                                                                                                                               // O2p
        {0.0000000000000E+00},                                                                                                                                                                                               // NOp
        {0.0000000000000E+00},                                                                                                                                                                                               // CNp
        {0.0000000000000E+00},                                                                                                                                                                                               // COp
        {0.0000000000000E+00, 2.0142863611443E+04, 3.0933683403288E+04, 3.4242868139454E+04, 3.5502516503155E+04, 4.1868380792357E+04, 4.7191851889667E+04, 4.7335729486892E+04, 4.8486750264689E+04, 5.8270426875961E+04},  // C3
        {0.0000000000000E+00, 4.3163279167378E+04, 4.7479607084116E+04, 5.1795935000854E+04, 6.4744918751068E+04},                                                                                                           // CO2
        {0.0000000000000E+00, 5.7551038889838E+03},                                                                                                                                                                          // C2H
        {0.0000000000000E+00, 4.5278279846580E+03},                                                                                                                                                                          // CH2
        {0.0000000000000E+00},                                                                                                                                                                                               // H2O
        {0.0000000000000E+00, 7.5185252716074E+04},                                                                                                                                                                          // HCN
        {0.0000000000000E+00, 6.6478643797624E+04},                                                                                                                                                                          // CH3
        {0.0000000000000E+00, 9.8887072572465E+04},                                                                                                                                                                          // CH4
        {0.0000000000000E+00, 3.5969399306149E+04, 5.0357159028608E+04, 6.0713468476835E+04, 7.1938798612298E+04, 7.7860800514062E+04},                                                                                      // C2H2
        {0.0000000000000E+00},                                                                                                                                                                                               // H2O2
        {0.0000000000000E+00}                                                                                                                                                                                                // e
    };
};
}  // namespace ANN
// extern std::vector<NeuralNetwork> networks;