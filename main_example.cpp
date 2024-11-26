#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <fstream>
#include <unordered_map>
#include <cmath>

#include "ann.h"

const std::vector<std::string> species_name = {"N", "O", "C", "H", "Ar", "Np", "Op", "Cp", "Hp", "Arp",
                                               "N2", "O2", "C2", "H2", "NO", "NH", "OH", "CN", "CO", "CH",
                                               "SiO", "N2p", "O2p", "NOp", "CNp", "COp", "C3", "CO2", "C2H", "CH2",
                                               "H2O", "HCN", "CH3", "CH4", "C2H2", "H2O2", "e"};

const std::vector<double> species_weight = {
  1.400670E-02, // N
  1.599900E-02, // O
  1.201100E-02, // C
  1.008000E-03, // H
  3.994800E-02, // Ar
  1.400670E-02, // Np
  1.599940E-02, // Op
  1.201100E-02, // Cp
  1.008000E-03, // Hp
  3.994800E-02, // Arp
  2.801340E-02, // N2
  3.199800E-02, // O2
  2.402200E-02, // C2
  2.015880E-03, // H2
  3.000610E-02, // NO
  1.501468E-02, // NH
  1.700740E-02, // OH
  2.601744E-02, // CN
  2.801010E-02, // CO
  1.302000E-02, // CH
  4.408400E-02, // SiO
  2.801000E-02, // N2p
  3.199880E-02, // O2p
  3.000610E-02, // NOp
  2.616890E-02, // CNp
  2.800960E-02, // COp
  3.603300E-02, // C3
  4.400900E-02, // CO2
  2.502930E-02, // C2H
  1.402658E-02, // CH2
  1.801528E-02, // H2O
  2.702538E-02, // HCN
  1.503452E-02, // CH3
  1.604246E-02, // CH4
  2.604000E-02, // C2H2
  3.401400E-02, // H2O2
  5.485790E-07  // e
};

const std::vector<std::vector<double>> thetv = {
  {0.000000E+00}, // N
  {0.000000E+00}, // O
  {0.000000E+00}, // C
  {0.000000E+00}, // H
  {0.000000E+00}, // Ar
  {0.000000E+00}, // Np
  {0.000000E+00}, // Op
  {0.000000E+00}, // Cp
  {0.000000E+00}, // Hp
  {0.000000E+00}, // Arp
  {3.393500E+03}, // N2
  {2.273576E+03}, // O2
  {2.668550E+03}, // C2
  {6.332449E+03}, // H2
  {2.739662E+03}, // NO
  {4.722518E+03}, // NH
  {5.377918E+03}, // OH
  {2.976280E+03}, // CN
  {3.121919E+03}, // CO
  {4.116036E+03}, // CH
  {1.786348E+03}, // SiO
  {3.175740E+03}, // N2p
  {2.740576E+03}, // O2p
  {3.419184E+03}, // NOp
  {2.925145E+03}, // CNp
  {3.185783E+03}, // COp
  {3.058622E+02, 3.058622E+02, 2.479591E+03, 4.302663E+03}, // C3
  {1.341115E+03, 1.341115E+03, 2.753973E+03, 4.854217E+03}, // CO2
  {5.838549E+02, 5.944933E+02, 4.178792E+03, 6.938147E+03}, // C2H
  {2.089006E+03, 6.250888E+03, 6.740219E+03}, // CH2
  {3.213415E+03, 7.657207E+03, 7.868867E+03}, // H2O
  {1.537370E+03, 1.537370E+03, 4.404282E+03, 6.925168E+03}, // HCN
  {1.078861E+03, 2.813475E+03, 2.813475E+03, 6.221624E+03, 6.582388E+03, 6.582388E+03}, // CH3
  {2.688448E+03, 2.688448E+03, 2.688448E+03, 3.126155E+03, 3.126155E+03, 6.065759E+03, 6.277572E+03,
   6.277572E+03, 6.277572E+03}, // CH4
  {1.298803E+03, 1.298803E+03, 1.550028E+03, 1.550028E+03, 4.136480E+03, 6.859573E+03, 7.065823E+03}, // C2H2
  {4.050701E+02, 6.583442E+02, 9.295760E+02, 1.580336E+03, 1.945645E+03, 2.467823E+03, 3.641613E+03,
   6.467357E+03, 6.478553E+03}, // H2O2
  {0.000000E+00} // e
};

const std::vector<std::vector<double>> ge = {
  {4.000E+00, 1.000E+01, 6.000E+00, 1.200E+01, 6.000E+00, 1.200E+01, 2.000E+00, 2.000E+01, 1.200E+01, 4.000E+00,
   1.000E+01, 6.000E+00, 1.000E+01, 1.200E+01, 6.000E+00, 6.000E+00, 2.800E+01, 1.400E+01, 1.200E+01, 2.000E+01,
   1.000E+01, 2.000E+00, 2.000E+01, 1.200E+01, 1.000E+01, 4.000E+00, 6.000E+00, 1.800E+01, 9.000E+01, 1.260E+02,
   5.400E+01, 9.000E+01, 2.880E+02, 6.480E+02, 8.820E+02, 1.152E+03, 1.458E+03, 1.800E+03}, // N
  {5.000E+00, 4.000E+00, 5.000E+00, 1.000E+00, 5.000E+00, 3.000E+00, 1.500E+01, 9.000E+00, 5.000E+00, 3.000E+00,
   2.500E+01, 1.500E+01, 1.500E+01, 9.000E+00, 1.500E+01, 5.000E+00, 3.000E+00, 5.000E+00, 2.500E+01, 1.500E+01,
   3.500E+01, 2.100E+01, 1.500E+01, 9.000E+00, 2.500E+01, 1.500E+01, 3.500E+01, 2.100E+01, 2.880E+02, 3.920E+02,
   5.120E+02, 6.480E+02, 8.000E+02}, // O
  {1.000E+00, 8.000E+00, 5.000E+00, 1.000E+00, 9.000E+00, 3.000E+00, 2.700E+01, 9.000E+00, 1.200E+01, 4.500E+01,
   1.500E+01, 3.600E+01, 6.000E+01, 1.200E+01, 8.400E+01, 3.600E+01, 6.000E+01, 1.920E+02, 4.320E+02, 5.880E+02,
   7.680E+02, 9.720E+02, 1.200E+03}, // C
  {2.000E+00, 8.000E+00, 1.800E+01, 3.200E+01, 5.000E+01, 7.200E+01, 9.800E+01, 1.280E+02, 1.620E+02, 2.000E+02}, // H
  {1.000E+00, 5.000E+00, 3.000E+00, 1.000E+00, 3.000E+00, 3.000E+00, 7.000E+00, 5.000E+00, 3.000E+00, 5.000E+00,
   1.000E+00, 3.000E+00, 5.000E+00, 3.000E+00, 1.000E+00, 1.000E+00, 3.000E+00, 5.000E+00, 9.000E+00, 7.000E+00,
   5.000E+00, 5.000E+00, 3.000E+00, 7.000E+00, 3.000E+00, 5.000E+00, 5.000E+00, 7.000E+00, 1.000E+00, 3.000E+00,
   3.000E+00}, // Ar
  {1.000E+00, 3.000E+00, 5.000E+00, 5.000E+00, 1.000E+00, 5.000E+00, 1.500E+01, 9.000E+00, 5.000E+00, 1.200E+01,
   3.000E+00, 2.100E+01, 3.000E+00, 1.500E+01, 6.000E+01}, // Np
  {4.000E+00, 6.000E+00, 4.000E+00, 4.000E+00, 2.000E+00, 6.000E+00, 4.000E+00, 2.000E+00, 6.000E+00, 4.000E+00,
   2.000E+00, 4.000E+00, 6.000E+00, 2.000E+00, 4.000E+00}, // Op
  {2.000E+00, 4.000E+00, 1.200E+01, 1.000E+01, 2.000E+00, 6.000E+00, 2.000E+00, 6.000E+00, 4.000E+00, 1.000E+01,
   1.000E+01, 2.000E+00}, // Cp
  {1.000E+00}, // Hp
  {4.000E+00, 2.000E+00, 2.000E+00, 2.000E+01, 2.800E+01, 6.000E+00, 1.200E+01, 1.400E+01}, // Arp
  {1.000E+00, 3.000E+00, 6.000E+00, 6.000E+00, 3.000E+00, 6.000E+00, 1.000E+00, 2.000E+00, 2.000E+00, 5.000E+00,
   6.000E+00, 2.000E+00, 1.000E+00, 1.000E+00, 1.000E+00, 2.000E+00, 6.000E+00, 2.000E+00, 1.000E+00, 2.000E+00,
   2.000E+00, 2.000E+00, 2.000E+00, 1.000E+00, 6.000E+00}, // N2
  {3.000E+00, 2.000E+00, 1.000E+00, 1.000E+00, 6.000E+00, 3.000E+00, 1.000E+01, 3.000E+00, 3.000E+00, 6.000E+00,
   6.000E+00, 6.000E+00, 2.000E+00, 6.000E+00, 2.000E+00, 3.000E+00, 1.000E+00, 3.000E+00, 6.000E+00, 2.000E+00,
   2.000E+00, 6.000E+00, 2.000E+00, 3.000E+00, 1.000E+00}, // O2
  {1.000E+00}, // C2
  {1.000E+00}, // H2
  {4.000E+00, 8.000E+00, 2.000E+00, 4.000E+00, 4.000E+00, 4.000E+00, 2.000E+00, 4.000E+00, 2.000E+00, 4.000E+00,
   2.000E+00, 4.000E+00, 2.000E+00, 2.000E+00, 2.000E+00, 4.000E+00, 2.000E+00, 4.000E+00, 2.000E+00, 4.000E+00,
   2.000E+00, 2.000E+00}, // NO
  {1.000E+00}, // NH
  {1.000E+00}, // OH
  {1.000E+00}, // CN
  {1.000E+00}, // CO
  {1.000E+00}, // CH
  {1.000E+00}, // SiO
  {1.000E+00}, // N2p
  {1.000E+00}, // O2p
  {1.000E+00}, // NOp
  {1.000E+00}, // CNp
  {1.000E+00}, // COp
  {1.000E+00, 6.000E+00, 6.000E+00, 3.000E+00, 2.000E+00, 6.000E+00, 3.000E+00, 1.000E+00, 2.000E+00, 2.000E+00}, // C3
  {1.000E+00, 3.000E+00, 6.000E+00, 3.000E+00, 2.000E+00}, // CO2
  {2.000E+00, 4.000E+00}, // C2H
  {3.000E+00, 1.000E+00}, // CH2
  {1.000E+00}, // H2O
  {1.000E+00, 1.000E+00}, // HCN
  {2.000E+00, 2.000E+00}, // CH3
  {1.000E+00, 1.000E+00}, // CH4
  {2.000E+00, 3.000E+00, 6.000E+00, 1.000E+00, 3.000E+00, 1.000E+00}, // C2H2
  {1.000E+00}, // H2O2
  {1.000E+00}  // e
};

const std::vector<std::vector<double>> thetel = {
  {0.00000000000000E+00, 2.76647843943454E+04, 4.14928602636014E+04, 1.19901834423090E+05, 1.24012417375796E+05,
   1.26802203985981E+05, 1.34643533034722E+05, 1.36450635655863E+05, 1.37417493109212E+05, 1.39203014090769E+05,
   1.39320993720493E+05, 1.40703657429822E+05, 1.43394168497922E+05, 1.49192435666073E+05, 1.49914701204140E+05,
   1.50536252424151E+05, 1.50668619813597E+05, 1.50854221914017E+05, 1.50858538241934E+05, 1.51081548517632E+05,
   1.51264273066107E+05, 1.53197987972806E+05, 1.53694365683231E+05, 1.53969171893930E+05, 1.54272753624074E+05,
   1.54590723113940E+05, 1.54833876253249E+05, 1.58295571242473E+05, 1.58718571378314E+05, 1.58964602069568E+05,
   1.60226408597227E+05, 1.62367307243929E+05, 1.62479531769765E+05, 1.64449216075769E+05, 1.65613185837316E+05,
   1.66367104446773E+05, 1.66885063796782E+05, 1.67254829221649E+05}, // N
  {0.00000000000000E+00, 2.52361305531942E+02, 2.28304971275990E+04, 4.86191176541356E+04, 1.06135625920640E+05,
   1.10490800788629E+05, 1.24639723699696E+05, 1.27520153196132E+05, 1.37370013502128E+05, 1.38447656705340E+05,
   1.40166993992174E+05, 1.40264830758287E+05, 1.42575504969714E+05, 1.43420066465422E+05, 1.45523556936846E+05,
   1.46923485957841E+05, 1.47349363645626E+05, 1.47709057638687E+05, 1.48001129161053E+05, 1.48062996527860E+05,
   1.48148315943014E+05, 1.48148459820611E+05, 1.49096037675933E+05, 1.49447099013161E+05, 2.95504441835710E+05,
   1.51661375234447E+05, 1.51706840555170E+05, 1.51706984432767E+05, 1.53422437024476E+05, 1.54780641542276E+05,
   1.55548947911456E+05, 1.56068346037436E+05, 1.56436672686331E+05}, // O
  {0.00000000000000E+00, 4.74796070841168E+01, 1.46668822610754E+04, 3.11466222471806E+04, 8.68632217724056E+04,
   8.91782123117494E+04, 1.01271124358477E+05, 1.02889747327253E+05, 1.12489260614078E+05, 1.12903628094085E+05,
   1.12637454539220E+05, 1.16348057771642E+05, 1.20635610168935E+05, 1.20688844879908E+05, 1.21011130697691E+05,
   1.22349192351880E+05, 1.24280029706634E+05, 1.24468509358998E+05, 1.26199356853610E+05, 1.27531663403910E+05,
   1.28278388133506E+05, 1.28794908707542E+05, 1.29171868012270E+05}, // C
  {0.00000000000000E+00, 1.18352416578578E+05, 1.40269578718995E+05, 1.47940700570219E+05, 1.51491599669722E+05,
   1.53419559472532E+05, 1.54582090458106E+05, 1.55337447843536E+05, 1.55855407193544E+05, 1.56225172618411E+05}, // H
  {0.00000000000000E+00, 1.34013493036475E+05, 1.34886686174031E+05, 1.36042167157342E+05, 1.37259515507459E+05,
   1.49780175650736E+05, 1.51737918116171E+05, 1.51960209003883E+05, 1.52636433710838E+05, 1.52852681739467E+05,
   1.54027873953597E+05, 1.54139235213849E+05, 1.54366561817464E+05, 1.54663956810927E+05, 1.56428183908095E+05,
   1.60665666901554E+05, 1.60881771052585E+05, 1.61343474262079E+05, 1.62222998013913E+05, 1.62611755281614E+05,
   1.63195322815957E+05, 1.63256470794777E+05, 1.63507968834726E+05, 1.63613431113492E+05, 1.64233687435127E+05,
   1.64943435622236E+05, 1.65179538759281E+05, 1.65203854073212E+05, 1.65260973479310E+05, 1.65423986796966E+05,
   1.65987843100489E+05}, // Ar
  {0.00000000000000E+00, 7.00683898483785E+01, 1.88191897169772E+02, 2.20365805461136E+04, 4.70318600015539E+04,
   6.73125583511385E+04, 1.32721328335801E+05, 1.57141672912732E+05, 2.07454143559643E+05, 2.14461068871039E+05,
   2.23192611777087E+05, 2.39340872166947E+05, 2.42998053647520E+05, 2.47954910239342E+05, 2.70048811620190E+05}, // Np
  {0.00000000000000E+00, 3.85743751426990E+04, 3.86031794376634E+04, 5.82244004326094E+04, 5.82272635967942E+04,
   1.72418898328994E+05, 1.72653735343184E+05, 1.72772333646576E+05, 2.38820207918110E+05, 2.38831775676927E+05,
   2.66512070076253E+05, 2.66663573186131E+05, 2.66891676728771E+05, 2.71768292784621E+05, 2.72027229296346E+05}, // Op
  {0.0000000000000E+00}, // Cp
  {0.0000000000000E+00}, // Hp
  {0.0000000000000E+00}, // Arp
  {0.00000000000000E+00, 7.22318197268058E+04, 8.57785141791696E+04, 8.60503133480866E+04, 9.53512806206707E+04,
   1.28248001184972E+05, 9.80564096526487E+04, 9.96828020116755E+04, 1.03732006781406E+05, 1.13375546612982E+05,
   1.26468407960420E+05, 1.45052933316324E+05, 1.50317558476369E+05, 1.50349211547759E+05, 1.50379425843176E+05,
   1.52052722298898E+05, 1.52107395785843E+05, 1.52321773405708E+05, 1.63211868739638E+05, 1.63744215849369E+05,
   1.64459575262769E+05, 1.66085104356213E+05, 1.66374154449037E+05, 1.66719604559973E+05, 1.74470290722462E+05}, // N2
  {0.00000000000000E+00, 1.13923720258408E+04, 1.89847928313827E+04, 4.75620489473265E+04, 4.99111384772125E+04,
   5.09295041103682E+04, 5.65136814138492E+04, 7.16413748433155E+04, 7.77385045564217E+04, 7.98865970829849E+04,
   8.20692202328820E+04, 9.42829894612780E+04, 9.95345217599758E+04, 1.07785901960806E+05, 1.08273647015398E+05,
   1.08282279671231E+05, 1.09477902504168E+05, 1.14933740990924E+05, 1.23544815184816E+05, 1.24603754300389E+05,
   1.24813815592337E+05, 1.24947621757756E+05, 1.24951938085673E+05, 1.25170632033454E+05, 1.25474213763598E+05}, // O2
  {0.0000000000000E+00},  // C2
  {0.0000000000000E+00},  // H2
  {0.00000000000000E+00, 5.58345791549491E+04, 6.32567927629744E+04, 6.60862895899933E+04, 6.89893078691940E+04,
   7.50750424766000E+04, 7.63769908538853E+04, 8.68505605438498E+04, 8.72312606661061E+04, 8.89163550848006E+04,
   8.98846513141221E+04, 8.99024921361780E+04, 9.05177127419104E+04, 9.27104073236132E+04, 9.62541125432550E+04,
   9.69360923540996E+04, 9.74871435514698E+04, 9.74943374313310E+04, 1.00322970992767E+05, 1.01450971355007E+05,
   1.01597726504177E+05, 1.02475379847247E+05}, // NO
  {0.00000000000000E+00}, // NH
  {0.00000000000000E+00}, // OH
  {0.00000000000000E+00}, // CN
  {0.00000000000000E+00}, // CO
  {0.00000000000000E+00}, // CH
  {0.00000000000000E+00}, // SiO
  {0.00000000000000E+00}, // N2p
  {0.00000000000000E+00}, // O2p
  {0.00000000000000E+00}, // NOp
  {0.00000000000000E+00}, // CNp
  {0.00000000000000E+00}, // COp
  {0.00000000000000E+00, 2.01428636114430E+04, 3.09336834032880E+04, 3.42428681394540E+04, 3.55025165031550E+04,
   4.18683807923570E+04, 4.71918518896670E+04, 4.73357294868920E+04, 4.84867502646890E+04, 5.82704268759610E+04}, // C3
  {0.00000000000000E+00, 4.31632791673780E+04, 4.74796070841160E+04, 5.17959350008540E+04, 6.47449187510680E+04}, // CO2
  {0.00000000000000E+00, 5.75510388898380E+03}, // C2H
  {0.00000000000000E+00, 4.52782798465800E+03}, // CH2
  {0.00000000000000E+00}, // H2O
  {0.00000000000000E+00, 7.51852527160740E+04}, // HCN
  {0.00000000000000E+00, 6.64786437976240E+04}, // CH3
  {0.00000000000000E+00, 9.88870725724650E+04}, // CH4
  {0.00000000000000E+00, 3.59693993061490E+04, 5.03571590286080E+04, 6.07134684768350E+04, 7.19387986122980E+04,
   7.78608005140620E+04}, // C2H2
  {0.00000000000000E+00}, // H2O2
  {0.00000000000000E+00} // e
};

const std::vector<double> lin = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

const std::vector<bool> molecule_flag = {false, false, false, false, false, false, false, false, false, false,
                                         true, true, true, true, true, true, true, true, true, true,
                                         true, true, true, true, true, true, true, true, true, true,
                                         true, true, true, true, true, true, true};

int main(void)
{
  // Print the list of units used by ANN
  const char *unit_info = ANN_Units();
  std::cout << std::string(unit_info) << std::endl;
  // Initialize ANN
  const std::string dir = "model/";
  const int nspecies = 5;
  std::unordered_map<int, std::string> species_pack;
  for (int i = 0; i < species_name.size(); i++)
  {
    if ((species_name[i].compare("N") != 0) &&
        (species_name[i].compare("O") != 0) &&
        (species_name[i].compare("N2") != 0) &&
        (species_name[i].compare("O2") != 0) &&
        (species_name[i].compare("NO") != 0))
      continue;
    species_pack.insert({i + 1, species_name[i]});
  }
  const std::string mode_pack[4] = {"T", "R", "V", "E"};
  std::vector<double> y =
  {
    0.7330439463792710,
    0.1856148633861940,
    0.0318961674512450,
    0.0232148408655277,
    0.0262301911965476
  };
  const double R = 8.31446261815324; // Universal gas constant (J/K-mol)
  std::vector<ANN_DEF> fptrs;
  std::vector<ANN_GRAD_DEF> gptrs;
  fptrs.resize(4 * nspecies);
  gptrs.resize(4 * nspecies);
  const char *initlog = ANN_Init(dir.c_str());
  std::cout << std::string(initlog) << std::endl;
  int i = 0;
  for (auto &species : species_pack)
  {
    for (int j = 0; j < 4; j++)
    {
      fptrs[i * 4 + j] = ANN_MODEL(species.second.c_str(), mode_pack[j].c_str());
      gptrs[i * 4 + j] = ANN_MODEL_GRAD(species.second.c_str(), mode_pack[j].c_str());
    }
    i++;
  }
  // Usage example of ANN subroutines
  double et_ann = 0.0; // Translational energy of N2 [J/kg]
  double er_ann = 0.0; // Rotational energy of N2 [J/kg]
  double ev_ann = 0.0; // Vibrational energy of N2 [J/kg]
  double ee_ann = 0.0; // Electronic energy of N2 [J/kg]
  double et_rrho = 0.0;
  double er_rrho = 0.0;
  double ev_rrho = 0.0;
  double ee_rrho = 0.0;
  // grad[0] = derivative of E with respect to Ttr (dEV/dTtr)
  // grad[1] = derivative of E with respect to Tve (dEV/dTve)
  std::vector<double> CvT(2, 0.0);
  std::vector<double> CvR(2, 0.0);
  std::vector<double> CvV(2, 0.0);
  std::vector<double> CvE(2, 0.0);
  std::cout << std::setprecision(8) << std::scientific << std::uppercase;
  std::string data_direc = "./database";
  double Cv = 0.0;
  double Cp = 0.0;
  double e_t_ann = 0.0;
  double e_ve_ann = 0.0;
  double e_t_rrho = 0.0;
  double e_ve_rrho = 0.0;
  int ispecies = 0;
#ifdef WRITE
  ispecies = 0;
  for (auto &species : species_pack)
  {
    const int index = species.first - 1;
    std::string species_direc = data_direc + "/" + std::string(species_name[index]) + "/";
    std::ofstream model_en_file(data_direc + "/model_en_" + std::string(species_name[index]) + ".plt");
    model_en_file << std::setprecision(8) << std::scientific << std::uppercase;
    model_en_file << "variables = \"Ttr\", \"Tve\", \"es\", \"est\", \"esr\", \"esv\", \"ese\"\n";
    model_en_file << "zone I=1000,J=1000\n";
    for (int itemp1 = 0; itemp1 < 1000; itemp1++)
    {
      const double temp1 = 50.0 + double(itemp1) * 50.0;
      for (int itemp2 = 0; itemp2 < 1000; itemp2++)
      {
        const double temp2 = 50.0 + double(itemp2) * 50.0;
        et_ann = ComputeEnergy(fptrs[ispecies * 4 + 0], temp1, temp2);
        er_ann = ComputeEnergy(fptrs[ispecies * 4 + 1], temp1, temp2);
        ev_ann = ComputeEnergy(fptrs[ispecies * 4 + 2], temp1, temp2);
        ee_ann = ComputeEnergy(fptrs[ispecies * 4 + 3], temp1, temp2);
        et_rrho = 1.5 * R / species_weight[index] * temp1;
        er_rrho = molecule_flag[index] ? 0.5 * R / species_weight[index] * temp1 * lin[index] : 0.0;
        ev_rrho = molecule_flag[index] ? R / species_weight[index] * thetv[index][0] / (exp(thetv[index][0] / temp2) - 1.0) : 0.0;
        double num = 0.0;
        double den = 0.0;
        den += ge[index][0] * exp(-thetel[index][0] / temp2);
        for (int i = 1; i < thetel[index].size(); i++)
        {
          num += ge[index][i] * thetel[index][i] * exp(-thetel[index][i] / temp2);
          den += ge[index][i] * exp(-thetel[index][i] / temp2);
        }
        ee_rrho = R / species_weight[index] * num / den;
        const double es = et_ann + er_ann + ev_ann + ee_ann;
        model_en_file << temp1 << "\t" << temp2 << "\t" << es << "\t" << et_ann << "\t" << er_ann << "\t" << ev_ann << "\t" << ee_ann << "\n";
      }
    }
    model_en_file.close();
    std::ofstream model_cv_file(data_direc + "/model_cv_" + std::string(species_pack[ispecies]) + ".plt");
    model_cv_file << std::setprecision(8) << std::scientific << std::uppercase;
    model_cv_file << "variables = \"Ttr\", \"Tve\", \"Cp\", \"Cvt\", \"Cvr\", \"Cvv\", \"Cve\"\n";
    model_cv_file << "zone I=1000,J=1000\n";
    for (int itemp1 = 0; itemp1 < 1000; itemp1++)
    {
      const double temp1 = 50.0 + double(itemp1) * 50.0;
      for (int itemp2 = 0; itemp2 < 1000; itemp2++)
      {
        const double temp2 = 50.0 + double(itemp2) * 50.0;
        et_ann = ComputeCv(gptrs[ispecies * 4 + 0], &CvT[0], temp1, temp2);
        er_ann = ComputeCv(gptrs[ispecies * 4 + 1], &CvR[0], temp1, temp2);
        ev_ann = ComputeCv(gptrs[ispecies * 4 + 2], &CvV[0], temp1, temp2);
        ee_ann = ComputeCv(gptrs[ispecies * 4 + 3], &CvE[0], temp1, temp2);
        for (int i = 0; i < 2; i++)
        {
          CvT[i] *= species_weight[ispecies] / R;
          CvR[i] *= species_weight[ispecies] / R;
          CvV[i] *= species_weight[ispecies] / R;
          CvE[i] *= species_weight[ispecies] / R;
        }
        double cp = (CvT[0] + CvR[0] + CvV[1] + CvE[1]) + 1.0;
        model_cv_file << temp1 << "\t" << temp2 << "\t" << cp << "\t" << CvT[0] << "\t" << CvR[0] << "\t" << CvV[1] << "\t" << CvE[1] << "\n";
      }
    }
    model_cv_file.close();
    ispecies++;
  }
#endif
#ifdef TEST
  ispecies = 0;
  for (auto &species : species_pack)
  {
    const double Ttr = 21209.8524181727;
    const double Tve = 6367.10208721561;
    std::cout << "Species " << species.second << std::endl;
    std::cout << "Translational-rotational temperature (Ttr) = " << Ttr << " K" << std::endl;
    std::cout << "Vibrational-electronic temperature (Tve)   = " << Tve << " K" << std::endl;
    // input: species, mode, Ttr, Tve
    // output: energy
    const int index = species.first - 1;
    et_ann = ComputeEnergy(fptrs[ispecies * 4 + 0], Ttr, Tve);
    er_ann = ComputeEnergy(fptrs[ispecies * 4 + 1], Ttr, Tve);
    ev_ann = ComputeEnergy(fptrs[ispecies * 4 + 2], Ttr, Tve);
    ee_ann = ComputeEnergy(fptrs[ispecies * 4 + 3], Ttr, Tve);
    et_rrho = 1.5 * R / species_weight[index] * Ttr;
    er_rrho = molecule_flag[index] ? 0.5 * R / species_weight[index] * Ttr * lin[index] : 0.0;
    ev_rrho = molecule_flag[index] ? R / species_weight[index] * thetv[index][0] / (exp(thetv[index][0] / Tve) - 1.0) : 0.0;
    double num = 0.0;
    double den = 0.0;
    den += ge[index][0] * exp(-thetel[index][0] / Tve);
    for (int i = 1; i < thetel[index].size(); i++)
    {
      num += ge[index][i] * thetel[index][i] * exp(-thetel[index][i] / Tve);
      den += ge[index][i] * exp(-thetel[index][i] / Tve);
    }
    ee_rrho = R / species_weight[index] * num / den;
    // std::cout << "et_ann = " << et_ann << std::endl;
    // std::cout << "er_ann = " << er_ann << std::endl;
    // std::cout << "ev_ann = " << ev_ann << std::endl;
    // std::cout << "ee_ann = " << ee_ann << std::endl;
    // std::cout << "es_ann = " << et_ann + er_ann + ev_ann + ee_ann << std::endl;
    //
    // std::cout << "et_rrho = " << et_rrho << std::endl;
    // std::cout << "er_rrho = " << er_rrho << std::endl;
    // std::cout << "ev_rrho = " << ev_rrho << std::endl;
    // std::cout << "ee_rrho = " << ee_rrho << std::endl;
    // std::cout << "es_rrho = " << et_rrho + er_rrho + ev_rrho + ee_rrho << std::endl;
    // std::cout << std::endl << "Example 1, ANN_MODEL" << std::endl;
    // std::cout << "Translational energy (ET) = " << et_ann << " J/kg" << std::endl;
    // std::cout << "Rotational energy (ER)    = " << er_ann << " J/kg" << std::endl;
    // std::cout << "Vibrational energy (EV)   = " << ev_ann << " J/kg" << std::endl;
    // std::cout << "Electronic energy (EE)    = " << ee_ann << " J/kg" << std::endl;
    // input: species, mode, Ttr, Tve
    // output: specific heat
    // ET = ComputeCv(gptrs[ispecies*4+0], &CvT[0], Ttr, Tve);
    // ER = ComputeCv(gptrs[ispecies*4+1], &CvR[0], Ttr, Tve);
    // EV = ComputeCv(gptrs[ispecies*4+2], &CvV[0], Ttr, Tve);
    // EE = ComputeCv(gptrs[ispecies*4+3], &CvE[0], Ttr, Tve);
    // std::cout << std::endl << "Example 2, ANN_MODEL_Grad" << std::endl;
    // std::cout << "Translational energy (ET) = " << ET << " J/kg" << std::endl;
    // std::cout << "dET/dTtr                  = " << CvT[0] << " J/kg-K" << std::endl;
    // std::cout << "dET/dTve                  = " << CvT[1] << " J/kg-K" << std::endl;
    // std::cout << "Rotational energy (ER)    = " << ER << " J/kg" << std::endl;
    // std::cout << "dER/dTtr                  = " << CvR[0] << " J/kg-K" << std::endl;
    // std::cout << "dER/dTve                  = " << CvR[1] << " J/kg-K" << std::endl;
    // std::cout << "Vibrational energy (EV)   = " << EV << " J/kg" << std::endl;
    // std::cout << "dEV/dTtr                  = " << CvV[0] << " J/kg-K" << std::endl;
    // std::cout << "dEV/dTve                  = " << CvV[1] << " J/kg-K" << std::endl;
    // std::cout << "Electronic energy (EE)    = " << EE << " J/kg" << std::endl;
    // std::cout << "dEE/dTtr                  = " << CvE[0] << " J/kg-K" << std::endl;
    // std::cout << "dEE/dTve                  = " << CvE[1] << " J/kg-K" << std::endl;
    std::cout << std::endl
              << std::endl;
    e_t_ann += y[ispecies] * (et_ann + er_ann + ev_ann + ee_ann);
    e_ve_ann += y[ispecies] * (ev_ann + ee_ann);
    e_t_rrho += y[ispecies] * (et_rrho + er_rrho + ev_rrho + ee_rrho);
    e_ve_rrho += y[ispecies] * (ev_rrho + ee_rrho);
    ispecies++;
  }
#endif
  // Finalize ANN
  const char *finlog = ANN_Finalize();
  std::cout << std::string(finlog) << std::endl;
  std::cout << "Total energy from ANN model = " << e_t_ann << std::endl;
  std::cout << "VE energy from ANN model = " << e_ve_ann << std::endl;
  std::cout << "Total energy from RRHO model = " << e_t_rrho << std::endl;
  std::cout << "VE energy from RRHO model = " << e_ve_rrho << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Diff. of total energy = " << fabs(e_t_ann - e_t_rrho) / e_t_ann * 100.0 << " %" << std::endl;
  std::cout << "Diff. of VE energy = " << fabs(e_ve_ann - e_ve_rrho) / e_ve_ann * 100.0 << " %" << std::endl;

  return 0;
}