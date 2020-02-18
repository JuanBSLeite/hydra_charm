#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <string.h>


#define torad(x)(x*M_PI/180.)

//hydra
#include <hydra/host/System.h>
#include <hydra/device/System.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/Placeholders.h>
#include <hydra/Complex.h>
#include <hydra/Tuple.h>
#include <hydra/Range.h>
#include <hydra/Distance.h>

#include <hydra/LogLikelihoodFCN.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/AddPdf.h>


#include <hydra/multivector.h>
#include <hydra/PhaseSpace.h>
#include <hydra/PhaseSpaceIntegrator.h>
#include <hydra/Decays.h>

#include <hydra/DenseHistogram.h>
#include <hydra/SparseHistogram.h>

#include <hydra/functions/BreitWignerLineShape.h>
#include <hydra/functions/CosHelicityAngle.h>
#include <hydra/functions/ZemachFunctions.h>

//Minuit2
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinimize.h"

//ROOT

#include "TROOT.h"
#include "TTree.h"
#include <TFile.h>

using namespace ROOT;
using namespace ROOT::Minuit2;
using namespace hydra::placeholders;

//compute the Wave parity in compile time
template<hydra::Wave L, bool Flag=(L%2)>
struct parity;
//positive
template<hydra::Wave L>
struct parity<L, false>: std::integral_constant<int,1>{};
//negative
template<hydra::Wave L>
struct parity<L, true>:  std::integral_constant<int,-1>{};

template<hydra::Wave L>
class Resonance: public hydra::BaseFunctor<Resonance<L>, hydra::complex<double>, 4>
{
	using hydra::BaseFunctor<Resonance<L>, hydra::complex<double>, 4>::_par;

public:

	Resonance() = delete;

	Resonance(hydra::Parameter const& c_re, hydra::Parameter const& c_im,
			  hydra::Parameter const& mass, hydra::Parameter const& width,
			  double mother_mass, double daugther1_mass, double daugther2_mass, double daugther3_mass, double radi):
			hydra::BaseFunctor<Resonance<L>, hydra::complex<double>, 4>{c_re, c_im, mass, width},
			fLineShape(mass, width, mother_mass, daugther1_mass, daugther2_mass, daugther3_mass, radi)
	{}


    __hydra_dual__
	Resonance( Resonance<L> const& other):
	hydra::BaseFunctor<Resonance<L>, hydra::complex<double>, 4>(other),
	fLineShape(other.GetLineShape())
	{}

    __hydra_dual__  inline
	Resonance<L>&
	operator=( Resonance<L> const& other)
	{
		if(this==&other) return *this;

		hydra::BaseFunctor<Resonance<L>, hydra::complex<double>, 4>::operator=(other);
		fLineShape=other.GetLineShape();

		return *this;
	}

    __hydra_dual__  inline
	hydra::BreitWignerLineShape<L> const& GetLineShape() const {	return fLineShape; }

    __hydra_dual__  inline
	hydra::complex<double> Evaluate(unsigned int n, double* p)  const {


		double s12 = hydra::get<0>(p);
		double s13 = hydra::get<1>(p);
		
		fLineShape.SetParameter(0, _par[2]);
		fLineShape.SetParameter(1, _par[3]);

		hydra::complex<double> contrib_12 = fLineShape(s12);
		hydra::complex<double> contrib_13 = fLineShape(s13);

		auto r = hydra::complex<double>(_par[0], _par[1])*(contrib_12 + double(parity<L>::value)*contrib_13 ) ;

		return r;

	}
    

private:

	mutable hydra::BreitWignerLineShape<L> fLineShape;


};


int main(int argv, char** argc)
{   
    //E791
    double f0_980_MASS    = 0.990;
    double f0_980_GPP     = 0.02;
    double f0_980_GKK     = 4.5*0.02;
    double f0_980_WIDTH   = 0.04;
    double f0_980_re      = 1.;
    double f0_980_img     = 0.;

    double f0_1370_MASS  = 1.370;
    double f0_1370_WIDTH = .3;
    double f0_1370_re    = 0.75*cos(torad(198));
    double f0_1370_img = 0.75*sin(torad(198));

    double rho770_MASS   = .77549;
    double rho770_WIDTH  = .1491;
    double rho770_re     = 0.32*cos(torad(109));
    double rho770_img    = 0.32*sin(torad(109));
    
    double rho1450_MASS   = 1.465 + 5*0.025;
    double rho1450_WIDTH  = 0.4 + 3*0.06;
    double rho1450_re     = 0.28*cos(torad(162));
    double rho1450_img    = 0.28*sin(torad(162));
   
    double omega_MASS   = 0.78265;
    double omega_WIDTH  = 0.00849;
    double omega_re     = 1.;
    double omega_img    =   0.;

    double f2_1270_MASS     = 1.2751;
    double f2_1270_WIDTH    = 0.1851;
    double f2_1270_re       = 0.59*cos(torad(133));
    double f2_1270_img      = 0.59*sin(torad(133));

    double D_MASS         = 1.96834;
    double PI_MASS        = 0.13957061;// pi mass

    auto mass       = hydra::Parameter("F0_980_MASS",f0_980_MASS,0.0001);
    auto width      = hydra::Parameter("F0_980_WIDTH",f0_980_WIDTH,0.0001);
    auto coef_re    = hydra::Parameter("F0_980_COEF_RE",f0_980_re,0.0001);
    auto coef_im    = hydra::Parameter("F0_980_COEF_IM",f0_980_img,0.0001);

    Resonance<hydra::SWave> F0980_Resonance(coef_re, coef_im, mass, width, D_MASS, PI_MASS, PI_MASS, PI_MASS , 5.0);

    mass            = hydra::Parameter("F0_1370_MASS",f0_1370_MASS,0.0001);
    width           = hydra::Parameter("F0_1370_WIDTH",f0_1370_WIDTH,0.0001);
    coef_re         = hydra::Parameter("F0_1370_COEF_RE",f0_1370_re,0.0001);
    coef_im         = hydra::Parameter("F0_1370_COEF_IM",f0_1370_img,0.0001);

    Resonance<hydra::SWave> F01370_Resonance(coef_re, coef_im, mass, width, D_MASS, PI_MASS, PI_MASS, PI_MASS , 5.0);

    mass            = hydra::Parameter("rho_770_MASS",rho770_MASS,0.0001);
    width           = hydra::Parameter("rho_770_WIDTH",rho770_WIDTH,0.0001);
    coef_re         = hydra::Parameter("rho_770_COEF_RE",rho770_re,0.0001);
    coef_im         = hydra::Parameter("rho_770_COEF_IM",rho770_img,0.0001);

    Resonance<hydra::PWave> RHO770_Resonance(coef_re, coef_im, mass, width, D_MASS, PI_MASS, PI_MASS, PI_MASS , 5.0);

    mass            = hydra::Parameter("rho_1450_MASS",rho1450_MASS,0.0001);
    width           = hydra::Parameter("rho_1450_WIDTH",rho1450_WIDTH,0.0001);
    coef_re         = hydra::Parameter("rho_1450_COEF_RE",rho1450_re,0.0001);
    coef_im         = hydra::Parameter("rho_1450_COEF_IM",rho1450_img,0.0001);

    Resonance<hydra::PWave> RHO1450_Resonance(coef_re, coef_im, mass, width, D_MASS, PI_MASS, PI_MASS, PI_MASS , 5.0);

    mass            = hydra::Parameter("omega_782_MASS",omega_MASS,0.0001);
    width           = hydra::Parameter("omega_782_WIDTH",omega_WIDTH,0.0001);
    coef_re         = hydra::Parameter("omega_782_COEF_RE",omega_re,0.0001);
    coef_im         = hydra::Parameter("omega_782_COEF_IM",omega_img,0.0001);

    Resonance<hydra::PWave> OMEGA782_Resonance(coef_re, coef_im, mass, width, D_MASS, PI_MASS, PI_MASS, PI_MASS , 5.0);

    mass            = hydra::Parameter("f2_1270_MASS",f2_1270_MASS,0.0001);
    width           = hydra::Parameter("f2_1270_WIDTH",f2_1270_WIDTH,0.0001);
    coef_re         = hydra::Parameter("f2_1270_COEF_RE",f2_1270_re,0.0001,true);
    coef_im         = hydra::Parameter("f2_1270_COEF_IM",f2_1270_img,0.0001,true);

    Resonance<hydra::DWave> F21270_Resonance(coef_re, coef_im, mass, width, D_MASS, PI_MASS, PI_MASS, PI_MASS , 5.0);

    //parametric lambda
	auto Norm = hydra::wrap_lambda( [] __hydra_dual__ ( unsigned int n, hydra::complex<double>* x){

				hydra::complex<double> r(0,0);

				for(unsigned int i=0; i< n;i++)	r += x[i];

				return hydra::norm(r);
	});

	//model-functor
	auto Model = hydra::compose(Norm,
		    F0980_Resonance,
			F01370_Resonance,
			RHO770_Resonance,
			RHO1450_Resonance,
            OMEGA782_Resonance,
			F21270_Resonance
			 );

    auto Model_PDF = hydra::make_pdf( Model,
				hydra::PhaseSpaceIntegrator<3, hydra::device::sys_t>(D_MASS, {PI_MASS, PI_MASS, PI_MASS}, 500000));
    
    std::cout << "-----------------------------------------"<<std::endl;
	std::cout <<"| Initial PDF Norm: "<< Model_PDF.GetNorm() << "̣ +/- " <<   Model_PDF.GetNormError() << std::endl;
	std::cout << "-----------------------------------------"<<std::endl;

    std::string file = "../../hydra_charm/Ds3pi_toyMC.root";
    std::cout << "-----------------------------------------"<<std::endl;
	std::cout <<"| Getting Data: "<< file << std::endl;
	std::cout << "-----------------------------------------"<<std::endl;

    double _s12=0, _s13=0;
    TFile f(file.c_str(),"READ");
    TTree *t = (TTree*)f.Get("DecayTree");
    t->SetBranchAddress("s12_pipi_DTF",&_s12);
    t->SetBranchAddress("s13_pipi_DTF",&_s13);

    hydra::multiarray<double,2, hydra::device::sys_t> particles;
    
    for(int i = 0 ; i < 200000; i++){
        particles.push_back(hydra::make_tuple(_s12,_s13));
    }

    f.Close();

    std::cout << "-----------------------------------------"<<std::endl;
	std::cout <<"| Load NEntries: "<< particles.size() << std::endl;
	std::cout << "-----------------------------------------"<<std::endl;

    auto fcn = hydra::make_loglikehood_fcn(Model_PDF, particles.begin(),particles.end());

    ROOT::Minuit2::MnPrint::SetLevel(3);
	hydra::Print::SetLevel(hydra::WARNING);

	//minimization strategy
	MnStrategy strategy(2);

	//create Migrad minimizer
	MnMigrad migrad_d(fcn, fcn.GetParameters().GetMnState() ,  strategy);

	//print parameters before fitting
	std::cout<<fcn.GetParameters().GetMnState()<<std::endl;

	//Minimize and profile the time
	auto start_d = std::chrono::high_resolution_clock::now();

	FunctionMinimum minimum_d =  FunctionMinimum( migrad_d(5000,250) );

	auto end_d = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;

	//time
	std::cout << "-----------------------------------------"<<std::endl;
	std::cout << "| [Migrad] Time (ms) ="<< elapsed_d.count() <<std::endl;
	std::cout << "-----------------------------------------"<<std::endl;

	//print parameters after fitting
	std::cout<<"minimum: "<<minimum_d<<std::endl;
}
