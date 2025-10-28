// main.cpp
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include "Profile.h" 

int main() 
{

	//std::string filename = "Crab_pulse/r6326e_bv_326-1702_ch01.vdif_7183.vdif";
	std::string filename = "rup106_bd_321-1252_ch07.vdif";
	std::string format = "IAA_vdif";

	Profile profile(filename, format); 


	// Access common header info via polymorphic interface
	BaseHeader* hdr = profile.getHeader();
	if (!hdr) 
		throw std::runtime_error("Header not available");


	size_t freq_num = 1<<11;
	//size_t time_step = 1000000;
	//std::vector<double> dyn(freq_num * time_step);

	hdr->sampling = hdr->sampling * 2;
	hdr->nchann = freq_num;
    hdr->fmax = 2675.8;
    hdr->fmin = 2675.8 - 512.0;
    hdr->period = 0.0333924123;

	hdr->print();
	profile.dedisperse_coherent(56.7712);

	// After computing dynamic_spectrum
	//std::ofstream output("data.bin", std::ios::binary);
	//output.write(reinterpret_cast<const char*>(profile.dyn.data()),
	//		profile.dyn.size() * sizeof(double));
	//output.close();



	return 0;
}
