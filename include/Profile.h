#ifndef PROFILE_H
#define PROFILE_H

#include <memory>
#include <string>
#include <vector>
#include <complex>

#include "BaseReader.h"
#include "BaseHeader.h"


class Profile 
{
	private:


	public:
		std::unique_ptr<BaseReader> reader;

		std::vector<double> dyn;
		std::vector<std::complex<double>> dync;
		std::vector<double> sum;
		std::vector<double> fr;
		std::vector<double> mask;
		std::vector<double> int_prf;

		// Construct from filename + format
		Profile(const std::string& filename, 
				const std::string& format, 
				size_t buffer_size = 1024 * 1024 * 1024 // Standard size: 1 GiB
				);

		// Forward fill functions
		void fill_2d(std::vector<double>& dyn_spec, size_t freq_num);
		void fill_1d(std::vector<std::complex<double>>& dyn_spec, size_t freq_num);

		void dedisperse_incoherent (double DM);
		void dedisperse_coherent (double DM);

		void fold(double P);

		// Access header info
		BaseHeader* getHeader();
};

#endif
