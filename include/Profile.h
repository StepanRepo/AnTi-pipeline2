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

		double *dyn;
		fftw_complex *dync;
		double *sum;
		double *fr;
		double *mask;
		double *int_prf;

		double redshift;

		// Construct from filename + format
		Profile(const std::string& filename, 
				const std::string& format, 
				size_t buffer_size = 1024 * 1024 * 1024 // Standard size: 1 GiB
				);

		// Forward fill functions
		size_t fill_2d(double *dyn_spec, size_t time_steps, size_t freq_num);
		size_t fill_1d(fftw_complex *vec, size_t n);

		void dedisperse_incoherent (double DM, size_t nchann);
		void dedisperse_coherent (double DM, size_t nchann);

		void dedisperse_incoherent_stream (double DM, size_t nchann);
		void dedisperse_coherent_stream (double DM, size_t nchann);

		void fold_dyn(double P, size_t nchann);
		void fold_dyn(std::string pred_file, size_t nchann);

		double get_redshift (std::string par_path);

		// Access header info
		BaseHeader* getHeader();
};

#endif
