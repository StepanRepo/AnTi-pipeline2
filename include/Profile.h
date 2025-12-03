#ifndef PROFILE_H
#define PROFILE_H

#include <memory>
#include <string>
#include <vector>
#include <complex>

#include "BaseReader.h"
#include "BaseHeader.h"
#include "PSRFITS_Writer.h"


class Profile 
{
	private:
		void check_incoherent(size_t nchann);
		void check_coherent();

		void matched_filter(double* data, size_t N, double threshold, std::vector<size_t>& pos, std::vector<double>& power);
		std::string csv_result (size_t left, size_t right, double power, size_t n_DM, std::vector<size_t>& pulses_dm0, std::vector<double>& power_dm0) const;


	public:
		BaseReader* reader;
		BaseHeader* hdr;

		double *raw;
		double *dyn;
		double *sum;
		double *fr;
		double *mask;

		double redshift;
		size_t sumidx;

		bool save_raw, save_dyn, save_sum;
		std::string output_dir;

		// Construct from filename + format
		Profile(const std::string& filename, 
				const std::string& format, 
				size_t buffer_size = 1024 * 1024 * 1024, // Standard size: 1 GiB
				bool save_raw_in = false, 
				bool save_dyn_in = false, 
				bool save_sum_in = false,
				std::string output_dir = "."
				);

		// Forward fill functions
		size_t fill_2d(double *dyn_spec, size_t& nchann, size_t& buf_pos, size_t& buf_max, size_t& buf_size);
		size_t fill_1d(double *vec, size_t& buf_pos, size_t& buf_max, size_t& buf_size);

		void dedisperse_incoherent (double DM, size_t nchann);
		void dedisperse_coherent (double DM, size_t nchann);

		std::string dedisperse_incoherent_stream (double DM, size_t nchann);
		std::string dedisperse_coherent_stream (double DM, size_t nchann);

		std::string dedisperse_incoherent_search (double DM, size_t nchann, double BL_window = 10e-6, double threshold = 5.0);
		std::string dedisperse_coherent_search   (double DM, size_t nchann, double BL_window = 10e-6, double threshold = 5.0);

		void create_mask(size_t nchann, double sig_threshold, double tail_threshold, size_t max_len = 0);

		void fold_dyn(double P, size_t nchann);
		void fold_dyn(std::string pred_file, size_t nchann);

		double get_redshift (std::string par_path, std::string site);

		// Access header info
		BaseHeader* getHeader();
};

#endif
