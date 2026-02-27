#ifndef PROFILE_H
#define PROFILE_H

#include <cstddef>
#include <string>
#include <vector>
#include "tempo2pred.h"  // API for TEMPO2 prediction files
#include "tempo2.h"

#include "BaseReader.h"
#include "BaseHeader.h"


class Profile 
{
	private:
		void check_incoherent(size_t nchann);
		void check_coherent();

		void shift_window_coherent(fftw_plan fft, fftw_plan ifft, fftw_complex* f_space, fftw_complex* dphase, size_t nchann);

		void matched_filter(double* data, size_t N, double threshold, std::vector<size_t>& pos, std::vector<double>& power);
		std::string csv_result (size_t left, size_t right, double power, size_t num) const;


		// Helper functions
		void detect(fftw_complex* t_space, double* sum, size_t nchann);
		void shift_window_incoherent(const double* in, double* out, const int* shift, const size_t nchann, const size_t obs_window, const double* mask = nullptr);
		void calc_shift_phase(fftw_complex *dphase, double DM, double fcomp, double *freqs, size_t nchann, double beta = 0.0);
		void calc_shift_int(int *shift, double DM, double fcomp, double *freqs, size_t nchann, double tau, double beta = 0.0);


	T2Predictor *pred;
	pulsar *psr;

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

		std::string output_dir;
		int verbose;

		// Construct from filename + format
		Profile(const std::string& filename, 
				const std::string& format, 
				size_t buffer_size = 1024 * 1024 * 1024, // Standard size: 1 GiB
				std::string output_dir = ".",
				int verbose = 0
				);

		// Virtual destructor
		Profile(const Profile&) = delete;
		Profile& operator=(const Profile&) = delete;
		~Profile();

		// Helper functions
		void fill_PSR();
		void fill_SEARCH();
		size_t fill_2d(double *dyn_spec, size_t& nchann, size_t& buf_pos, size_t& buf_max, size_t& buf_size);
		size_t fill_1d(double *vec, size_t& buf_pos, size_t& buf_max, size_t& buf_size);
		void skip(const double t);
		void set_limit(const double t);

		void load_predictor(std::string filename);
		void load_psr(std::string filename, std::string site);

		double get_redshift (std::string par_path, std::string site);
		void normilize(double BL_window = -1.0);

		void save_raw(std::string mode, std::string stream_file = "");
		void save_dyn(std::string mode, std::string stream_file = "");
		void save_sum(std::string mode, std::string stream_file = "");
		void save_filt();

		// Functions for streaming profile processing
		std::string dedisperse_incoherent_stream (
				double DM, size_t nchann, 
				bool is_save_raw = false, bool is_save_dyn = false, bool is_save_sum = false
				);
		std::string dedisperse_coherent_stream (
				double DM, size_t nchann,
				bool is_save_raw = false, bool is_save_dyn = false, bool is_save_sum = false
				);


		// Functions for time-domain single-pulse search
		std::string dedisperse_incoherent_search (
				double DM, 
				size_t nchann, 
				double BL_window = 10e-6, 
				double threshold = 5.0, 
				std::string conv_type = "", 
				double fwhm = 0.0);
		std::string dedisperse_coherent_search   (
				double DM, 
				size_t nchann, 
				double BL_window = 10e-6, 
				double threshold = 5.0, 
				std::string conv_type = "", 
				double fwhm = 0.0);


		// Functions for timing purposes
		void accumulate_prf(Profile& other, std::string t2_pred_file = "");
		void finish_accumulation();


		void get_toa (const Profile& tpl, long double *toa, double *toa_err);


		// Functions for channel filtration
		void load_mask(size_t nchann);
		void create_mask(size_t nchann, double sig_threshold, double tail_threshold, size_t max_len = 0, size_t downsample = 0);


		// Functions for profile folding
		void fold_dyn(double P, size_t nchann);
		void fold_dyn(std::string pred_file, size_t nchann);

		void dedisperse_incoherent (double DM, size_t nchann);
		void dedisperse_coherent (double DM, size_t nchann);


		// Access header info
		BaseHeader* getHeader();
};

#endif

