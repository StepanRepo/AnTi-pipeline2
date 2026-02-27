#ifndef PSRFITS_WRITER_H
#define PSRFITS_WRITER_H

#include "BaseHeader.h"

extern "C" {
#include <fitsio.h>
}

#include <string>



/**
 * @brief Writer class for creating PSRFITS files from Profile data.
 */
class PSRFITS_Writer 
{

	private:
		fitsfile* fptr; // CFITSIO file pointer
		int status;     // CFITSIO status code

		BaseHeader hdr = {};

		void check_status(std::string operation);


	public:
		/**
		 * @brief Constructor.
		 * @param filename Output PSRFITS filename.
		 */
		PSRFITS_Writer(std::string filename);

		/**
		 * @brief Destructor.
		 */
		~PSRFITS_Writer();

		/**
		 * @brief Create the primary HDU with basic metadata.
		 * @param header The BaseHeader object.
		 * @return True if successful.
		 */
		bool createPrimaryHDU(std::string obs_mode, const BaseHeader* header);
		bool append_subint(double *data_double, double *mask);
		bool append_subint(std::string stream_file, double *mask);
		bool append_history(std::string dds_mtd = "", double* mask = nullptr);
		bool append_bandpass(double *fr);

		bool append_history(
				const size_t nsubint, 
				const size_t nbin, 
				const size_t npol, 
				const size_t nchan, 
				const double dm, 
				const double *freqs,
				const double fcomp, 
				const double tau, 
				std::string dds_mtd = "", 
				const double* mask = nullptr);


		bool append_subint_fold(
				double *data_double,
				double *dat_freq,
				double *mask,
				const size_t nbin,
				const size_t nchan,
				const size_t npol,
				const double period,
				const double dm,
				const double fcomp,
				const double tau,
				std::string dds_mtd = ""
				);


		bool append_subint_stream(
				std::string stream_file, 
				double *dat_freq,
				double *mask, 
				const size_t nchan, 
				const size_t npol, 
				const double dm, 
				const double fcomp,
				const double tau, 
				std::string dds_mtd = "",
				const bool cmp = false);


		bool append_subint_search(
				double* data_double, 
				double *dat_freq,
				double *mask, 
				const size_t nbin, 
				const size_t nchan, 
				const size_t npol, 
				const double dm, 
				const double fcomp,
				const double tau, 
				std::string dds_mtd = "",
				bool cmp = false);

};

#endif // PSRFITS_WRITER_H

