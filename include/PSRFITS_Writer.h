#ifndef PSRFITS_WRITER_H
#define PSRFITS_WRITER_H

#include "Profile.h"
#include "BaseHeader.h"

extern "C" {
#include <fitsio.h>
}

#include <memory>
#include <string>
#include <vector>



/**
 * @brief Writer class for creating PSRFITS files from Profile data.
 */
class PSRFITS_Writer 
{
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
		bool append_history(
				const size_t nsubint, 
				const size_t npol, 
				const size_t nchan, 
				const size_t nbin, 
				const double dm, 
				const double fmin, 
				const double fmax, 
				const double fcomp, 
				const double tau, 
				std::string dds_mtd = "", 
				const double* mask = nullptr);
		bool append_subint_fold(
				double *data_double,
				double *mask,
				const size_t nbin,
				const size_t nchan,
				const size_t npol,
				const double period,
				const double dm,
				const double fmin,
				const double fmax,
				const double fcomp,
				const double tau,
				std::string dds_mtd = ""
				);
		bool append_subint_stream(
				std::string stream_file, 
				double *mask, 
				const size_t nchan, 
				const size_t npol, 
				const double dm, 
				const double fmin, 
				const double fmax, 
				const double fcomp,
				const double tau, 
				std::string dds_mtd = "",
				const bool cmp = false);
		bool append_subint_search(
				double* data_double, 
				double *mask, 
				const size_t nbin, 
				const size_t nchan, 
				const size_t npol, 
				const double dm, 
				const double fmin, 
				const double fmax, 
				const double fcomp,
				const double tau, 
				std::string dds_mtd = "",
				bool cmp = false);

	private:
		fitsfile* fptr; // CFITSIO file pointer
		int status;     // CFITSIO status code


		void check_status(std::string operation);


};

#endif // PSRFITS_WRITER_H
