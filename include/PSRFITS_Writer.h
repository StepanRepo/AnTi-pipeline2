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
		PSRFITS_Writer(Profile& profile_in, std::string filename);

		/**
		 * @brief Destructor.
		 */
		~PSRFITS_Writer();


		/**
		 * @brief Create the primary HDU with basic metadata.
		 * @param header The BaseHeader object.
		 * @return True if successful.
		 */
		bool createPrimaryHDU(std::string obs_mode);
		bool append_subint_fold(double *data_double, const size_t nbin, const size_t nchan, const size_t npol) ;
		bool append_subint_stream(std::string stream_file, const size_t nchan, const size_t npol, bool cmp = false);
		bool append_subint_search(double *data_double, const size_t nbin, const size_t nchan, const size_t npol, bool cmp = false);

	private:
		fitsfile* fptr; // CFITSIO file pointer
		int status;     // CFITSIO status code


		Profile *profile;
		BaseHeader *header;

		void check_status(std::string operation);


};

#endif // PSRFITS_WRITER_H
