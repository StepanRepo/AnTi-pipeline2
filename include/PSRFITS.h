#ifndef PSRFITS_H
#define PSRFITS_H



extern "C" {
#include <fitsio.h>
}

#include <string>       // For std::string
#include <cstdint>      // For fixed-width integer types (uint32_t, uint8_t, etc.)
#include <fftw3.h>      // For FFTW library types (fftw_complex, fftw_plan)
#include <cstring>      // For std::memcpy, std::memmove, std::strlcpy

// Include the base class templates.
#include "BaseReader.h" // Defines BaseReader
#include "BaseHeader.h" // Defines BaseHeader
						//
// Auxilarry functions
long double ADCTime2MJD(std::string const time_in);
void strip_white(std::string& line);
void str_split(char* buffer_c, std::string& name, std::string& value);

/*
 * ADCHeader struct
 * Represents the decoded fields of a PRAO ADC header block.
 * It inherits from BaseHeader
 * to its members from processing classes while maintaining a polymorphic interface.
 */
class PSRFITSHeader : public BaseHeader 
{
	public:
		std::string MODE;
		size_t nsblk;
		size_t nbits;
		int sign;

		PSRFITSHeader();
		void fill(fitsfile *fptr, int *status);
		//void print() const override;

};

class PSRFITS : public BaseReader
{ 
	private:
		fitsfile *fptr;
		int status;

		size_t subint_index, subint_pos;
		size_t start_subint_index, start_subint_pos;

		int8_t* raw_data;
		bool fill_buffer() override;
		void check_status(std::string);
	public:
		// Public member to hold the header information (read from the beginning of the file)
		PSRFITSHeader header;

		// Constructor: Opens the file, reads and decodes the header 
		// to initialize internal structures.
		// buffer_size: Size of the main processing buffer in bytes (default is 1 GiB).
		PSRFITS(const std::string& filename_in, 
				size_t buffer_size = 1 << 30);

		// Destructor: Cleans up allocated memory and closes the file.
		~PSRFITS() override;

		virtual double point2time(size_t point) override;
		virtual void skip(double sec) override;
		virtual void set_limit(double t) override; 
		virtual void reset() override; 
		virtual bool allow_1d() override;
		virtual bool allow_2d() override;
};

#endif // PSRFITS_H

