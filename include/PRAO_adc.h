/*
 * PRAO_adc.h
 *
 * Header file for the PRAO_adc class.
 * This class is designed to read and decode data from PRAO ADC files.
 * It inherits from BaseReader<ADCHeader> to provide a standardized interface for processing classes.
 * The header structure itself is defined in ADCHeader, which inherits from BaseHeader<ADCHeader>.
 */

#ifndef PRAO_ADC_H
#define PRAO_ADC_H

#include <fstream>      // For std::ifstream, file operations
#include <vector>       // For std::vector
#include <string>       // For std::string
#include <cstdint>      // For fixed-width integer types (uint32_t, uint8_t, etc.)
#include <iostream>     // For std::cout, std::cerr, std::endl
#include <iomanip>      // For std::setprecision
#include <fftw3.h>      // For FFTW library types (fftw_complex, fftw_plan)
#include <complex>      // For std::complex
#include <cstring>      // For std::memcpy, std::memmove, std::strlcpy
#include <stdexcept>    // For std::runtime_error, std::invalid_argument, std::out_of_range
#include <algorithm>    // For std::min, std::fill_n

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
class ADCHeader : public BaseHeader 
{
	public:
		std::string start_date_s;
		std::string start_utc_s;

		ADCHeader();
		void decode(const char* h_buff);
		void print() const override; // implement in .cpp
};

/*
 * PRAO_adc class
 * Implements the logic for reading, buffering, and processing data from a PRAO ADC file.
 * It inherits from BaseReader<ADCHeader> using CRTP, providing the required interface
 * for processing classes and allowing direct access to the ADCHeader members via get_header().
 */
class PRAO_adc : public BaseReader
{ 
	private:
		// Handles conversion from int8_t raw data to double.
		int8_t* raw_data;
		bool fill_buffer() override;
	public:
		// Public member to hold the header information (read from the beginning of the file)
		ADCHeader header;

		// Constructor: Opens the file, reads and decodes the header 
		// to initialize internal structures.
		// buffer_size: Size of the main processing buffer in bytes (default is 1 GiB).
		PRAO_adc(const std::string& filename_in, 
				size_t buffer_size = 1 << 30);

		// Destructor: Cleans up allocated memory and closes the file.
		~PRAO_adc() override;

		// Handles conversion from voltage to oservable 
		// power spectrum -- dynamic spectrum for 
		// incoherent dedispersion
		void fill_2d(std::vector<double>& dyn_spec, size_t freq_num) override;

		// Handles conversion from voltage to oservable 
		// complex spectrum -- complex dynamic spectrum
		// for coherent dedispersion
		void fill_1d(std::vector<std::complex<double>>& dyn_spec, size_t freq_num) override;
};

#endif // PRAO_ADC_H
