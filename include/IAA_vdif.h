/*
 * IAA_vdif.h
 *
 * Header file for the IAA_vdif class.
 * This class is designed to read and decode data from VDIF (VLBI Data Interchange Format) files.
 * It inherits from BaseReader<VDIFHeader> to provide a standardized interface for processing classes.
 * The header structure itself is defined in VDIFHeader, which inherits from BaseHeader<VDIFHeader>.
 *
 * Author: Stepan Andrianov
 * Date: 2025-10-26
 */

#ifndef IAA_VDIF_H
#define IAA_VDIF_H

#include <fstream>      // For std::ifstream, file operations
#include <vector>       // For std::vector
#include <string>       // For std::string
#include <cstdint>      // For fixed-width integer types (uint32_t, uint8_t, etc.)
#include <iostream>     // For std::cout, std::cerr, std::endl
#include <iomanip>      // For std::setprecision
#include <fftw3.h>      // For FFTW library types (fftw_complex, fftw_plan)
#include <complex>      // For std::complex
#include <cstring>      // For std::memcpy, std::memmove
#include <stdexcept>    // For std::runtime_error, std::invalid_argument, std::out_of_range

// Include the base class templates.
#include "BaseReader.h" // Defines BaseReader
#include "BaseHeader.h" // Defines BaseHeader

// Auxilarry functions
long double vdifTimeToMJD
(
    uint16_t half_year,         // Index into the precomputed MJD start array
    uint32_t seconds,           // Seconds since the reference epoch start
    long double fractional_sec = 0.0L // Fractional seconds 
);

/*
 * VDIFHeader struct
 * Represents the decoded fields of a VDIF header block.
 * It inherits from BaseHeader<VDIFHeader> using CRTP to allow direct access
 * to its members from processing classes while maintaining a polymorphic interface.
 */
class VDIFHeader : public BaseHeader
{ // Inherits from BaseHeader template
    // Raw header words read directly from the file (8 words * 4 bytes each = 32 bytes typically)
	public:
		uint32_t word0;
		uint32_t word1;
		uint32_t word2;
		uint32_t word3;
		uint32_t word4; // Present in non-legacy mode
		uint32_t word5; // Present in non-legacy mode
		uint32_t word6; // Present in non-legacy mode
		uint32_t word7; // Present in non-legacy mode

		// Decoded fields parsed from the raw words
		bool invalid_data;              // Bit 31 of word0: indicates if data is invalid
		bool legacy_mode;               // Bit 30 of word0: indicates if header is 16 or 32 bytes
		uint32_t seconds_from_ref_epoch; // Bits 0-29 of word0: seconds since reference epoch
		uint8_t ref_epoch;              // Bits 24-29 of word1: reference epoch index
		uint32_t frame_number;          // Bits 0-23 of word1: sequential frame number
		uint8_t version;                // Bits 29-31 of word2: VDIF version number
		uint8_t log2_channels;          // Bits 24-28 of word2: log2 of number of channels
		uint32_t num_channels;          // Calculated from log2_channels: number of channels
		uint32_t frame_length;          // Bits 0-23 of word2 * 8: frame length in bytes
		bool complex_data;              // Bit 31 of word3: indicates if data is complex
		uint8_t bits_per_sample;        // Bits 26-30 of word3 + 1: bits per sample
		uint16_t thread_id;             // Bits 16-25 of word3: thread identifier
		uint16_t station_id;            // Bits 0-15 of word3: station identifier
		uint8_t extended_data_version;  // Bits 24-31 of word4: extended data version
		bool uflag;                     // Bit 23 of word4: frequency unit flag (MHz vs kHz)
		uint32_t sync_pattern;          // word5: 32-bit sync pattern (e.g., 0xACABFEED)
		uint64_t das_id;                // word6 and word7 combined: Data Acquisition System ID
		long double t;                  // Calculated time based on VDIF fields
		size_t n_samples;               // Calculated number of samples in the data block

		VDIFHeader();

		void decode(const char* byte_array);
		void print() const override; // implement in .cpp
};

class IAA_vdif : public BaseReader
{ 
	private:

		// Data buffer members for raw and decoded data
		char* h_buffer; 			// Buffer to hold the raw 32-byte header
		char* raw_data;

		size_t n_read;

		// Private helper function to unpack N-bit integer data into double precision floating point.
		void unpack_nbit_to_double
			(
			 const char* input,          // Pointer to the input byte array containing packed bits
			 double* output,             // Pointer to the output array for double values
			 size_t num_elements,        // Number of elements to unpack
		 int n                       // Number of bits per sample
		);

    // Reads the next VDIF frame from the file, decodes its header, and appends its data to the main buffer.
    // Returns true if a frame was successfully read, false on EOF or error.
    bool read_frame();

    // Fills the main buffer by reading frames from the file until the buffer is full or EOF is reached.
    // Handles gaps in time between frames by padding with zeros.
    bool fill_buffer();

public:
    // Public member to hold the current frame's header information
    VDIFHeader header;

    // Constructor: Opens the file, reads the first header to initialize internal structures.
    // buffer_size: Size of the main processing buffer in bytes (default is 2 GiB).
    IAA_vdif(const std::string& filename_in, 
			size_t buffer_size = 1<<30 // Default buffer size is 1 GiB
			);

    // Destructor: Cleans up allocated memory and closes the file.
    ~IAA_vdif() override;



	// Transforms a number of points, used in the analisys
	// to corresponding time from the beginning of 
	// the section being processed 
	virtual double point2time(size_t point) override;

	virtual void skip(double sec) override;
	virtual void set_limit(double t) override; // implement in .cpp
};

#endif // IAA_VDIF_H
