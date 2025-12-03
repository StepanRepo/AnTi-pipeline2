// BaseReader.h
#ifndef BASE_READER_H
#define BASE_READER_H

#include "BaseHeader.h"
#include <string>
#include <fstream>
#include <vector>
#include <complex>
#include <fftw3.h>      // For FFTW library types (fftw_complex, fftw_plan)

class BaseReader 
{
	protected:

		double* buffer;                 // Main processing buffer holding decoded data (double)
		size_t buf_pos;                 // Current read position within the main buffer
		size_t buf_max;                 // Number of valid samples currently in the main buffer
		size_t buf_size;                // Total allocated size of the main buffer (in samples)
										//
		std::streamoff data_start_pos;	// Position in the file where reader should start to read data (either beginning of the data section or is set by skip(t0)) 


		// FFT members for processing
		fftw_complex *fft_arr = nullptr;          // FFTW output array for complex FFT results
		fftw_plan p{};                    // FFTW plan for performing the real-to-complex FFT
										
		// Pure virtual methods — must be implemented by derived classes
		virtual bool fill_buffer() = 0;

	public:

		// File handling members
		std::string filename = "";
		std::ifstream file;             // Input file stream for the ADC file
		bool is_open = false;



		// Pointer to a polymorphic header object
		BaseHeader* header_ptr = nullptr;

		size_t fill_1d(fftw_complex* vec, size_t n);
		size_t fill_1d(double* vec, size_t n);

		size_t fill_2d(double *dyn_spec, size_t time_steps, size_t freq_num);

		virtual void reset();

		// Virtual destructor
		virtual ~BaseReader() = default;

		// Pure virtual methods — must be implemented by derived classes
		virtual double point2time(size_t point) = 0;
		virtual void skip(double sec) = 0;
		virtual void set_limit(double t) = 0;
};

#endif // BASE_READER_H
