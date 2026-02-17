#include "BaseReader.h"
#include "fftw3.h"
#include "aux_math.h"

#include <cstddef>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <string>

BaseReader::~BaseReader()
{
	// The only thing to delete is header_ptr
	// But if we define custom header in an 
	// inherited class as an object, it is 
	// deleted automatically
}

// Common implementation of the resetting function
// It may be applied to files with no internal data
// structurization. Raw file position is used to 
// navigate through file
void BaseReader::reset()
{
	if (!file.is_open())
		throw std::runtime_error("File is not open");

	// Reset the file state
	file.clear();
	file.seekg(data_start_pos, std::ios::beg);

	// Reset the buffer state
	buf_pos = 0;
	buf_max = 0;
}	


size_t BaseReader::fill_2d_baseband_re(double* dyn_spec, size_t time_steps, size_t freq_num)
{
    // Each FFT processes a chunk of 2 * freq_num real samples
    size_t samples_per_chunk = 2 * freq_num;
    double* chunk_start = nullptr; // Pointer to the start of the current processing chunk
    double re, im; 
	size_t filled = 0;

	thread_local static fftw_complex *fft_arr;
	thread_local static fftw_plan p;
	thread_local static size_t alloc_size = 0;

    // Initialize FFTW plan and output array if they haven't been created yet
    if (fft_arr == nullptr || alloc_size != freq_num || !p)  
	{
		if (fft_arr)
			fftw_free(fft_arr);


        // Allocate memory for FFTW's output array (freq_num + 2 complex numbers for R2C FFT - safer size)
        fft_arr = (fftw_complex*)(fftw_malloc(sizeof(fftw_complex) * (freq_num + 1)));


        // Create the FFTW plan for a real-to-complex FFT of size samples_per_chunk
        p = fftw_plan_dft_r2c_1d(samples_per_chunk, chunk_start, fft_arr, FFTW_ESTIMATE);

        if (!p || !fft_arr) 
		{
            fftw_free(fft_arr);
            throw std::runtime_error("Failed to create FFTW plan for fill_2d");
        }
		else 
		{
			alloc_size = freq_num;
		}

    }

    // Process the buffer in chunks
    for (size_t chunk = 0; chunk < time_steps; ++chunk) 
	{
        // Check if the current chunk fits within the currently filled part of the buffer
		if (buf_pos + samples_per_chunk > buf_max) 
		{
			// If not, try to fill the buffer with more data from the file
			if (!fill_buffer()) 
				// If fill_buffer fails (e.g., EOF reached), stop processing
				break;
		}

        // Set the pointer to the start of the current chunk within the buffer
        chunk_start = buffer + buf_pos;

        // Execute the FFT on the current chunk
        fftw_execute(p);

        // Advance the buffer read position by the chunk size
        buf_pos += samples_per_chunk;

        // Calculate the power spectrum from the complex FFT output and store it
		#pragma omp simd
        for (size_t k = 0; k < freq_num; ++k) 
		{
            // Access the k+1 element of the FFT output (skip DC component at index 0 often)
            re = fft_arr[k+1][0]; // Real part
            im = fft_arr[k+1][1]; // Imaginary part
            // Store the power (magnitude squared)
            dyn_spec[chunk * freq_num + k] = re*re + im*im;
        }

		filled += 1;
    }

	return filled;
}


size_t BaseReader::fill_2d_baseband_cmplx(double* dyn_spec, size_t time_steps, size_t freq_num)
{

    size_t samples_per_chunk = 2 * freq_num;
    fftw_complex* chunk_start = nullptr; // Pointer to the start of the current processing chunk
    double re, im; 
	size_t filled = 0;

	static fftw_complex *fft_arr;
	static fftw_plan p;
	static size_t alloc_size = 0;

    // Initialize FFTW plan and output array if they haven't been created yet
    if (fft_arr == nullptr || alloc_size != freq_num || !p)  
	{
		if (fft_arr)
			fftw_free(fft_arr);


        // Allocate memory for FFTW's output array (freq_num + 2 complex numbers for R2C FFT - safer size)
        fft_arr = (fftw_complex*)(fftw_malloc(sizeof(fftw_complex) * (freq_num)));


        // Create the FFTW plan for a real-to-complex FFT of size samples_per_chunk
        p = fftw_plan_dft_1d(freq_num, chunk_start, fft_arr, FFTW_FORWARD, FFTW_ESTIMATE);

        if (!p || !fft_arr) 
		{
            fftw_free(fft_arr);
            throw std::runtime_error("Failed to create FFTW plan for fill_2d");
        }
		else 
		{
			alloc_size = freq_num;
		}

    }

    // Process the buffer in chunks
    for (size_t chunk = 0; chunk < time_steps; ++chunk) 
	{
        // Check if the current chunk fits within the currently filled part of the buffer
		if (buf_pos + samples_per_chunk > buf_max) 
		{
			// If not, try to fill the buffer with more data from the file
			if (!fill_buffer()) 
				// If fill_buffer fails (e.g., EOF reached), stop processing
				break;
		}

        // Set the pointer to the start of the current chunk within the buffer
        chunk_start = (fftw_complex*) (buffer + buf_pos);

        // Execute the FFT on the current chunk
        fftw_execute(p);

        // Advance the buffer read position by the chunk size
        buf_pos += samples_per_chunk;

        // Calculate the power spectrum from the complex FFT output and store it
		#pragma omp simd
        for (size_t k = 0; k < freq_num; ++k) 
		{
            // Access the k+1 element of the FFT output (skip DC component at index 0 often)
            re = fft_arr[k+1][0]; // Real part
            im = fft_arr[k+1][1]; // Imaginary part
            // Store the power (magnitude squared)
            dyn_spec[chunk * freq_num + k] = re*re + im*im;
        }

		filled += 1;
    }

	return filled;
}

size_t BaseReader::fill_2d_spectrum(double* dyn_spec, size_t time_steps, size_t freq_num)
{
	size_t npol, nchann;
	size_t samples_per_chunk;
	size_t filled, to_fill;

	filled = 0;
	npol = header_ptr->npol;
	nchann = header_ptr->nchann;
	samples_per_chunk = nchann*npol;

	if (nchann != freq_num) 
		throw std::invalid_argument(
				"The number of requested channels is not equal to the recorded number" + 
				std::to_string(nchann));

	while (filled < time_steps)
	{

		if (buf_pos + samples_per_chunk > buf_max) 
		{
			// If not, try to fill the buffer with more data from the file
			if (!fill_buffer()) 
				// If fill_buffer fails (e.g., EOF reached), stop processing
				break;
		}

		to_fill = (buf_max - buf_pos) / samples_per_chunk;
		to_fill = std::min(to_fill, time_steps  - filled);

		math::vec_copy(
				dyn_spec + filled*samples_per_chunk,
				buffer + buf_pos, 
				to_fill * samples_per_chunk);

		filled += to_fill;
		buf_pos += to_fill * samples_per_chunk;
	}


	return filled;
}


// Common method implementation: fill_2d
// Depending of type of processed data its outputs dynamic 
// spectrum. If the data is baseband recorded performs FFT.
// Otherwise filles spectrum directly from the file
size_t BaseReader::fill_2d(double* dyn_spec, size_t time_steps, size_t freq_num) 
{

	if (header_ptr->nchann == 1 && !(header_ptr->cmplx))
	{
		return fill_2d_baseband_re(dyn_spec, time_steps, freq_num);
	}
	else if (header_ptr->nchann == 1 && header_ptr->cmplx) 
	{
		return fill_2d_baseband_cmplx(dyn_spec, time_steps, freq_num);
	}
	else if (header_ptr->nchann > 1) 
	{
		return fill_2d_spectrum(dyn_spec, time_steps, freq_num);
	}

	return 0;
}



size_t BaseReader::fill_1d(double *vec, size_t n) 
{

	size_t i = 0;
	size_t available;
	size_t remaining;
	size_t chunk;
	double* buf_ptr;
	double* vec_ptr; 

	while (i < n) 
	{
		// Ensure buffer has data
		if (buf_pos >= buf_max) 
		{
			fill_buffer();
			if (buf_pos >= buf_max) break; // no more data
		}

		// Determine how many elements we can copy without refilling
		available = buf_max - buf_pos;
		remaining = n - i;
		chunk = std::min(available, remaining);

		// Vectorizable loop: no conditionals, just assignments
		buf_ptr = buffer + buf_pos;
		vec_ptr = vec + i;

		std::memcpy(vec_ptr, buf_ptr, chunk * sizeof(double));

		buf_pos += chunk;
		i += chunk;
	}

	return i;
}

bool BaseReader::fill_wts(double* mask, size_t freq_num)
{
	throw std::runtime_error("This format doesn't support channels' weights storage");
	return 0;
}
