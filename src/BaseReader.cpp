#include "BaseReader.h"

#include <iostream>
#include <cstring>

void BaseReader::reset()
{
	if (!file.is_open())
		throw std::runtime_error("File is not open");

	// Reset the file state
	file.clear();
	file.seekg(data_start_pos, std::ios::beg);

	// Delete Fourier information
	if (fft_arr)
	{
		fftw_destroy_plan(p);
		fftw_free(fft_arr);
		fft_arr = nullptr;
	}

	// Reset the buffer state
	buf_pos = 0;
	buf_max = 0;
}	

// Public method implementation: fill_2d
// Processes the data in the main buffer using FFTs to generate a 2D dynamic spectrum (power).
size_t BaseReader::fill_2d(double* dyn_spec, size_t time_steps, size_t freq_num) 
{

    // Each FFT processes a chunk of 2 * freq_num real samples
    size_t samples_per_chunk = 2 * freq_num;

    double* chunk_start = nullptr; // Pointer to the start of the current processing chunk
    double re, im; // Variables to hold real and imaginary parts of FFT output
	size_t filled = 0;

    // Initialize FFTW plan and output array if they haven't been created yet
    if (fft_arr == nullptr) 
	{
        // Allocate memory for FFTW's output array (freq_num + 2 complex numbers for R2C FFT - safer size)
        fft_arr = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (freq_num + 2)));


        // Allocate a temporary input buffer JUST FOR PLANNING
        // as planning overwrites the input array
        double* plan_input = static_cast<double*>(fftw_malloc(sizeof(double) * samples_per_chunk));

        // Initialize with zeros
        std::fill_n(plan_input, samples_per_chunk, 0.0);

        // Create the FFTW plan for a real-to-complex FFT of size samples_per_chunk
        // Uses FFTW_MEASURE for potentially better performance at the cost of initialization time
        p = fftw_plan_dft_r2c_1d(samples_per_chunk, plan_input, fft_arr, FFTW_ESTIMATE);
        // Clean up scratch buffer -- plan is now independent
        fftw_free(plan_input);

        if (!p) 
		{
            fftw_free(fft_arr);
            throw std::runtime_error("Failed to create FFTW plan for fill_2d");
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

			// After filling, check again if the chunk fits
			if (buf_pos + samples_per_chunk > buf_max) 
				// Even after refilling, the chunk doesn't fit. Likely end of data.
				break;
		}

        // Set the pointer to the start of the current chunk within the buffer
        chunk_start = buffer + buf_pos;

        // Execute the FFT on the current chunk
        fftw_execute_dft_r2c(p, chunk_start, fft_arr);

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

size_t BaseReader::fill_1d(fftw_complex *__restrict__ vec, size_t n) 
{
	size_t i = 0;
	size_t available;
	size_t remaining;
	size_t chunk;
	double *__restrict__ buf_ptr;
	fftw_complex* vec_ptr; 

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

		#pragma omp simd
		for (size_t j = 0; j < chunk; ++j) 
		{
			vec_ptr[j][0] = buf_ptr[j]; // real part
			vec_ptr[j][1] = 0.0;        // imaginary part
		}

		buf_pos += chunk;
		i += chunk;
	}

	return i;
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
