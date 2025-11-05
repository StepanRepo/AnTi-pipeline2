/*
 * IAA_vdif.cpp
 *
 * Implementation file for the IAA_vdif class.
 * Provides the definitions for the constructor, destructor,
 * helper functions, and all public/private methods declared in IAA_vdif.h.
 * This class handles VDIF file reading, decoding, buffering, and FFT processing.
 * The constructor now reads the first frame and allocates all necessary arrays.
 */

#include "IAA_vdif.h" // Include the header file defining the class interface and base classes
#include <stdexcept>  // For std::runtime_error, std::bad_alloc
#include <iostream>   // For std::cerr, std::endl
#include <iomanip>    // For std::setprecision
#include <cstring>    // For std::memcpy, std::memmove
#include <limits>     // For std::numeric_limits (if needed for validation)
#include <algorithm>  // For std::min (if needed for safer copying)
					  


// --- Helper Function Implementation ---

// Implementation of the helper function to convert VDIF time fields to MJD.
// This function is defined here because it's used within the header decoding logic.
long double vdifTimeToMJD(
		uint16_t half_year,
		uint32_t seconds,
		long double fractional_sec
		) 
{
	// Precomputed MJD start dates for each half-year period.
	// Index 0 corresponds to 2000-01-01, index 1 to 2000-07-01, etc.
	constexpr long double HALF_YEAR_MJD_START[] = {
		51544.0L,  // 2000-01-01
		51726.0L,  // 2000-07-01
		51910.0L,  // 2001-01-01
		52091.0L,  // 2001-07-01
		52275.0L,  // 2002-01-01
		52456.0L,  // 2002-07-01
		52640.0L,  // 2003-01-01
		52821.0L,  // 2003-07-01
		53005.0L,  // 2004-01-01
		53186.0L,  // 2004-07-01
		53370.0L,  // 2005-01-01
		53551.0L,  // 2005-07-01
		53735.0L,  // 2006-01-01
		53916.0L,  // 2006-07-01
		54100.0L,  // 2007-01-01
		54281.0L,  // 2007-07-01
		54465.0L,  // 2008-01-01
		54646.0L,  // 2008-07-01
		54830.0L,  // 2009-01-01
		55011.0L,  // 2009-07-01
		55195.0L,  // 2010-01-01
		55376.0L,  // 2010-07-01
		55560.0L,  // 2011-01-01
		55741.0L,  // 2011-07-01
		55925.0L,  // 2012-01-01
		56106.0L,  // 2012-07-01
		56290.0L,  // 2013-01-01
		56471.0L,  // 2013-07-01
		56655.0L,  // 2014-01-01
		56836.0L,  // 2014-07-01
		57020.0L,  // 2015-01-01
		57201.0L,  // 2015-07-01
		57385.0L,  // 2016-01-01
		57566.0L,  // 2016-07-01
		57750.0L,  // 2017-01-01
		57931.0L,  // 2017-07-01
		58115.0L,  // 2018-01-01
		58296.0L,  // 2018-07-01
		58480.0L,  // 2019-01-01
		58661.0L,  // 2019-07-01
		58845.0L,  // 2020-01-01
		59026.0L,  // 2020-07-01
		59210.0L,  // 2021-01-01
		59391.0L,  // 2021-07-01
		59575.0L,  // 2022-01-01
		59756.0L,  // 2022-07-01
		59940.0L,  // 2023-01-01
		60121.0L,  // 2023-07-01
		60305.0L,  // 2024-01-01
		60486.0L,  // 2024-07-01
		60670.0L,  // 2025-01-01
		60851.0L,  // 2025-07-01
		61035.0L,  // 2026-01-01
		61216.0L,  // 2026-07-01
		61400.0L,  // 2027-01-01
		61581.0L,  // 2027-07-01
		61765.0L,  // 2028-01-01
		61946.0L,  // 2028-07-01
		62130.0L,  // 2029-01-01
		62311.0L,  // 2029-07-01
		62495.0L,  // 2030-01-01
				   // ... extend as needed
	};

	constexpr size_t MAX_HALF_YEARS = sizeof(HALF_YEAR_MJD_START) / sizeof(HALF_YEAR_MJD_START[0]);

	// Validate the half_year index to prevent out-of-bounds access
	if (half_year >= MAX_HALF_YEARS) 
		throw std::out_of_range("half_year exceeds precomputed range in vdifTimeToMJD");

	long double mjd_base = HALF_YEAR_MJD_START[half_year];
	constexpr long double SECONDS_PER_DAY = 86400.0L; // Number of seconds in a day

	// Calculate the MJD by adding the base date, the seconds converted to days,
	// and the fractional seconds converted to days.
	return mjd_base + (static_cast<long double>(seconds) + fractional_sec) / SECONDS_PER_DAY;
}


// --- VDIFHeader Method Implementations ---

// Implementation of the VDIFHeader constructor.
// Initializes all member variables to their default values (0, false, etc.).
VDIFHeader::VDIFHeader(): 
	BaseHeader(),
	word0(0), word1(0), word2(0), word3(0), 
	word4(0), word5(0), word6(0), word7(0),
	invalid_data(false), legacy_mode(false), 
	seconds_from_ref_epoch(0), ref_epoch(0),
	frame_number(0), version(0), log2_channels(0), 
	num_channels(0), frame_length(0),
	complex_data(false), bits_per_sample(0), 
	thread_id(0), station_id(0), extended_data_version(0), 
	uflag(false), sync_pattern(0), das_id(0), 
	t(0.0L), n_samples(0) 
{
		// All members are initialized in the member initializer list above.
}

void VDIFHeader::print() const
{
    std::cout << "MJD         " << std::setprecision(20) << t0 << std::endl;
    std::cout << "sampling    " << size_t (sampling) << std::endl;
    std::cout << "tau         " << tau << std::endl;
    std::cout << "BPS         " << size_t(bits_per_sample) << std::endl;
    std::cout << "station_id  " << station_id << std::endl;
    std::cout << "das_id      ";

	for (int i = 0; i < 8; ++i)
	std::cout << std::endl;
}


// Implementation of the VDIFHeader::decode method.
// Parses the raw byte array into the struct's member variables according to the VDIF specification.
void VDIFHeader::decode(const char* byte_array) 
{
	if (byte_array == nullptr) 
		// Handle null pointer gracefully, though the caller should ensure validity.
		// Throwing an exception might be appropriate here.
		throw std::invalid_argument("Byte array for header decoding is null");

	// Read the first 4 mandatory words (16 bytes) from the byte array.
	// Assumes little-endian byte order for uint32_t interpretation.
	word0 = *reinterpret_cast<const uint32_t*>(byte_array);
	word1 = *reinterpret_cast<const uint32_t*>(byte_array + 4);
	word2 = *reinterpret_cast<const uint32_t*>(byte_array + 8);
	word3 = *reinterpret_cast<const uint32_t*>(byte_array + 12);

	// --- Decode mandatory fields from the first 4 words ---
	invalid_data = (word0 & 0x80000000) != 0; // Bit 31
	legacy_mode = (word0 & 0x40000000) != 0;  // Bit 30
	seconds_from_ref_epoch = word0 & 0x3FFFFFFF; // Bits 0-29
	ref_epoch = (word1 >> 24) & 0x3F;         // Bits 24-29
	frame_number = word1 & 0x00FFFFFF;        // Bits 0-23
	version = (word2 >> 29) & 0x07;           // Bits 29-31
	log2_channels = (word2 >> 24) & 0x1F;     // Bits 24-28
	num_channels = 0x00000001U << log2_channels; // Calculate number of channels from log2
	frame_length = (word2 & 0x00FFFFFF) * 8;  // Bits 0-23 * 8 (length in bytes)
	complex_data = (word3 & 0x80000000) != 0; // Bit 31
	bits_per_sample = ((word3 >> 26) & 0x1F) + 1; // Bits 26-30 + 1
	thread_id = (word3 >> 16) & 0x03FF;       // Bits 16-25
	station_id = word3 & 0xFFFF;              // Bits 0-15

	// --- Validation checks ---
	if (complex_data) 
		throw std::invalid_argument("IAA_vdif cannot process complex data");
	if (num_channels != 1U)
		throw std::invalid_argument("IAA_vdif cannot process multi-channel data");

	// Read the next 4 words (16 bytes) if it's NOT a legacy header.
	if (!legacy_mode) 
	{
		word4 = *reinterpret_cast<const uint32_t*>(byte_array + 16);
		word5 = *reinterpret_cast<const uint32_t*>(byte_array + 20);
		word6 = *reinterpret_cast<const uint32_t*>(byte_array + 24);
		word7 = *reinterpret_cast<const uint32_t*>(byte_array + 28);
	}

	// --- Decode optional fields from the next 4 words (if not legacy) ---
	extended_data_version = (word4 >> 24) & 0xFF; // Bits 24-31
	if (extended_data_version != 0x01) 
		throw std::invalid_argument("Unknown extended data version: EDV = " + std::to_string(extended_data_version));

	uflag = (word4 >> 23) & 0x01; // Bit 23
	sampling = static_cast<double>(word4 & 0x007FFFFF); // Bits 0-22
	if (uflag) 
	{
		// Sampling rate is in MHz
		sampling = sampling;
	} 
	else 
	{
		// Sampling rate is in kHz, convert to MHz
		sampling = sampling / 1000.0;
	}


	sync_pattern = word5; // Full 32-bit word
	das_id = (static_cast<uint64_t>(word6) << 32) | (word7); // Combine word6 and word7

	// --- Calculate derived fields ---
	// Determine the number of samples based on header length and bits per sample.
	// Legacy headers have 16 bytes of header data, non-legacy have 32.
	if (legacy_mode) 
	{
		n_samples = (frame_length - 16U) * 8U / bits_per_sample;
	} 
	else 
	{
		n_samples = (frame_length - 32U) * 8U / bits_per_sample;
	}

	// Calculate the absolute time 't' using the helper function.
	// Fractional seconds are estimated based on frame number, number of samples, and sampling rate.
	t = vdifTimeToMJD(ref_epoch, seconds_from_ref_epoch, static_cast<long double>(frame_number) * n_samples * 1e-6 / sampling);
}


// --- IAA_vdif Method Implementations ---

// Constructor implementation.
// Opens the file, reads the *first* frame, initializes buffers and FFTW plan based on the first frame's parameters.
IAA_vdif::IAA_vdif(const std::string& filename_in, size_t buffer_size):
	BaseReader(), header{} // Initialize the base class part first 
{
		// CRITICAL: Define base class member
		// to point at the right header
		header_ptr = &header;

		filename = filename_in;
		file.open(filename, std::ios::binary);
		is_open = file.is_open();

		if (!is_open) 
		{
			throw std::runtime_error("Failed to open file: " + filename);
		}

		// --- Read and decode the *first* header and data ---
		h_buffer = new char[32];
		file.read(h_buffer, 32);
		file.seekg(0);

		// Check for EOF immediately after the first read
		std::ios_base::iostate state = file.rdstate();
		if (state & std::ios_base::eofbit) 
			throw std::runtime_error("File is empty or header could not be read: " + filename);
		header.decode(h_buffer); // Decode the header struct from the first frame
		uint32_t n_samples = header.n_samples; // Get the number of samples from the decoded header
		header.t0 = header.t; // Store the time of the first frame for relative calculations
		header.nchann = 1;
		header.tau = 1e-3/header.sampling;

		// Find the total size of observation (number of points)
		file.seekg(0, std::ios::end); // Move the read pointer to the end of the file
		size_t tot = file.tellg();
		file.seekg(0, std::ios::beg); // Move the read pointer to the beginning of the file

		// Set the corresponding header field
		header.OBS_SIZE = tot / header.frame_length * header.n_samples;

		if (header.legacy_mode)
		{
			delete[] h_buffer;
			h_buffer = nullptr;
			h_buffer = new char[16];
		}



		// --- Allocate buffers based on the *first frame's* characteristics ---
		// Raw data buffer size depends on bits_per_sample and n_samples
		size_t raw_data_size = (n_samples * header.bits_per_sample) / 8U; // Calculate size in bytes
		raw_data = new char[raw_data_size];

		// Main processing buffer size depends on the requested buffer_size in bytes
		try 
		{
			buffer = new double[buffer_size / sizeof(double)]; // Allocate space for 'buffer_size' bytes worth of doubles
			buf_size = buffer_size / sizeof(double); // Store the buffer size in number of doubles
		} catch (const std::bad_alloc& e) 
		{
			std::cerr << "Allocation failed for main buffer: " << e.what() << std::endl;
			// Clean up previously allocated memory if buffer allocation fails
			delete[] raw_data;
			raw_data = nullptr;
			throw; // Re-throw to signal failure
		}

		// Initialize buffer read/write pointers
		buf_pos = 0;
		buf_max = 0;
	}

// Destructor implementation.
// Closes the file and frees all allocated memory.
IAA_vdif::~IAA_vdif() 
{
	if (is_open) 
	{
		file.close();
		is_open = false; // Mark as closed
	}

	// Clean up dynamically allocated memory
	if (raw_data != nullptr) 
	{
		delete[] raw_data;
		raw_data = nullptr;
	}

	if (buffer != nullptr) 
	{
		delete[] buffer;
		buffer = nullptr;
	}

	// Clean up FFTW resources
	if (p != nullptr) 
	{
		fftw_destroy_plan(p);
		p = nullptr;
	}
	if (fft_arr != nullptr) 
	{
		fftw_free(fft_arr);
		fft_arr = nullptr;
	}
}


// Private helper function implementation.
// Unpacks N-bit integer samples from a byte array into double precision values.
void IAA_vdif::unpack_nbit_to_double(
		const char* input, 
		double* output, 
		size_t num_elements, 
		int n) 
{
	size_t bit_pos = 0; // Tracks the current bit position in the input stream

	for (size_t i = 0; i < num_elements; ++i) 
	{
		uint32_t value = 0; // Temporary storage for the unpacked N-bit value

		// Extract 'n' bits for the current sample
		for (int bit = 0; bit < n; ++bit) {
			size_t byte_idx = bit_pos >> 3;           // Calculate the byte index (bit_pos / 8)
			size_t bit_idx  = bit_pos & 7;            // Calculate the bit index within the byte (bit_pos % 8)
			uint8_t byte_val = static_cast<uint8_t>(input[byte_idx]); // Read the byte
			uint32_t bit_val = (byte_val >> bit_idx) & 1; // Extract the specific bit (LSB first)
														  // Insert the extracted bit as the next least significant bit of 'value'
			value |= (bit_val << bit);
			++bit_pos; // Move to the next bit in the stream
		}

		// Convert the packed integer value to a double and store it
		output[i] = static_cast<double>(value);
	}
}

// Public method implementation: read_frame
// Reads the next VDIF frame from the file, decodes its header, and appends its data to the main buffer.
bool IAA_vdif::read_frame() 
{
	if (!is_open) return false; // Cannot read if file is not open

	uint32_t  data_bytes;

	// Check if there's enough space left in the main buffer to append this frame's data
	if (buf_max + header.n_samples > buf_size) 
		// Buffer is full, cannot read more data without processing
		return false;


	// Read the 32-byte header
	if (header.legacy_mode) 
		file.read(h_buffer, 16);
	else
		file.read(h_buffer, 32);

	// Check for End Of File (EOF)
	std::ios_base::iostate state = file.rdstate();
	if (state & std::ios_base::eofbit) 
		return false; // Indicate EOF reached

	// Decode the header for this frame
	header.decode(h_buffer);

	// Calculate the number of bytes representing the sample data in this frame
	data_bytes = (header.n_samples * header.bits_per_sample) / 8U;

	// Read the raw data bytes for this frame into the temporary raw_data buffer
	// Note: This assumes raw_data buffer is large enough for *any* frame in the file,
	// which is guaranteed if all frames have the same parameters as the first one.
	file.read(raw_data, data_bytes);


	// Unpack the raw N-bit data into double precision and append it to the main buffer
	unpack_nbit_to_double(raw_data, buffer + buf_max, header.n_samples, header.bits_per_sample);

	// Update the maximum fill level of the main buffer
	buf_max += header.n_samples;

	return true; // Successfully read and appended one frame
}

// Public method implementation: fill_buffer
// Fills the main buffer by reading frames from the file until the buffer is full or EOF is reached.
// Handles gaps in time between frames by padding with zeros.
bool IAA_vdif::fill_buffer() 
{
	// Shift any remaining data (from a previous incomplete processing chunk) to the front of the buffer
	size_t remaining = buf_max - buf_pos;
	if (remaining > 0) 
	{
		std::memmove(buffer, buffer + buf_pos, remaining * sizeof(double)); // Move remaining data
	}
	buf_max = remaining; // Update the fill level after shifting
	buf_pos = 0;         // Reset the read position to the beginning of the valid data

	// Variables for time gap detection and handling
	long double t_prev, t_cur; // Previous and current frame times
	double time_step;          // Expected time step between consecutive frames

	// Calculate the expected time step based on the *current* header's parameters.
	// Note: If the first frame read by the constructor had different parameters
	time_step = 1e-6 / header.sampling * header.n_samples; // Time per frame in seconds

	// Use the time of the frame that *will* be read next as the starting point for gap calculation.
	// This requires knowing the header of the *next* frame. We can use the current header
	// as the 'previous' time for the *first* frame read in this loop.
	t_prev = (header.t - header.t0) * 86400.0L; // Convert time difference to seconds

	// Loop to read frames until the buffer is full or EOF is reached
	while (read_frame()) 
	{ // Keep calling read_frame until it returns false (EOF or buffer full)
		t_cur = (header.t - header.t0) * 86400.0L; // Get current frame time in seconds

		// Print current time to console (for progress tracking)
		std::cout << std::setprecision(6) << "\rtime: " << t_cur << " s";
		std::cout.flush(); // Ensure output is displayed immediately

		// Check if there's a significant gap in time between the previous and current frame
		if (t_cur - t_prev > 1.001 * time_step) 
		{
			// Calculate how many zero samples are needed to fill the gap
			size_t to_fill = static_cast<size_t>((t_cur - t_prev) / time_step);

			// Fill the buffer with zeros for the duration of the gap
			for (size_t i = 0; i < to_fill; ++i) 
			{
				if (buf_max + i >= buf_size) 
					break;

				buffer[buf_max + i] = 0.0; // Insert zero sample
			}

			buf_max += to_fill; // Update the buffer fill level
		}

		t_prev = t_cur; // Update previous time for the next iteration
	}

	std::cout << std::endl; // New line after time tracking output
							
	return true; // Indicate successful filling (or reaching EOF/buffer limit)
}

// Public method implementation: fill_2d
// Processes the data in the main buffer using FFTs to generate a 2D dynamic spectrum (power).
size_t IAA_vdif::fill_2d(
		double *dyn_spec, 
		size_t time_steps,
		size_t freq_num) 
{
	// Each FFT processes a chunk of 2 * freq_num real samples
	size_t samples_per_chunk = 2 * freq_num;

	double* chunk_start = buffer; // Pointer to the start of the current processing chunk
	double re, im; // Variables to hold real and imaginary parts of FFT output
				   //
	size_t filled = 0;

	// Initialize FFTW plan and output array if they haven't been created yet
	if (fft_arr == nullptr) 
	{
		// Allocate memory for FFTW's output array (freq_num + 1 complex numbers for R2C FFT)
		fft_arr = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (freq_num + 1)));

		// Allocate a temporary input buffer JUST FOR PLANNING
		// as planning overwrites the input array
		double* plan_input = (double*) fftw_malloc(sizeof(double) * samples_per_chunk);
		std::fill_n(plan_input, samples_per_chunk, 0.0); // Initialize with zeros

		// Create the FFTW plan for a real-to-complex FFT of size samples_per_chunk
		// Uses FFTW_PATIENT for potentially better performance at the cost of initialization time
		p = fftw_plan_dft_r2c_1d(samples_per_chunk, chunk_start, fft_arr, FFTW_PATIENT);

		// Clean up scratch buffer — plan is now independent
		fftw_free(plan_input);

		if (!p) 
		{
			fftw_free(fft_arr);
			throw std::runtime_error("Failed to create FFTW plan");
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
			// Access the k+1 element of the FFT output (skip DC component at index 0)
			//re = fft_arr[k+1][0]; // Real part
			//im = fft_arr[k+1][1]; // Imaginary part
			// Store the power (magnitude squared)
			//
			re = fft_arr[freq_num - k][0]; // Real part
			im = fft_arr[freq_num - k][1]; // Imaginary part
			dyn_spec[chunk * freq_num + k] = re*re + im*im;
		}

		filled += 1;
	}

	return filled;
}

// Public method implementation: fill_1d
// Processes the data in the main buffer using FFTs to generate a 1D complex dynamic spectrum.
void IAA_vdif::fill_1d(fftw_complex *vec, size_t n)
{
	size_t i = 0;
	size_t available;
	size_t remaining;
	size_t chunk;
	double* buf_ptr;
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
}

void IAA_vdif::skip(double sec)
{
	if (!file.is_open())
		throw ("The file was not opened"); 

	double frames = sec / (header.n_samples * 1.0e-6 / header.sampling);
    file.seekg(size_t(frames) * header.frame_length, std::ios::cur);

	size_t steps = size_t(frames - size_t(frames)) * header.n_samples;

	read_frame();
	buf_pos += steps;
	header.t0 = (size_t(frames) * header.n_samples + steps) * 1.0e-6 / header.sampling /86400.0;
}	

double IAA_vdif::point2time(size_t points)
{
	return header.tau*1e3 * static_cast<double> (points);
}
