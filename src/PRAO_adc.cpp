/*
 * PRAO_adc.cpp
 *
 * Implementation file for the PRAO_adc class.
 * Provides the definitions for the constructor, destructor,
 * helper functions, and all public/private methods declared in PRAO_adc.h.
 * This class handles PRAO ADC file reading, decoding, buffering, and FFT processing.
 */

#include "PRAO_adc.h" // Include the header file defining the class interface and base classes
#include <stdexcept>  // For std::runtime_error, std::invalid_argument, std::bad_alloc
#include <iostream>   // For std::cerr, std::endl
#include <iomanip>    // For std::setprecision
#include <cstring>    // For std::memcpy, std::memmove, std::strlcpy
#include <limits>     // For std::numeric_limits (if needed for validation)
#include <algorithm>  // For std::min, std::fill_n, std::remove_if
#include <cctype>     // For std::isspace
#include <fftw3.h>      // For FFTW library types (fftw_complex, fftw_plan)

// --- Helper Function Implementations ---

// Implementation of the helper function to convert PRAO time string to MJD.
long double ADCTime2MJD(std::string const time_in) {
    std::string time_s = time_in;
    // remove spaces from the line
    time_s.erase(std::remove_if(time_s.begin(), time_s.end(), ::isspace), time_s.end());
    // if year is represented like yy then make it like 20yy
    if (time_s.length() == 23)
        time_s = time_s.substr(0, 6) + "20" + time_s.substr(6);

    long d, m, y;
    long double h, min, s;

    // try to get date numbers from given string
    try {
        d = std::stoi(time_s.substr(0, 2));
        m = std::stoi(time_s.substr(3, 2));
        y = std::stoi(time_s.substr(6, 4));
        h = static_cast<long double>(std::stoi(time_s.substr(10, 2)));
        min = static_cast<long double>(std::stoi(time_s.substr(13, 2)));
        s = std::stold(time_s.substr(16, 2) + "." + time_s.substr(18));
    } catch (const std::invalid_argument& err) {
        throw std::invalid_argument("Can't read time string value: " + time_s);
    }

    long double mjd = static_cast<long double>(367 * y - 7 * (y + (m + 9) / 12) / 4 - 3 * (1 + (y + (m - 9) / 7) / 100) / 4 + 275 * m / 9 + d);
    mjd += 1721028.0L - 2400000.0L + h / 24.0L + min / 1440.0L + s / 86400.0L;
    return mjd;
}

// Implementation of the helper function to remove whitespace.
void strip_white(std::string& line) {
    if (line == "" || line.size() == 0) return;
    size_t n = line.size();
    for (size_t i = 0; i < n; i++) {
        if (line[i] == ' ' || line[i] == '\t') {
            line.erase(i, 1);
            i--;
            n--;
        }
    }
}

// Implementation of the helper function to split name/value pairs.
void str_split(char* buffer_c, std::string& name, std::string& value) {
    std::string buffer = buffer_c;
    size_t i = 0;
    name = "";
    value = "";
    size_t comma = buffer.find(',');
    if (comma < 200)
        buffer[comma] = '.';
    while (buffer[i] != ' ') {
        name += buffer[i];
        i++;
    }
    while (buffer[i] == ' ')
        i++;
    // костыль из-за формата записи времени в заголовочном файле
    if (name == "dt_utc" || name == "time" || name == "date") {
        value = buffer.substr(i);
        // remove the last character from the line
        // if it's bad character
        size_t null = value.find('\r');
        if (null < 200)
            value = value.substr(0, null);
        null = value.find('\0');
        if (null < 200)
            value = value.substr(0, null);
        return;
    }
    while (buffer[i] != ' ') {
        value += buffer[i];
        if (i > 40 || i == buffer.length())
            break;
        i++;
    }
    strip_white(name);
    strip_white(value);
    // remove the last character from the line
    // if it's a special character
    size_t null = value.find('\r');
    if (null < 200)
        value = value.substr(0, null);
    null = value.find('\0');
    if (null < 200)
        value = value.substr(0, null);
}


// --- ADCHeader Method Implementations ---

// Implementation of the ADCHeader constructor.
ADCHeader::ADCHeader(): 
	BaseHeader(),
	start_date_s(""), start_utc_s("")
{
	// All members are initialized in the member 
	// initializer list above as defaut values
}

// Implementation of the ADCHeader::decode method.
void ADCHeader::decode(const char* h_buff) 
{
    if (h_buff == nullptr) 
	{
        throw std::invalid_argument("Byte array for header decoding is null");
    }

    std::string key, value;
    char buffer_c[41];

    // read the first 40 symbols
    // the number of parameters is stored here
    std::strncpy(buffer_c, h_buff, 40);
	buffer_c[40] = '\0'; // Explicitly null-terminate after strncpy
    str_split(buffer_c, key, value);
    numpar = std::stoi(value);

    for (size_t i = 1; i < numpar; ++i) 
	{
        // read the header by 40 symbols
        std::strncpy(buffer_c, h_buff + i * 40, 40);
        str_split(buffer_c, key, value);
        if (key == "name") 
		{
            name = value; 
        } else if (key == "date") 
		{
            start_date_s = value;
        }
        // time should always go after the date
        else if (key == "time") 
		{
			if (start_date_s == "")
                throw std::runtime_error("The format of timestring is wrong: time is defined before date");
			else
				start_date_s = start_date_s.substr(0, start_date_s.find(' ')) + " " + value;
        } 
		else if (key == "period") 
		{
            period = std::stold(value);
        } 
		else if (key == "numpuls") 
		{
            total_pulses = std::stoi(value);
        } 
		else if (key == "tay" || key == "tau") 
		{
            // time sampling is stored in ms
            tau = std::stod(value);
            // sampling rate stored in MHz
            sampling = 1.0e-3 / tau;
        }
	   	else if (key == "numpointwin") 
		{
            obs_window = std::stoi(value);
        } 
		else if (key == "sumchan") 
		{
            if (value != "adc")
                throw std::runtime_error("The format of file is wrong");
            else
                folded = false;
        } 
		else if (key == "dm") 
		{
            dm = std::stod(value);
        }
	   	else if (key == "freq0" || key == "F0" || key == "Fmin") 
		{
            fmin = std::stod(value);
        }
	   	else if (key == "freq511" || key == "F511" || key == "Fmax") 
		{
            fmax = std::stod(value);
        }
	   	else if (key == "dt_utc") 
		{
            start_utc_s = value;
        }
    }

    if (start_utc_s == "") 
        throw std::invalid_argument("There is no utc time in observation file");
	else 
        t0 = ADCTime2MJD(start_utc_s);

    // OBS_SIZE for an ADC file is
    // time of observation * (5 MiB/sec).
    // It is (time*5 * 2^20)  int_8t numbers (bytes)
    OBS_SIZE = static_cast<size_t>(period * (total_pulses - 1) * 5 * 1024 * 1024);

    // sampling rate is wrong in the file's header
    // it is 200 ns
	nchann = 1;
	tau = 200.0e-6; // !!! NEED PRECICE VALUE. CONTACT CONSTRUCTORS !!!
    sampling = 1.0e-3 / tau; // Recalculate sampling rate in MHz based on corrected tau
}

// Implementation of the ADCHeader::print method.
void ADCHeader::print() const
{
    std::cout << "numpar      " << numpar << std::endl;
    std::cout << "name        " << name << std::endl;
    std::cout << "date        " << start_date_s << std::endl;
    std::cout << "dt_utc      " << start_utc_s << std::endl;
    std::cout << "MJD         " << std::setprecision(20) << t0 << std::endl;
    std::cout << "period      " << period << std::endl;
    std::cout << "numpuls     " << total_pulses << std::endl;
    std::cout << "tau         " << tau << std::endl;
    std::cout << "numpointwin " << obs_window << std::endl;
    std::cout << "sumchan     " << (folded ? "true" : "false") << std::endl; // Print true/false instead of 0/1
    std::cout << "F0          " << fmin << std::endl;
    std::cout << "F511        " << fmax << std::endl;
}


// --- PRAO_adc Method Implementations ---

// Constructor implementation.
// Opens the file, reads and decodes the header, initializes buffers and FFTW plan.
PRAO_adc::PRAO_adc(const std::string& filename_in, size_t buffer_size): 
	BaseReader(), header{}
{

    header_ptr = &header; // <--- CRITICAL: Set the base class's header_ptr member here

    filename = filename_in;
    file.open(filename, std::ios::binary);
    is_open = file.is_open();

    if (!is_open) 
	{
        throw std::runtime_error("Failed to open file: " + filename);
    }

    char line[40];
    int numpar;

    // find the length of header
    file.read(line, 40);
    std::string s = line;
    size_t first_digit = s.find_first_of("0123456789");
    if (first_digit == std::string::npos) 
	{
        throw std::runtime_error("Could not find number of parameters in header: " + filename);
    }
    numpar = std::stoi(s.substr(first_digit));

    // read the entire header at once
    char* h_buff = nullptr;
	h_buff = new char[numpar * 40];

    file.seekg(0);
    file.read(h_buff, numpar * 40);

    // fill the header
    header.decode(h_buff);
    delete[] h_buff;
    h_buff = nullptr;

    // Correct the OBS_SIZE if the machine
    // somehow failed and missed the end of file
    // Seek to end and get size
    file.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(file.tellg());
    header.OBS_SIZE = (file_size - header.numpar * 40) / sizeof(int8_t);
    file.seekg(header.numpar * 40, std::ios::beg); // Seek to the start of data after header

    // ADC format are files that are directly
    // written from adc of the antenna
    // numbers in adc files are stored as
    // signed 8-bit (1-byte) integers
    // split the given buffer to equal
    // amount of numbers in the raw and decoded arrays
    // 8 parts (bytes) goes to double data
    // 1 part (byte) goes to int8_t raw_data
    buf_size = buffer_size / (sizeof(double) + sizeof(int8_t));

    try 
	{
        raw_data = new int8_t[buf_size]; // Allocate raw buffer for 'buf_size' bytes (int8_t)
        buffer = new double[buf_size];   // Allocate main buffer for 'buf_size' doubles
        buf_pos = 0;
        buf_max = 0;
    } 
	catch (const std::bad_alloc& e) 
	{
        std::cerr << "Allocation failed for main buffers: " << e.what() << std::endl;
        // Clean up partially allocated memory if one allocation succeeds and the other fails
		delete[] raw_data;
		raw_data = nullptr;

		delete[] buffer;
		buffer = nullptr;
        throw; // Re-throw to signal failure
    }

    fft_arr = nullptr;
}

// Destructor implementation.
// Closes the file and frees all allocated memory.
PRAO_adc::~PRAO_adc() {
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

// Public method implementation: fill_buffer
// Fills the main buffer by reading data from the file until 
// the buffer is full or EOF is reached.
// Additionally andles conversion from int8_t raw data to double.
bool PRAO_adc::fill_buffer() 
{
    // Shift remaining data (from a previous incomplete 
	// processing chunk) to the front of the buffer
    size_t remaining = buf_max - buf_pos;
    if (remaining > 0) 
	{
        // Move raw data
        std::memmove(raw_data, raw_data + buf_pos, remaining);
        // Move decoded data
        std::memmove(buffer, buffer + buf_pos, remaining * sizeof(double));
    }
    buf_max = remaining; // Update the fill level after shifting
    buf_pos = 0;         // Reset the read position to the beginning of the valid data

    // Compute how many new samples we can read
    std::streamoff current_pos = file.tellg();
    if (current_pos < 0) return false; // Error getting position

    // Calculate how much data has been read from the *data* section (after header)
    size_t data_section_start = header.numpar * 40;
    std::streamoff data_read_so_far_off = current_pos - static_cast<std::streamoff>(data_section_start);

    // Check for underflow or error in calculation
    if (data_read_so_far_off < 0) return false;
    size_t data_read_so_far = static_cast<size_t>(data_read_so_far_off);

    // Check if we've already read all the expected data
    if (data_read_so_far >= header.OBS_SIZE) 
        return false;

    // Determine how much more data we can read
    size_t max_to_read = header.OBS_SIZE - data_read_so_far;
    size_t space_available = buf_size - buf_max; // Space left in the buffer
    size_t to_read = std::min(max_to_read, space_available);

    if (to_read == 0) 
        return false; // Buffer is full or no more data to read

    // Read raw int8_t data into the raw_data buffer starting at offset 'buf_max'
    file.read(reinterpret_cast<char*>(raw_data + buf_max), to_read);

    // Check how many bytes were actually read
    size_t actually_read = static_cast<size_t>(file.gcount());

    if (actually_read == 0) 
        return false; // EOF or error occurred during read

    // Convert ONLY the newly read samples from int8_t to double
    // The loop processes data starting from 'buf_max'
    for (size_t i = 0; i < actually_read; ++i) 
	{
        buffer[buf_max + i] = static_cast<double>(raw_data[buf_max + i]);
    }

    // Update the maximum fill level of the main buffer
    buf_max += actually_read;
    return true; // Successfully filled the buffer (or reached EOF/data limit)
}

// Public method implementation: fill_2d
// Processes the data in the main buffer using FFTs to generate a 2D dynamic spectrum (power).
size_t PRAO_adc::fill_2d(double* dyn_spec, size_t time_steps, size_t freq_num) 
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
        p = fftw_plan_dft_r2c_1d(samples_per_chunk, plan_input, fft_arr, FFTW_MEASURE);
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

// Public method implementation: fill_1d
// Processes the data in the main buffer using FFTs to generate a 1D complex dynamic spectrum.
void PRAO_adc::fill_1d(fftw_complex *vec, size_t n) 
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

void PRAO_adc::skip(double sec)
{
	if (!file.is_open())
		throw ("The file was not opened"); 

	size_t steps = sec * (header.sampling * 1.0e6);
    file.seekg(steps * sizeof(int8_t), std::ios::cur);
	header.t0 += steps * header.sampling * 1.0e6 / 86400.0;
}	

double PRAO_adc::point2time(size_t point) 
{
	return header.tau*1.0e-3 * static_cast<double> (point);
}
