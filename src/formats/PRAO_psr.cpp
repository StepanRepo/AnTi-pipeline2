#include "formats/PRAO_psr.h" // Include the header file defining the class interface and base classes

#include <cstdint>
#include <sstream>    // For std::stringstream
#include <csignal>
#include <cstddef>
#include <ios>
#include <iostream>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument, std::bad_alloc
#include <cstring>    // For std::memcpy, std::memmove, std::strlcpy
#include <algorithm>  // For std::min, std::fill_n, std::remove_if
#include <cctype>     // For std::isspace
#include <filesystem> // For convenient file names handling
#include <unistd.h>
#include <cmath>


// --- Helper Function Implementations ---

// Implementation of the helper function to convert PRAO time string to MJD.
long double PRAOTime2MJD(std::string const time_in) {
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

// Implementation of the ADCHeader constructor.
PRAOHeader::PRAOHeader(): BaseHeader()
{
	// All members are initialized in the member 
	// initializer list above as defaut values
}

PRAOHeader::~PRAOHeader()
{
}

// Implementation of the ADCHeader::decode method.
void PRAOHeader::read_header(std::ifstream &file) 
{

	char c_line[41];
	std::string line = "";
	std::istringstream iss{};

	// find the length of header
	file.seekg(0, std::ios::beg);
	file.read(c_line, 40);
	c_line[40] = '\0';
	line = std::string(c_line);

	size_t first_digit = line.find_first_of("0123456789");
	if (first_digit == std::string::npos) 
	{
		throw std::runtime_error("Could not find number of parameters in header");
	}
	numpar = std::stoi(line.substr(first_digit));


	std::string key = "";
	std::string value = "";
	for (size_t i = 1; i < numpar; ++i) 
	{
		// read the header by 40 symbols
		file.read(c_line, 40);
		c_line[40] = '\0';
		line = std::string(c_line);

		iss = std::istringstream(line);
		iss >> key;

		if (key == "name") 
		{
			iss >> name; 
		} else if (key == "date") 
		{
			iss >> start_date_s;
		}
		// time should always go after the date
		else if (key == "time") 
		{
			if (start_date_s == "")
				throw std::runtime_error("The format of timestring is wrong: time is defined before date");
			else
			{
				iss >> value;
				start_date_s = start_date_s.substr(0, start_date_s.find(' ')) + " " + value;
			}
		} 
		else if (key == "period") 
		{
			iss >> period;
		} 
		else if (key == "numpuls") 
		{
			iss >> nsubint;
		} 
		else if (key == "tay" || key == "tau") 
		{
			// time sampling is stored in ms
			iss >> tau;
		}
		else if (key == "numpointwin") 
		{
			iss >> obs_window;
		} 
		else if (key == "sumchan") 
		{
			iss >> value;
			if (value == "adc")
				throw std::runtime_error("The format of file is wrong. Check sumchan: adc | yes | no");

			if (value == "yes" || value == "no")
				MODE = "PSR";
			else 
				throw std::runtime_error("The format of file is wrong. Check sumchan: adc | yes | no");
		} 
		else if (key == "dm") 
		{
			iss >> dm;
		}
		else if (key == "freq0" || key == "F0" || key == "Fmin") 
		{
			iss >> fmin;
		}
		else if (key == "freq511" || key == "F511" || key == "Fmax") 
		{
			iss >> fmax;

			if (key == "freq511")
				nchann = 512;
		}
		else if (key == "dt_utc") 
		{
			std::getline(iss, start_utc_s);
		}
	}

	if (start_utc_s == "") 
		throw std::runtime_error("There is no utc time in observation file");
	else 
		t0 = PRAOTime2MJD(start_utc_s);

	if (fmin == 0.0 || fmax == 0.0)
		throw std::runtime_error("PRAO file should contain frequency information. Check header");


	npol = 1;
	pol_type = "LIN";

	double df = (fmax-fmin)/double(nchann);
	freqs.resize(nchann);

	for (size_t i = 0; i < nchann; ++i)
		freqs[i] = fmin + df * (double(i) + .5);

	if (nsubint > 0)
	{
		t_subint.resize(nsubint);

		for (size_t i = 0; i < nsubint; ++i)
			t_subint[i] = period * (double(i) + .5);

	}

	OBS_SIZE = nchann * obs_window;
}

// Constructor implementation.
// Opens the file, reads and decodes the header, initializes buffers and FFTW plan.
PRAO_psr::PRAO_psr(const std::string& filename_in, size_t buffer_size): 
	BaseReader(), header{}
{
    header_ptr = &header; // <--- CRITICAL: Set the base class's header_ptr member here

	std::filesystem::path p = filename_in;
    filename = p.stem();

    file.open(p, std::ios::binary);
    is_open = file.is_open();

    if (!is_open) 
	{
        throw std::runtime_error("Failed to open file: " + filename);
    }


	header.read_header(file);

    file.seekg(header.get_numpar() * 40, std::ios::beg);
	data_start_pos = file.tellg();
	t0_orig = header.t0;

    // PRAO format are files written by
	// a pulsar machine. The raw spectrum is 
	// averaged over time (usually over 6 steps).
	// The numbers are coded as 32-byte ints as
	// num & 0x00FFFFFF is the number's mantisa
	// num & 0x7F000000 is its exponent
	// 
    buf_size = buffer_size / (sizeof(double) + sizeof(int32_t));
	buf_size = std::max(buf_size, header.OBS_SIZE*header.nchann*header.npol);

    try 
	{
        raw_data = new int32_t[buf_size]; 
        buffer = new double[buf_size];   
        buf_pos = 0;
        buf_max = 0;
    } 
	catch (const std::bad_alloc& e) 
	{
        std::cout << "Allocation failed for main buffers: " << e.what() << std::endl;
        // Clean up partially allocated memory if one allocation succeeds and the other fails
		delete[] raw_data;
		raw_data = nullptr;

		delete[] buffer;
		buffer = nullptr;
        throw; // Re-throw to signal failure
    }
}

// Destructor implementation.
// Closes the file and frees all allocated memory.
PRAO_psr::~PRAO_psr() {
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
}

// Public method implementation: fill_buffer
// Fills the main buffer by reading data from the file until 
// the buffer is full or EOF is reached.
// Additionally andles conversion from int8_t raw data to double.
bool PRAO_psr::fill_buffer() 
{
    // Shift remaining data (from a previous incomplete 
	// processing chunk) to the front of the buffer
    size_t remaining = buf_max - buf_pos;
    if (remaining > 0) 
	{
        // Move raw data
        std::memmove(raw_data, raw_data + buf_pos, remaining * sizeof(int32_t));
        // Move decoded data
        std::memmove(buffer, buffer + buf_pos, remaining * sizeof(double));
    }
    buf_max = remaining; // Update the fill level after shifting
    buf_pos = 0;         // Reset the read position to the beginning of the valid data
	size_t nchann = header.nchann;
	size_t npol = header.npol;

    // Compute how many new samples we can read
    std::streamoff current_pos = file.tellg();
    if (current_pos < 0) return false; // Error getting position

    // Calculate how much data has been read from the *data* section (after header)
    size_t data_section_start = header.numpar * 40;
    std::streamoff data_read_so_far_off = current_pos - static_cast<std::streamoff>(data_section_start);

    // Check for underflow or error in calculation
    if (data_read_so_far_off < 0) return false;
    size_t data_read_so_far = static_cast<size_t>(data_read_so_far_off);
	data_read_so_far = data_read_so_far / nchann / npol;

    // Check if we've already read all the expected data
    if (data_read_so_far >= header.OBS_SIZE || data_read_so_far >= header.CUT_SIZE) 
        return false;

    // Determine how much more data we can read
    size_t max_to_read = header.OBS_SIZE - data_read_so_far;
    size_t cut_to_read = header.CUT_SIZE - data_read_so_far;
    size_t space_available = (buf_size - buf_max)/nchann/npol; // Space left in the buffer
						
    size_t to_read = std::min(max_to_read, space_available);
	to_read = std::min(to_read, cut_to_read);

    if (to_read == 0) 
        return false; // Buffer is full or no more data to read

    // Read raw int8_t data into the raw_data buffer starting at offset 'buf_max'
    file.read(reinterpret_cast<char*>(raw_data + buf_max), sizeof(int32_t) *to_read * npol * nchann);

    // Check how many bytes were actually read
    size_t actually_read = static_cast<size_t>(file.gcount()) / sizeof(int32_t);

    if (actually_read == 0) 
        return false; // eof or error occurred during read

    // The loop processes data starting from 'buf_max'
	int32_t spectr_t, exp;
	size_t arg;

	// this ratio comes from binning time samples 
	// 2 is from real-valued baseband sampling
	// 200 ns is detector's temporal resolution (5 MHz)
	// nchann is from fourier spectrum procedure
	// tau is actual written temporal resolution
	double ratio = 2.0 * 200.0e-9 * header.nchann / (header.tau*1e-3);


    for (size_t i = 0; i < actually_read; ++i) 
	{
		arg = buf_max + i;
		spectr_t = raw_data[arg] & 0x00FFFFFF;
		exp      = (raw_data[arg] & 0x7F000000) >> 24;

		exp = exp-64-24;

		// spectr_t * (2^exp)
		buffer[arg] = std::ldexp (double(spectr_t), exp) * ratio;
    }

    // Update the maximum fill level of the main buffer
    buf_max += actually_read;
    return true; // Successfully filled the buffer (or reached EOF/data limit)
}



void PRAO_psr::skip(double sec)
{
	if (!file.is_open())
		throw ("The file was not opened"); 

	size_t steps = size_t(sec / (header.tau * 1e-3));

	// Skip only the whole number of subints
	size_t si_skip = steps / header.obs_window;
	size_t bytes_skip = si_skip * header.obs_window * header.nchann * sizeof(int32_t);

    file.seekg(
			data_start_pos + 
			static_cast<std::streamoff>(bytes_skip), 
			std::ios::beg);

	t0_orig = header.t0;
	double skipped = si_skip * header.obs_window * header.tau*1.0e-3;
	header.nsubint -= si_skip;

	for (size_t i = 0; i < header.nsubint; ++i)
		header.t_subint[i] = header.t_subint[i+si_skip] - skipped;

	header.t0 += skipped / 86400.0;
	data_start_pos = file.tellg(); // Update effective start
}	

void PRAO_psr::set_limit(double t)
{
	double correction = double(header.t0 - t0_orig)*86400.0;
	if (correction < 0.0)
		return;

	size_t steps = size_t ((t - correction) * 1.0e3 / header.tau);
	size_t si_limit = steps / header.obs_window + 1;
	header.nsubint = si_limit;

	header.CUT_SIZE = si_limit * header.obs_window;
}


double PRAO_psr::point2time(size_t point) 
{
	size_t current_si = point / header.obs_window;
	double t0_si = header.t_subint[current_si] - header.obs_window * header.tau*.5e-3;

	size_t reminder = point - current_si * header.obs_window;
	double t = t0_si + reminder * header.tau * 1.0e-3;

	return t;
}

bool PRAO_psr::allow_1d()
{return false;}

bool PRAO_psr::allow_2d()
{return true;}
