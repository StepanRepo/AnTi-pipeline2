#include "formats/PRAO_lpa3.h" // Include the header file defining the class interface and base classes


#include <csignal>
#include <cstddef>
#include <ios>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument, std::bad_alloc
#include <sstream>    // For std::stringstream
#include <iostream>   // For std::cout
#include <cstring>    // For std::memcpy, std::memmove, std::strlcpy
#include <algorithm>  // For std::min, std::fill_n, std::remove_if
#include <cctype>     // For std::isspace
#include <filesystem> // For convenient file names handling
#include <fstream>    // For file I/O


// --- Helper Function Implementations ---

// Implementation of the helper function to convert PRAO time string to MJD.
long double LPA3Time2MJD(std::string const time_in) {
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
LPA3Header::LPA3Header(): 
	BaseHeader(),
	start_date_s(""), start_utc_s("")
{
	// All members are initialized in the member 
	// initializer list above as defaut values
}


LPA3Header::~LPA3Header()
{
}

// Implementation of the ADCHeader::decode method.
void LPA3Header::read_header(std::ifstream &file) 
{
    std::string key, value, buffer;

	std::getline(file, buffer);
	std::istringstream iss(buffer);


	iss >> key;

	if (key == "numpar")
		iss >> numpar;
	else
		throw std::runtime_error("The file is not valid LPA3 format");


    for (size_t i = 1; i < numpar; ++i) 
	{

		std::getline(file, buffer);
		std::istringstream iss(buffer);

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
				start_date_s = start_date_s.substr(0, start_date_s.find(' ')) + " " + value;
        } 
		else if (key == "tay" || key == "tau") 
		{
            // time sampling is stored in ms
            iss >> tau;
            // sampling rate stored in MHz
            sampling = 1.0e-3 / tau;
        }
	   	else if (key == "freq0" || key == "F0" || key == "Fmin") 
		{
            iss >> fmin;
        }
	   	else if (key == "freq511" || key == "F511" || key == "Fmax") 
		{
            iss >> fmax;
        }
	   	else if (key == "dt_utc") 
		{
			std::getline(iss, start_utc_s);
        }
		else if (key == "nbands")
		{
			iss >> nchann;
		}
		else if (key == "fbands")
		{
			if (nchann == 0)
				throw std::runtime_error("Unknown number of frequncy channels in the file");
			if (freqs.size() == 0)
				freqs.resize(nchann);

			for (size_t f = 0; f < nchann; ++f)
				iss >> freqs[f];


			double df = (freqs[nchann-1] - freqs[0]) / double(nchann-1);
			fmin = freqs[0] - df/2.0;
			fmax = freqs[nchann-1] + df/2.0;
		}
    }


	if (start_utc_s == "") 
		throw std::invalid_argument("There is no utc time in observation file");
	else 
		t0 = LPA3Time2MJD(start_utc_s);

	npol = 1;
	nsubint = 1;
	MODE = "SEARCH";

	if (freqs.size() == 0)
	{
		if (fmin > 0.0 && fmax > 0.0)
		{
			double df = (fmax-fmin)/double(nchann);
			freqs.resize(nchann);

			for (size_t i = 0; i < nchann; ++i)
				freqs[i] = fmin + df * (double(i) + .5);
		}
		else {
			throw std::runtime_error("Frequency information was not provided in the file");
		}
	}

}


// Constructor implementation.
PRAO_lpa3::PRAO_lpa3(const std::string& filename_in, size_t buffer_size): 
	BaseReader(), header{}
{

    header_ptr = &header; // <--- CRITICAL: Set the base class's header_ptr member here

	std::filesystem::path p = filename_in;
    filename = p.filename();


    file.open(p, std::ios::binary);
    is_open = file.is_open();

    if (!is_open) 
	{
        throw std::runtime_error("Failed to open file: " + filename);
    }


	header.read_header(file);
	t0_orig = header.t0;
	data_start_pos = static_cast<std::streamoff>(file.tellg());

    // Find total number of points in the file
    file.seekg(0, std::ios::end);
    size_t data_size = static_cast<size_t>(file.tellg()) - 
		static_cast<size_t>(data_start_pos);

    header.OBS_SIZE = data_size / sizeof(float) / header.nchann;
    file.seekg(data_start_pos, std::ios::beg); // Seek to the start of data after header


    // LPA3 formated files are 
    // written after binning observed spectra
	// in float (32-byte) numbers
    // split the given buffer to equal
    // amount of numbers in the raw and decoded arrays
    // 2 parts (64 bytes) goes to double data
    // 1 part (32 bytew) goes to float raw_data
    buf_size = buffer_size / (sizeof(double) + sizeof(float));
	buf_size = std::max(buf_size, header.OBS_SIZE*header.nchann);

    try 
	{
        raw_data = new float[buf_size]; // Allocate raw buffer for 'buf_size' bytes (int8_t)
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
}

// Destructor implementation.
// Closes the file and frees all allocated memory.
PRAO_lpa3::~PRAO_lpa3() 
{
    if (is_open) 
	{
        file.close();
        is_open = false; // Mark as closed
    }

    // Clean up dynamically allocated memory
    if (raw_data) 
	{
        delete[] raw_data;
        raw_data = nullptr;
    }
	
    if (buffer) 
	{
        delete[] buffer;
        buffer = nullptr;
    }
}

// Public method implementation: fill_buffer
// Fills the main buffer by reading data from the file until 
// the buffer is full or EOF is reached.
// Additionally andles conversion from int8_t raw data to double.
bool PRAO_lpa3::fill_buffer() 
{
    // Shift remaining data (from a previous incomplete 
	// processing chunk) to the front of the buffer
    size_t remaining = buf_max - buf_pos;
    if (remaining > 0) 
	{
        // Move raw data
        std::memmove(raw_data, raw_data + buf_pos, remaining * sizeof(float));
        // Move decoded data
        std::memmove(buffer, buffer + buf_pos, remaining * sizeof(double));
    }
    buf_max = remaining; // Update the fill level after shifting
    buf_pos = 0;         // Reset the read position to the beginning of the valid data

    // Compute how many new samples we can read
    std::streamoff current_pos = file.tellg();
    if (current_pos < 0) return false; // Error getting position

    // Calculate how much data has been read from the *data* section (after header)
    std::streamoff data_read_so_far_off = current_pos - data_start_pos;

    // Check for underflow or error in calculation
    if (data_read_so_far_off < 0) return false;
    size_t data_read_so_far = static_cast<size_t>(data_read_so_far_off) / header.nchann / header.npol;

    // Check if we've already read all the expected data
    if (data_read_so_far >= header.OBS_SIZE || data_read_so_far >= header.CUT_SIZE) 
        return false;

    // Determine how much more data we can read
    size_t max_to_read = header.OBS_SIZE - data_read_so_far;
    size_t cut_to_read = header.CUT_SIZE - data_read_so_far;
    size_t space_available = (buf_size - buf_max)/header.nchann/header.npol; // Space left in the buffer
												 //
    size_t to_read = std::min(max_to_read, space_available);
	to_read = std::min(to_read, cut_to_read);

    if (to_read == 0) 
        return false; // Buffer is full or no more data to read

    // Read raw int8_t data into the raw_data buffer starting at offset 'buf_max'
    file.read(reinterpret_cast<char*>(raw_data + buf_max), sizeof(float)*to_read*header.nchann);

    // Check how many bytes were actually read
    size_t actually_read = static_cast<size_t>(file.gcount()) / sizeof(float);

    if (actually_read == 0) 
        return false; // eof or error occurred during read

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



void PRAO_lpa3::skip(double sec)
{
	if (!file.is_open())
		throw ("The file was not opened"); 

	size_t steps = sec / (header.tau * 1.0e-3);
    file.seekg(
			data_start_pos + 
			static_cast<std::streamoff>(steps * header.nchann * sizeof(float)), 
			std::ios::beg);

	t0_orig = header.t0;
	header.t0 += steps * (header.tau * 1.0e-3) / 86400.0;
	data_start_pos = file.tellg(); // Update effective start
}	

void PRAO_lpa3::set_limit(double t)
{
	double correction = double(header.t0 - t0_orig)*86400.0;
	if (correction < 0.0)
		return;

	header.CUT_SIZE = size_t ((t - correction) * 1.0e3 / header.tau);
}

double PRAO_lpa3::point2time(size_t point) 
{
	return header.tau*1.0e-3 * static_cast<double> (point);
}

bool PRAO_lpa3::allow_1d()
{return false;}

bool PRAO_lpa3::allow_2d()
{return true;}
