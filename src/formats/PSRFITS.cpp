#include "formats/PSRFITS.h" // Include the header file defining the class interface and base classes
#include "fitsio.h"
#include "aux_math.h"


#include <cstdint>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdio>
#include <iostream>   // For std::cerr, std::endl
#include <cstring>    // For std::memcpy, std::memmove, std::strlcpy
#include <cctype>     // For std::isspace
#include <fftw3.h>    // For FFTW library types (fftw_complex, fftw_plan)
#include <filesystem> // For std::filesystem::path.stem()
#include <stdexcept>

void PSRFITS::check_status(std::string operation)
{
    if (status) 
	{
        char errtext[FLEN_STATUS];
        fits_get_errstatus(status, errtext);
        // You might want to use your logging system here
        fprintf(stderr, "FITS Error during %s: %s\n", operation.c_str(), errtext);

		status = 0;
    }
}

void read_key_bin(fitsfile *fptr, int datatype, std::string key, long firstrow, long firstelem, long nelements, void *nulval, void *array, int *anynull, int *status)
{
	int colum; // column number in the table

	fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(key.c_str()), &colum, status);

	fits_read_col(fptr, datatype, colum, firstrow, firstelem, nelements, nulval, array, anynull, status);
}

template<typename T>
void read_key(fitsfile *fptr, int datatype, std::string key, T *array, int* status)
{

	if (datatype == TINT)
	{
		int a;
		fits_read_key(fptr, datatype, const_cast<char*>(key.c_str()), &a, NULL, status);
		array[0] = static_cast<T>(a);
	}
	else
	{
		fits_read_key(fptr, datatype, const_cast<char*>(key.c_str()), array, NULL, status);
	}
}


void read_data(
		fitsfile *fptr, 
		int KIND, 
		size_t subint_index, 
		size_t subint_pos,
		int8_t *raw_data,
		double *data,
		int *status,
		size_t nbin, size_t npol, size_t nchann, size_t c
		)
{
	int anynull;
	size_t sample_size = npol*nchann*c;

	thread_local std::vector<float> dat_scl, dat_offs, dat_wts;
	if(dat_scl.size() != npol*nchann*c) dat_scl.resize(npol*nchann*c);
	if(dat_offs.size() != npol*nchann*c) dat_offs.resize(npol*nchann*c);
	if(dat_wts.size() != nchann) dat_wts.resize(nchann);

	read_key_bin(
			fptr, KIND, "DATA", 
			subint_index, subint_pos, nbin*sample_size, 
			NULL, raw_data, &anynull, status);

	read_key_bin(
			fptr, TFLOAT, "DAT_WTS", 
			subint_index, 1, nchann, 
			NULL, dat_wts.data(), &anynull, status);

	read_key_bin(
			fptr, TFLOAT, "DAT_OFFS", 
			subint_index, 1, nchann*npol*c, 
			NULL, dat_offs.data(), &anynull, status);

	read_key_bin(
			fptr, TFLOAT, "DAT_SCL", 
			subint_index, 1, nchann*npol*c, 
			NULL, dat_scl.data(), &anynull, status);



	if (KIND == TBYTE)
	{
		for (size_t t = 0; t < nbin; ++t) 
		{
			for (size_t p = 0; p < npol; ++p) 
			{
				for (size_t f = 0; f < nchann; ++f) 
				{
					for (size_t comp = 0; comp < c; ++comp) 
					{
						size_t idx = ((t * npol + p) * nchann + f) * c + comp;
						size_t stat_idx = (f * npol + p) * c + comp; 


						data[idx] = double (raw_data[idx]) * double(dat_scl[stat_idx]) \
									+ double (dat_offs[stat_idx]);
					}
				}
			}
		}
	}
	else if (KIND == TSHORT)
	{
		uint16_t ures;
		int16_t res;
		uint8_t byte1, byte2;

		// decode input data
		for (size_t t = 0; t < nbin; ++t) 
		{
			for (size_t p = 0; p < npol; ++p) 
			{
				for (size_t f = 0; f < nchann; ++f) 
				{
					size_t idx = (t * npol + p) * nchann + f;

					byte1 = raw_data[2*idx];
					byte2 = raw_data[2*idx + 1];
					ures = (static_cast<uint16_t>(byte2) << 8) | byte1;
					memcpy(&res, &ures, sizeof(ures));

					data[idx] = double (res);
				}
			}
		}

		// Un-quantize input data
		std::vector<size_t> shape(2);
		shape = {nchann,nbin};

		math::layout_c_to_f(data, shape);

		for (size_t t = 0; t < nbin; ++t) 
		{
			for (size_t f = 0; f < nchann; ++f) 
			{
				data[f + t*nchann] = double (data[f + t*nchann]) 
					* double(dat_scl[f]) 
					+ double (dat_offs[f]);
			}
		}
	}
}


PSRFITSHeader::PSRFITSHeader(): 
	BaseHeader()
{
	// All members are initialized in the member 
	// initializer list above as defaut values
}
void PSRFITSHeader::fill(fitsfile *fptr, int *status)
{
	// Read information from prymary header
	// It has lower priority than the HISTORY 
	// bintable, but may be used if it is absent
	char str[FLEN_VALUE];
	fits_read_key_str(fptr, "OBS_MODE", str, NULL, status);
	MODE = std::string(str);

	fits_read_key_str(fptr, "SRC_NAME", str, NULL, status);
	name = std::string(str);

	read_key<size_t>(fptr, TINT, "NRCVR", &npol, status);
	read_key<size_t>(fptr, TINT, "OBSNCHAN", &nchann, status);

	double fcenter, bw;
	read_key<double>(fptr, TDOUBLE, "OBSFREQ", &fcenter, status);
	read_key<double>(fptr, TDOUBLE, "OBSBW", &bw, status);
	read_key<double>(fptr, TDOUBLE, "CHAN_DM", &dm, status);

	fmax = fcenter + bw/2.0;
	fmin = fcenter - bw/2.0;

	int imjd;
	double smjd, offs;
	read_key<int>(fptr, TINT, "STT_IMJD", &imjd, status);
	read_key<double>(fptr, TDOUBLE, "STT_SMJD", &smjd, status);
	read_key<double>(fptr, TDOUBLE, "STT_OFFS", &offs, status);

	t0 = (long double) imjd + (long double)(smjd + offs)/86400.0L;

	char extname[] = "HISTORY";
	fits_movnam_hdu(fptr, BINARY_TBL, extname, 0, status);
	if (*status != 0)
	{
		status[0] = 0;
		return;
	}

	int anynull;
	int dedisp;
	char dds_mtd_c[33];
	read_key_bin(fptr, TINT,    "NSUB", 1, 1, 1, NULL, &nsubint, &anynull, status);
	read_key_bin(fptr, TINT,    "NBIN", 1, 1, 1, NULL, &obs_window, &anynull, status);
	read_key_bin(fptr, TDOUBLE, "TBIN", 1, 1, 1, NULL, &tau, &anynull, status);
	read_key_bin(fptr, TINT,    "NCHAN", 1, 1, 1, NULL, &nchann, &anynull, status);
	read_key_bin(fptr, TDOUBLE, "CHAN_BW", 1, 1, 1, NULL, &bw, &anynull, status);
	read_key_bin(fptr, TDOUBLE, "DM", 1, 1, 1, NULL, &dm, &anynull, status);
	read_key_bin(fptr, TINT, 	"DEDISP", 1, 1, 1, NULL, &dedisp, &anynull, status);
	read_key_bin(fptr, TBYTE, 	"DDS_MTHD", 1, 1, 32, NULL, &dds_mtd_c, &anynull, status);
	read_key_bin(fptr, TDOUBLE, "REF_FREQ", 1, 1, 1, NULL, &fcomp, &anynull, status);


	char char_subint[] = "SUBINT";
	fits_movnam_hdu(fptr, BINARY_TBL, char_subint, 0, status);

	if (*status != 0)
	{
		status[0] = 0;
		return;
	}
	

	if (MODE == "SEARCH")
	{
		read_key<size_t>(fptr, TINT, "NSTOT", &OBS_SIZE, status);
		read_key<size_t>(fptr, TINT, "NSBLK", &nsblk, status);
		read_key<size_t>(fptr, TINT, "NBITS", &nbits, status);

		read_key<bool>(fptr, TINT, "CMPLX", &cmplx, status);
		if (status[0] == 307)
		{
			cmplx = false;
			status[0] = 0;
		}

		obs_window = nsblk*nbits/8;
	}
	else if (MODE == "PSR")
	{
		OBS_SIZE = nsubint*obs_window;
		nbits = 16;

		read_key_bin(
				fptr, TDOUBLE, "PERIOD", 
				1l, 1l, 1, 
				NULL, &period, &anynull, status);
	}


	read_key<int>(fptr, TINT, "SIGNINT", &sign, status);

	// Time since the observation start at
	// the centre of each sub-integration (or row). 
	t_subint.resize(nsubint);
	freqs.resize(nchann);
	

	if (nchann > 1)
	{
		for (int i = 0; i < (int) nsubint; ++i)
			read_key_bin(fptr, TDOUBLE, "OFFS_SUB", i+1, 1, 1, NULL, t_subint.data() + i, &anynull, status);

		// fill in frequency information
		read_key_bin(
				fptr, TDOUBLE, "DAT_FREQ", 
				1, 1, nchann, 
				NULL, freqs.data(), &anynull, status);

		fmin = freqs[0];
		fmax = freqs[nchann - 1];
	}

	tau *= 1e3;

	dds_mtd_c[32] = '\0';
	dds_mthd = std::string(dds_mtd_c);

	dds_mthd.erase(0, dds_mthd.find_first_not_of(" \t\n\r"));
	dds_mthd.erase(dds_mthd.find_last_not_of(" \t\n\r") + 1);

}

PSRFITS::PSRFITS(const std::string& filename_in, size_t buffer_size): 
	BaseReader(), header{}
{

    header_ptr = &header; // <--- CRITICAL: Set the base class's header_ptr member here

	std::filesystem::path p = filename_in;
    filename = p.stem();

	status = 0;
	fits_open_file(&fptr, filename_in.c_str(), READONLY, &status);
	check_status("Opening file");
	is_open = true;

    if (!is_open) 
	{
        throw std::runtime_error("Failed to open file: " + filename);
    }

	header.fill(fptr, &status);
	check_status("Reading file header");


	if (header.MODE == "PSR")
		buf_size = buffer_size / (sizeof(double) + sizeof(int16_t));
	else if (header.MODE == "SEARCH")
		buf_size = buffer_size / (sizeof(double) + sizeof(int8_t));

	buf_size = std::min(header.OBS_SIZE*header.nchann*header.npol, buf_size);

    try 
	{
		if (header.MODE == "PSR")
			raw_data = new int8_t[buf_size*2]; 
		else if (header.MODE == "SEARCH")
			raw_data = new int8_t[buf_size]; 

        buffer = new double[buf_size];   // Allocate main buffer for 'buf_size' doubles
        buf_pos = 0;
        buf_max = 0;
		subint_index = 1;
		subint_pos = 1;

		start_subint_pos = 1;
		start_subint_index = 1;
    } 
	catch (const std::bad_alloc& e) 
	{
        std::cerr << "Allocation failed for main buffers: " << e.what() << std::endl;
        // Clean up partially allocated memory if one allocation succeeds and the other fails
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
        throw; // Re-throw to signal failure
    }

	if (header_ptr->npol > 1)
		throw std::invalid_argument("This progran can not process multy-polarized data yet");


	// Set the file to the data
	char char_subint[] = "SUBINT";
	fits_movnam_hdu(fptr, BINARY_TBL, char_subint, 0, &status);
}

PSRFITS::~PSRFITS() 
{
    if (is_open) 
	{
		fits_close_file(fptr, &status);
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

bool PSRFITS::fill_buffer() 
{

	size_t nsubint = header.nsubint;
	size_t nbin = header.obs_window;
	size_t npol = header.npol;
	size_t nchann = header.nchann;
	size_t c = header.cmplx ? 2 : 1;
	size_t nbits = header.nbits;

	size_t sample_size = npol*nchann*c;
	int KIND = (header.MODE == "SEARCH") ? TBYTE : TSHORT;


	// Shift remaining data (from a previous incomplete 
	// processing chunk) to the front of the buffer
	size_t remaining = buf_max - buf_pos;
	if (remaining > 0) 
	{
		// Move raw data
		std::memmove(raw_data, raw_data + buf_pos, remaining*nbits/8);
		// Move decoded data
		std::memmove(buffer, buffer + buf_pos, remaining * sizeof(double));
	}
	buf_max = remaining; // Update the fill level after shifting
	buf_pos = 0;         // Reset the read position to the beginning of the valid data


	if (subint_index > nsubint) return false;



	// Clamp to actual available space
	size_t data_read_so_far = ((subint_index-1)*nbin + subint_pos-1)*sample_size;
    size_t space_avail = buf_size - buf_max;
	size_t subint_to_read = space_avail / (nbin*sample_size);

	if (data_read_so_far/sample_size >= header_ptr->OBS_SIZE)
		return false;
	if (data_read_so_far/sample_size >= header_ptr->CUT_SIZE)
		return false;



	if (subint_pos > 1)
	{
		read_data(
				fptr, 
				KIND, 
				subint_index, 
				subint_pos,
				raw_data + buf_max,
				buffer + buf_max,
				&status,
				nbin - (subint_pos-1), npol, nchann, c
				);

		buf_max += (nbin - (subint_pos-1))*sample_size;
		subint_index += 1;
		subint_pos = 1;
	}

	for (size_t i = 0; i < subint_to_read; ++i)
	{
		read_data(
				fptr, 
				KIND, 
				subint_index, 
				subint_pos,
				raw_data + buf_max,
				buffer + buf_max,
				&status,
				nbin - (subint_pos-1), npol, nchann, c
				);

		check_status("Filling Buffer");
		buf_max += nbin*sample_size;
		subint_index += 1;
		subint_pos = 1;

		if (subint_index >= nsubint) break; // EOF
		if (buf_max + sample_size > buf_size) break; // buffer is full
	}

	data_read_so_far = ((subint_index-1)*nbin + subint_pos-1)*sample_size;
	space_avail = buf_size - buf_max;


	size_t to_add = std::min(header.OBS_SIZE*sample_size - data_read_so_far, space_avail);
	to_add = std::min(to_add, header.CUT_SIZE*sample_size - data_read_so_far);
	to_add /= sample_size;

	if (to_add)
		read_data(
				fptr, 
				KIND, 
				subint_index, 
				subint_pos,
				raw_data + buf_max,
				buffer + buf_max,
				&status,
				to_add, npol, nchann, c
				);

	buf_max += to_add*sample_size;
	subint_pos = 1+to_add;

	check_status("Filling Buffer");

	return true;
}

double PSRFITS::point2time(size_t point) 
{
	if (header.MODE == "PSR")
	{
		// Number of subint containing the point
		// The time of its middle point
		size_t si = point / header.obs_window;
		double ti = header.t_subint[si];

		// Number of the point iside the subint
		long sp = (point % header.obs_window);

		return ti + header.tau*sp*1.0e-3;
	}
	else 
	{
		return header.tau * point * 1.0e-3;
	}
}


void PSRFITS::skip(double sec) 
{
	size_t steps = sec / (header.tau * 1.0e-3);

	start_subint_index = steps / header.obs_window + 1;
	start_subint_pos = steps % header.obs_window + 1;

	header.t0 += (long double) (steps * (header.tau * 1.0e-3)) / 86400.0L;

	reset();
}


void PSRFITS::set_limit(double t) 
{
	header.CUT_SIZE = size_t (t * 1.0e3 / header.tau);
} 



void PSRFITS::reset() 
{
	buf_max = 0;
	buf_pos = 0;
	
	subint_pos = start_subint_pos;
	subint_index = start_subint_index;
}

bool PSRFITS::allow_1d() 
{
	if (header.nchann > 1)
		return false;
	else if (header.nchann == 1 && header.MODE == "SEARCH")
		return true;

	return false;
}


bool PSRFITS::allow_2d() 
{
	if (header.nchann > 1)
		return true;

	if (header.nchann == 1 && header.MODE == "SEARCH")
		return true;

	return false;
	
}

bool PSRFITS::fill_wts(double* mask, double* fr, size_t nchann) 
{

	int anynull;
	std::vector<float> dat_wts(nchann);

	read_key_bin(
			fptr, TFLOAT, "DAT_WTS", 
			1, 1, nchann, 
			NULL, dat_wts.data(), &anynull, &status);

	check_status("Reading mask information");
	if (status != 0)
		throw std::runtime_error("Can not load channel weights information");

	for (size_t i = 0; i < nchann; ++i)
		mask[i] = double(dat_wts[i]);

	// Read bandpass
	char char_bandpass[] = "BANDPASS";
	fits_movnam_hdu(fptr, BINARY_TBL, char_bandpass, 0, &status);

	std::vector<float> dat_scl(header.npol), dat_offs(header.npol);
	std::vector<int16_t> data(header.npol * header.nchann);

	read_key_bin(
			fptr, TSHORT, "DATA", 
			1, 1, header.nchann * header.npol, 
			NULL, data.data(), &anynull, &status);

	read_key_bin(
			fptr, TFLOAT, "DAT_SCL", 
			1, 1, header.npol, 
			NULL, dat_scl.data(), &anynull, &status);

	read_key_bin(
			fptr, TFLOAT, "DAT_OFFS", 
			1, 1, header.npol, 
			NULL, dat_offs.data(), &anynull, &status);

	//for (size_t i = 0; i < header.npol; ++i)
	for (size_t f = 0; f < header.nchann; ++f)
		fr[f] = double(data[f]) * dat_scl[0] + dat_offs[0];

	char char_subint[] = "SUBINT";
	fits_movnam_hdu(fptr, BINARY_TBL, char_subint, 0, &status);

	
	check_status("Filling bandpass");
	return true;
}
