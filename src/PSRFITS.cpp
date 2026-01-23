#include "PSRFITS.h" // Include the header file defining the class interface and base classes
#include "aux_math.h"

#include <stdexcept>  // For std::runtime_error, std::invalid_argument, std::bad_alloc
#include <iostream>   // For std::cerr, std::endl
#include <iomanip>    // For std::setprecision
#include <cstring>    // For std::memcpy, std::memmove, std::strlcpy
#include <limits>     // For std::numeric_limits (if needed for validation)
#include <algorithm>  // For std::min, std::fill_n, std::remove_if
#include <cctype>     // For std::isspace
#include <fftw3.h>    // For FFTW library types (fftw_complex, fftw_plan)
#include <filesystem> // For std::filesystem::path.stem()

void PSRFITS::check_status(std::string operation)
{
    if (status) 
	{
        char errtext[FLEN_STATUS];
        fits_get_errstatus(status, errtext);
        // You might want to use your logging system here
        fprintf(stderr, "FITS Error during %s: %s\n", operation.c_str(), errtext);
    }
}

void read_key_bin(fitsfile *fptr, int datatype, std::string key, long firstrow, long firstelem, long nelements, void *nulval, void *array, int *anynull, int *status)
{
	int colum; // column number in the table

	fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(key.c_str()), &colum, status);
	fits_read_col(fptr, datatype, colum, firstrow, firstelem, nelements, nulval, array, anynull, status);
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

PSRFITSHeader::PSRFITSHeader(): 
	BaseHeader()
{
	// All members are initialized in the member 
	// initializer list above as defaut values
}
void PSRFITSHeader::fill(fitsfile *fptr, int *status)
{
	// Read information from prymary header
	// It hal lower priority than the HISTORY 
	// bintable, but may be used if it is absent
	char str[FLEN_VALUE];
	fits_read_key_str(fptr, "OBS_MODE", str, NULL, status);
	MODE = std::string(str);

	fits_read_key(fptr, TINT, "NRCVR", &npol, NULL, status);
	fits_read_key(fptr, TINT, "OBSNCHAN", &nchann, NULL, status);

	double fcenter, bw;
	fits_read_key(fptr, TDOUBLE, "OBSFREQ", &fcenter, NULL, status);
	fits_read_key(fptr, TDOUBLE, "OBSBW", &bw, NULL, status);
	fits_read_key(fptr, TDOUBLE, "CHAN_DM", &dm, NULL, status);

	fmax = fcenter + bw/2.0;
	fmin = fcenter - bw/2.0;

	int imjd;
	double smjd, offs;
	fits_read_key(fptr, TINT, "STT_IMJD", &imjd, NULL, status);
	fits_read_key(fptr, TDOUBLE, "STT_SMJD", &smjd, NULL, status);
	fits_read_key(fptr, TDOUBLE, "STT_OFFS", &offs, NULL, status);

	t0 = (long double) imjd + (long double)(smjd + offs)/86400.0L;


	char extname[] = "HISTORY";
	fits_movnam_hdu(fptr, BINARY_TBL, extname, 0, status);
	
	if (*status != 0)
		return;

	int anynull;
	int dedisp;
	char dds_mtd[33];
	read_key_bin(fptr, TINT,    "NSUB", 1, 1, 1, NULL, &nsubint, &anynull, status);
	read_key_bin(fptr, TINT,    "NBIN", 1, 1, 1, NULL, &obs_window, &anynull, status);
	read_key_bin(fptr, TDOUBLE, "TBIN", 1, 1, 1, NULL, &tau, &anynull, status);
	read_key_bin(fptr, TINT,    "NCHAN", 1, 1, 1, NULL, &nchann, &anynull, status);
	read_key_bin(fptr, TDOUBLE, "CHAN_BW", 1, 1, 1, NULL, &bw, &anynull, status);
	read_key_bin(fptr, TDOUBLE, "DM", 1, 1, 1, NULL, &dm, &anynull, status);
	read_key_bin(fptr, TINT, 	"DEDISP", 1, 1, 1, NULL, &dedisp, &anynull, status);
	read_key_bin(fptr, TBYTE, 	"DDS_MTHD", 1, 1, 32, NULL, &dds_mtd, &anynull, status);
	read_key_bin(fptr, TDOUBLE, "REF_FREQ", 1, 1, 1, NULL, &fcomp, &anynull, status);


	char char_subint[] = "SUBINT";
	fits_movnam_hdu(fptr, BINARY_TBL, char_subint, 0, status);





	if (MODE == "SEARCH")
	{
		fits_read_key(fptr, TINT, "NSTOT", &OBS_SIZE, NULL, status);
		fits_read_key(fptr, TINT, "CMPLX", &cmplx, NULL, status);
		fits_read_key(fptr, TINT, "NSBLK", &nsblk, NULL, status);
		fits_read_key(fptr, TINT, "NBITS", &nbits, NULL, status);

		obs_window = nsblk*nbits/8;
	}
	else if (MODE == "PSR")
	{
		OBS_SIZE = nsubint*obs_window;
		nbits = 16;
	}

	fits_read_key(fptr, TINT, "SIGNINT", &sign, NULL, status);

	// timing information about every subint
	t_subint = new double[nsubint];

	for (size_t i = 0; i < nsubint; ++i)
		read_key_bin(fptr, TDOUBLE, "OFFS_SUB", 1, 1, 1, NULL, t_subint + i, &anynull, status);

	tau *= 1e3;
	dds_mtd[32] = '\0';
}

void PSRFITSHeader::print() const
{ // TDIM# = (*,*,*) / (NBIN,NCHAN,NPOL) or (NCHAN,NPOL,NSBLK*NBITS/8)

	std::cout << "=== PSRFITS HEADER ===" << std::endl;
	std::cout << "MODE     " << MODE << std::endl;
	std::cout << "MJD t0   " << std::setprecision(17) << t0 << std::endl;
	std::cout << "tau      " << std::setprecision(7) << tau << " ms"<< std::endl;
	std::cout << "fmin     " << fmin << " MHz"<< std::endl;
	std::cout << "fmax     " << fmax << " MHz"<< std::endl;
	std::cout << "nbin     " << nsubint << "*" << obs_window << std::endl;
	std::cout << "nchann   " << nchann << std::endl;
	std::cout << "npol     " << npol << std::endl;
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
	int anynull;

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


	if (subint_index >= nsubint) return false;



	// Clamp to actual available space
	size_t data_read_so_far = ((subint_index-1)*nbin + subint_pos-1)*sample_size;
    size_t space_avail = buf_size - buf_max;
	size_t subint_to_read = (space_avail - (nbin-subint_pos+1)*sample_size) / nbin/sample_size;

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

		buf_max += nbin*sample_size;
		subint_index += 1;
		subint_pos = 1;

		if (subint_index >= nsubint) break; // EOF
		if (buf_max + sample_size > buf_size) break; // buffer is full
	}

	data_read_so_far = ((subint_index-1)*nbin + subint_pos-1)*sample_size;
	space_avail = buf_size - buf_max;

	size_t to_add = std::min(header.OBS_SIZE*sample_size - data_read_so_far, space_avail);
	to_add /= sample_size;


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

	std::ofstream test("data/test.bin");
	test.write((char*) buffer, sizeof(double)*buf_max);
	test.close();

	return false;
}

double PSRFITS::point2time(size_t point) {return 0.0;}
void PSRFITS::skip(double sec) {}
void PSRFITS::set_limit(double t) {} 
void PSRFITS::reset() {} 
bool PSRFITS::allow_1d() {return true;}
bool PSRFITS::allow_2d() {return true;}
