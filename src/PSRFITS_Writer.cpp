#include "PSRFITS_Writer.h"


#include "aux_math.h"
#include <iostream>
#include <iomanip>
#include <bitset>
#include <algorithm>
#include <cstring> // for memcpy
#include <ctime>
#include <cstdio> // Required for remove()

#include <limits>

// Helper to get current UTC date/time string
std::string getCurrentUTCTime() 
{
    time_t now = time(nullptr);
    char buffer[24];
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", gmtime(&now));
    return std::string(buffer);
}

std::string mjd2utc(long double mjd)
{
	long double jd = mjd + 2400000.5;
	long double jd_unix_epoch = 2440587.5;
	long int unix_time = static_cast<long>((jd - jd_unix_epoch)* 86400.0);
	time_t raw_time = static_cast<time_t>(unix_time);

	// Convert the time_t to a tm structure in UTC
    std::tm* utc_tm = std::gmtime(&raw_time);

    char buffer[24];
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", utc_tm);

    return std::string(buffer);
}	



/*-----------------------
 * Bit magic function
 *------------------------
 */
inline void quantize(double *data, size_t nbin, size_t nchan,
                     char *quantized, size_t nbit,
                     float* dat_scl, float* dat_offs)
{
    // --- 1. Per-thread min/max vectors (properly resized) ---
    thread_local static std::vector<double> min_val, max_val;
    if (min_val.size() != nchan) min_val.resize(nchan);
    if (max_val.size() != nchan) max_val.resize(nchan);

    const double inf = std::numeric_limits<double>::infinity();
    std::fill(min_val.begin(), min_val.end(), inf);
    std::fill(max_val.begin(), max_val.end(), -inf);

    const size_t nsamp = nbin * nchan;
    const double bit_range = static_cast<double>((1ULL << nbit) - 1);

    // --- 2. Compute per-channel min/max ---
    for (size_t i = 0; i < nsamp; ++i) 
	{
        size_t f = i % nchan;
        double val = data[i];
        if (val < min_val[f]) min_val[f] = val;
        if (val > max_val[f]) max_val[f] = val;
    }

    // --- 3. Compute scaling and offset ---
    for (size_t f = 0; f < nchan; ++f) 
	{
		// Signed int → use midpoint
		dat_offs[f] = static_cast<float>((min_val[f] + max_val[f]) / 2.0);
		double range = max_val[f] - min_val[f];
		dat_scl[f] = (range == 0.0) ? 1.0f : static_cast<float>(range / bit_range);
    }

    // --- 4. Normalize data to quantized float range ---
    for (size_t i = 0; i < nsamp; ++i) 
	{
        size_t f = i % nchan;
        data[i] = (data[i] - dat_offs[f]) / dat_scl[f];
    }

	std::fill(quantized, quantized + nsamp * nbit/8, 0.0);

    if (nbit == 16) 
	{
        // Fold mode: one sample = one int16_t (big-endian for FITS)
        for (size_t i = 0; i < nsamp; ++i) 
		{
            int16_t q = std::llround(data[i]);
            uint16_t u = static_cast<uint16_t>(static_cast<int16_t>(q));

            quantized[i*2 + 0] = u & 0xFF;
            quantized[i*2 + 1] = (u >> 8) & 0xFF;
        }
    } 
	else 
	{
        // Search mode: nbit = 1,2,4,8 → pack into bytes, MSB first
        int samples_per_byte = 8 / static_cast<int>(nbit);

        for (size_t i = 0; i < nsamp; ++i) 
		{
            int64_t q = std::llround(data[i]);
            uint8_t sample = static_cast<uint8_t>(q);

            size_t byte_idx = i / samples_per_byte;
            int sample_in_byte = i % samples_per_byte;
            int shift = 8 - nbit * (sample_in_byte + 1); // MSB first

            quantized[byte_idx] |= (sample << shift);
        }
    }
}



void PSRFITS_Writer::check_status(std::string operation)
{
    if (status) 
	{
        char errtext[FLEN_STATUS];
        fits_get_errstatus(status, errtext);
        // You might want to use your logging system here
        fprintf(stderr, "FITS Error during %s: %s\n", operation.c_str(), errtext);
    }
}



PSRFITS_Writer::PSRFITS_Writer(Profile& profile_in, std::string filename)
    : fptr(nullptr), status(0) 
{
    // Create new FITS file
	// ! in the beginning overwrites 
	// the file if it's already exists
    fits_create_file(&fptr, ("!" + filename).c_str(), &status);
	fits_create_img(fptr, BYTE_IMG, 0, 0, &status);

	check_status("Creating FITS file");

	profile = &profile_in;
	header = profile->getHeader();
}

PSRFITS_Writer::~PSRFITS_Writer() 
{
	if (fptr) 
	{
		fits_close_file(fptr, &status);
		check_status("Closing FITS file");
	}
}

bool PSRFITS_Writer::createPrimaryHDU(std::string obs_mode) 
{
    // Basic FITS preamble
    fits_write_key(fptr, TSTRING, "HDRVER", (void*)"6.1", "Header version", &status);
    fits_write_key(fptr, TSTRING, "FITSTYPE", (void*)"PSRFITS", "FITS definition for pulsar data files", &status);

    // File creation date
    std::string date = getCurrentUTCTime();
    fits_write_key(fptr, TSTRING, "DATE", (void*)date.c_str(), "File creation UTC date", &status);

    // Observer/Project info (placeholder - adapt as needed)
    fits_write_key(fptr, TSTRING, "OBSERVER", (void*)"", "Observer name(s)", &status);
    fits_write_key(fptr, TSTRING, "PROJID", (void*)"", "Project name", &status);
    fits_write_key(fptr, TSTRING, "TELESCOP", (void*)"", "Telescope name", &status);

    // Source info
    fits_write_key(fptr, TSTRING, "SRC_NAME", (void*)header->name.c_str(), "", &status);

    // Observation mode (assume folded)
    fits_write_key(fptr, TSTRING, "OBS_MODE", (void*)obs_mode.c_str(), "Observation mode (PSR, CAL, SEARCH)", &status);
	std::string utc_obs = mjd2utc(header->t0);
    fits_write_key(fptr, TSTRING, "DATE-OBS", (void*) utc_obs.c_str(), "UTC date of observation (YYYY-MM-DDThh:mm:ss)", &status);

    // Time info (from BaseHeader)

    // Frequency info
    double obsfreq = (header->fmin + header->fmax) / 2.0; // Simple center freq
    double obsbw = std::abs(header->fmax - header->fmin);
    fits_write_key(fptr, TDOUBLE, "OBSFREQ", &obsfreq, "[MHz] Centre frequency for observation", &status);
    fits_write_key(fptr, TDOUBLE, "OBSBW", &obsbw, "[MHz] Bandwidth for observation", &status);
    fits_write_key(fptr, TINT, "OBSNCHAN", &(header->nchann), "Number of frequency channels", &status);
    fits_write_key(fptr, TDOUBLE, "CHAN_DM", new double(0.0), "[cm-3 pc] DM used for on-line dedispersion", &status);
    fits_write_key(fptr, TINT, "STT_IMJD", new int(header->t0), "[days] Start MJD (UTC)", &status);
    fits_write_key(fptr, TDOUBLE, "STT_SMJD", new double(fmod(header->t0, 1.0)*86400.0), "[s] Start time (sec past UTC 00h)", &status);
    fits_write_key(fptr, TDOUBLE, "STT_OFFS", new double(0.0), "[s] Start time offset", &status);

	check_status("Writing PRIMARY HDU");


    return true;
}


bool PSRFITS_Writer::append_subint_fold(double *data_double, const size_t nbin, const size_t nchan, const size_t npol) 
{
    if (!fptr) 
	{
        std::cerr << "FITS file not initialized." << std::endl;
        return false;
    }

    // Fixed parameters for fold-mode

	char freq_form[32], wts_form[32], offs_form[32], scl_form[32], data_form[32];
	snprintf(freq_form, sizeof(freq_form), "%dD", int(nchan));
	snprintf(wts_form,  sizeof(wts_form),  "%dE", int(nchan));
	snprintf(offs_form, sizeof(offs_form), "%dE", int(nchan * npol)); // usually nchan * 1
	snprintf(scl_form,  sizeof(scl_form),  "%dE", int(nchan * npol));
	snprintf(data_form, sizeof(data_form), "%dI", int(nbin * nchan * npol));

	const char* ttype[] = { "TSUBINT", "OFFS_SUB", "DAT_FREQ", "DAT_WTS", "DAT_OFFS", "DAT_SCL", "DATA" };
	const char* tform[] = { "1D", "1D", freq_form, wts_form, offs_form, scl_form, data_form };
	const char* tunit[] = { "s", "s", "MHz", "", "", "", ""};

	// Create the binary table
    fits_create_tbl(fptr, BINARY_TBL, 1, 7,
                    const_cast<char**>(ttype), // names
                    const_cast<char**>(tform), //sizes and dtypes
                    const_cast<char**>(tunit), // units
                    "SUBINT", &status);

	int naxis = 3;
	long naxes[3] = {long(nbin), long(nchan), long(npol)};
	fits_write_tdim(fptr, 7, naxis, naxes, &status);
	

	// Additional keys
    fits_write_key(fptr, TSTRING, "EPOCHS", (void*) "STT_MJD", "Epoch convention (VALID, MIDTIME, STT_MJD)", &status);
    fits_write_key(fptr, TSTRING, "INT_TYPE", (void*) "TIME", "Time axis (TIME, BINPHSPERI, BINLNGASC, etc)", &status);
    fits_write_key(fptr, TSTRING, "INT_UNIT", (void*) "SEC", "Unit of time axis (SEC, PHS (0-1), DEG)", &status);
    fits_write_key(fptr, TSTRING, "SCALE", (void*) "Counts", "Intensity units (FluxDen/RefFlux/Jansky)", &status);
    fits_write_key(fptr, TSTRING, "POL_TYPE", (void*) "", "Polarisation identifier (e.g., AABBCRCI, AA+BB)", &status);
    fits_write_key(fptr, TINT, "NPOL", new int (1), "Number of polarisations", &status); // Assuming 1 pol
    fits_write_key(fptr, TDOUBLE, "TBIN", new double(header->tau*1.0e-3), "[s] Time per bin/sample", &status);
    fits_write_key(fptr, TINT, "NBIN", (void*) &nbin, "Nr of bins (PSR/CAL mode; else 1)", &status); 
    fits_write_key(fptr, TDOUBLE, "PHS_OFFS", new double(0.0), "Phase offset of bin 0 for gated data", &status); 
    fits_write_key(fptr, TINT, "NCHAN", (void*) &nchan, "Number of channels/sub-bands in this file", &status); 
	double dB = std::abs(header->fmax - header->fmin) / double(nchan);
    fits_write_key(fptr, TDOUBLE, "CHAN_BW", &dB, "[MHz] Channel/sub-band width", &status);
    fits_write_key(fptr, TDOUBLE, "DM", &(header->dm), "[cm-3 pc] DM used for dedispersion", &status);
    fits_write_key(fptr, TDOUBLE, "RM", new double(0.0), "[rad m-2] RM for post-detection deFaraday", &status);

	check_status("Creating SUBINT (dynamic profile) bin table");

    if (!profile->dyn) 
	{
        std::cerr << "Profile data (dyn) is null." << std::endl;
        return false;
    }

    // --- 2. Compute DAT_OFFS and DAT_SCL per channel
    std::vector<float> dat_offs(nchan * npol);
    std::vector<float> dat_scl(nchan * npol);
    std::vector<char> data_int(2*nbin * nchan * npol);

	quantize(data_double, nbin, nchan, 
			data_int.data(), 16,
			dat_scl.data(), dat_offs.data());


    // --- 3. Prepare frequency array (linear spacing)
    std::vector<double> dat_freq(nchan);
    double df = (header->fmax - header->fmin) / nchan;
    for (size_t i = 0; i < nchan; ++i) 
        dat_freq[i] = header->fmin + (i + 0.5) * df;

    // --- 4. Prepare mask (DAT_WTS)
    std::vector<float> dat_wts(nchan);
    if (profile->mask) 
	{
        for (size_t i = 0; i < nchan; ++i) 
            dat_wts[i] = static_cast<float>(profile->mask[i]);

		// Clamp to [0,1] per PSRFITS spec
        for (size_t i = 0; i < nchan; ++i) 
            dat_wts[i] = std::clamp(dat_wts[i], 0.0f, 1.0f);
    } 
	else 
	{
        std::fill(dat_wts.begin(), dat_wts.end(), 1.0f);
    }

    // --- 5. Write to SUBINT table (row = subint_index + 1, since CFITSIO is 1-indexed)
	size_t subint_index = 0;
    long row = static_cast<long>(subint_index + 1);

    // TSUBINT and OFFS_SUB 
    double tsubint = header->tau * nbin * 1.0e-3; // Total subint duration
    double offs_sub = (subint_index + 0.5) * tsubint; // Center of subint
									
    fits_write_col(fptr, TDOUBLE, 1, row, 1, 1, &tsubint, &status);
    fits_write_col(fptr, TDOUBLE, 2, row, 1, 1, &offs_sub, &status);

    // DAT_FREQ, DAT_WTS
    fits_write_col(fptr, TDOUBLE, 3, row, 1, nchan, dat_freq.data(), &status);
    fits_write_col(fptr, TFLOAT, 4, row, 1, nchan, dat_wts.data(), &status);

    // DAT_OFFS, DAT_SCL
    fits_write_col(fptr, TFLOAT, 5, row, 1, nchan * npol, dat_offs.data(), &status);
    fits_write_col(fptr, TFLOAT, 6, row, 1, nchan * npol, dat_scl.data(), &status);

    // DATA (int16)
    fits_write_col(fptr, TSHORT, 7, row, 1, nbin * nchan * npol, data_int.data(), &status);

	check_status("Writing SUBINT bintable (dynamic spectrum)");

    return true;
}


bool PSRFITS_Writer::append_subint_stream(std::string stream_file, const size_t nchan, const size_t npol, bool cmp)
{
    if (!fptr) 
	{
        std::cerr << "FITS file not initialized." << std::endl;
        return false;
    }

	int nsblk = 4096;
	int nbits = 8;

	std::ifstream stream(stream_file);
    stream.seekg(0, std::ios::end);
    long nstot = static_cast<long>(stream.tellg()) / sizeof(double);
	nstot = nstot / (nchan*npol);
    stream.seekg(0, std::ios::beg);

	if (cmp)
		nstot = nstot / 2;

	long nsubint = nstot / (nsblk*nbits/8);
	if ((nstot/nchan/npol) % (nsblk*nbits/8) != 0)
		nsubint += 1;

    // Fixed parameters for search-mode

	char freq_form[32], wts_form[32], offs_form[32], scl_form[32], data_form[32];
	snprintf(freq_form, sizeof(freq_form), "%dD", int(nchan));
	snprintf(wts_form,  sizeof(wts_form),  "%dE", int(nchan));
	snprintf(offs_form, sizeof(offs_form), "%dE", int(nchan * npol)); // usually nchan * 1
	snprintf(scl_form,  sizeof(scl_form),  "%dE", int(nchan * npol));
	snprintf(data_form, sizeof(data_form), "%dB", int((nsblk*nbits/8) * nchan * npol));

	const char* ttype[] = { "TSUBINT", "OFFS_SUB", "DAT_FREQ", "DAT_WTS", "DAT_OFFS", "DAT_SCL", "DATA" };
	const char* tform[] = { "1D", "1D", freq_form, wts_form, offs_form, scl_form, data_form };
	const char* tunit[] = { "s", "s", "MHz", "", "", "", ""};

	// Create the binary table
    fits_create_tbl(fptr, BINARY_TBL, nsubint, 7,
                    const_cast<char**>(ttype), // names
                    const_cast<char**>(tform), //sizes and dtypes
                    const_cast<char**>(tunit), // units
                    "SUBINT", &status);

	int naxis = 3;
	long naxes[3] = {long(nchan), long(npol), long(nsblk * nbits/8)};
	fits_write_tdim(fptr, 7, naxis, naxes, &status);
	
	

	// Additional keys
    fits_write_key(fptr, TSTRING, "EPOCHS", (void*) "STT_MJD", "Epoch convention (VALID, MIDTIME, STT_MJD)", &status);
    fits_write_key(fptr, TSTRING, "INT_TYPE", (void*) "TIME", "Time axis (TIME, BINPHSPERI, BINLNGASC, etc)", &status);
    fits_write_key(fptr, TSTRING, "INT_UNIT", (void*) "SEC", "Unit of time axis (SEC, PHS (0-1), DEG)", &status);
    fits_write_key(fptr, TSTRING, "SCALE", (void*) "Counts", "Intensity units (FluxDen/RefFlux/Jansky)", &status);
    fits_write_key(fptr, TSTRING, "POL_TYPE", (void*) "", "Polarisation identifier (e.g., AABBCRCI, AA+BB)", &status);
    fits_write_key(fptr, TINT, "NPOL", new int (1), "Number of polarisations", &status); // Assuming 1 pol
    fits_write_key(fptr, TDOUBLE, "TBIN", new double(header->tau*1.0e-3), "[s] Time per bin/sample", &status);
    fits_write_key(fptr, TINT, "NBIN", new int(1), "Nr of bins (PSR/CAL mode; else 1)", &status); 
    fits_write_key(fptr, TDOUBLE, "PHS_OFFS", new double(0.0), "Phase offset of bin 0 for gated data", &status); 
    fits_write_key(fptr, TINT, "NBITS", &nbits, "Nr of bits/datum (SEARCH mode data, else 1)", &status); 
    fits_write_key(fptr, TDOUBLE, "ZERO_OFF", new double(0.0), "Zero offset for SEARCH-mode data", &status);
    fits_write_key(fptr, TINT, "SIGNINT", new int(1), "1 for signed ints in SEARCH-mode data, else 0", &status); 
    fits_write_key(fptr, TINT, "NCHAN", (void*) &nchan, "Number of channels/sub-bands in this file", &status); 
	double dB = std::abs(header->fmax - header->fmin) / double(nchan);
    fits_write_key(fptr, TDOUBLE, "CHAN_BW", &dB, "[MHz] Channel/sub-band width", &status);
    fits_write_key(fptr, TDOUBLE, "DM", &(header->dm), "[cm-3 pc] DM used for dedispersion", &status);
    fits_write_key(fptr, TDOUBLE, "RM", new double(0.0), "[rad m-2] RM for post-detection deFaraday", &status);
    fits_write_key(fptr, TINT, "NSBLK", &nsblk, "Samples/row (SEARCH mode, else 1)", &status); 
    fits_write_key(fptr, TINT, "NSTOT", &nstot, "Total number of samples (SEARCH mode, else 1)", &status); 


    // Prepare frequency array (linear spacing)
    std::vector<double> dat_freq(nchan);
    double df = (header->fmax - header->fmin) / nchan;
    for (size_t i = 0; i < nchan; ++i) 
        dat_freq[i] = header->fmin + (i + 0.5) * df;

    // Prepare mask (DAT_WTS)
    std::vector<float> dat_wts(nchan);
    if (profile->mask) 
	{
        for (size_t i = 0; i < nchan; ++i) 
            dat_wts[i] = static_cast<float>(profile->mask[i]);

		// Clamp to [0,1] per PSRFITS spec
        for (size_t i = 0; i < nchan; ++i) 
            dat_wts[i] = std::clamp(dat_wts[i], 0.0f, 1.0f);
    } 
	else 
	{
        std::fill(dat_wts.begin(), dat_wts.end(), 1.0f);
    }


    // Prepare offset and double arrays for data
    std::vector<float> dat_offs(nchan * npol);
    std::vector<float> dat_scl(nchan * npol);
    std::vector<char> data_int((nsblk*nbits/8) * nchan * npol);
	double *data_double = new double[nsblk*nchan*npol];

	size_t actually_read = 0;


	for (int row = 1; row < nsubint+1; ++row)
	{
		stream.read(reinterpret_cast<char*>(data_double), sizeof(double)*nsblk*nchan*npol);

		// Check how many bytes were actually read
		// transform it to bin steps
		actually_read = static_cast<size_t>(stream.gcount());
		actually_read /= (sizeof(double) * nchan * npol);

		if (actually_read < size_t(nsblk))
			std::fill(data_double + actually_read*nchan*npol, data_double + nsblk*nchan*npol, 0.0);


		quantize(data_double, actually_read, nchan, 
				data_int.data(), nbits,
				dat_scl.data(), dat_offs.data());

		// TSUBINT and OFFS_SUB 
		double tsubint = header->tau * actually_read * 1.0e-3; // Total subint duration
		double offs_sub = (double(row) - 0.5) * tsubint; // Center of subint

		fits_write_col(fptr, TDOUBLE, 1, row, 1, 1, &tsubint, &status);
		fits_write_col(fptr, TDOUBLE, 2, row, 1, 1, &offs_sub, &status);

		// DAT_FREQ, DAT_WTS
		fits_write_col(fptr, TDOUBLE, 3, row, 1, nchan, dat_freq.data(), &status);
		fits_write_col(fptr, TFLOAT, 4, row, 1, nchan, dat_wts.data(), &status);

		// DAT_OFFS, DAT_SCL
		fits_write_col(fptr, TFLOAT, 5, row, 1, nchan * npol, dat_offs.data(), &status);
		fits_write_col(fptr, TFLOAT, 6, row, 1, nchan * npol, dat_scl.data(), &status);

		// DATA (int16)
		fits_write_col(fptr, TBYTE, 7, row, 1, (actually_read*nbits/8) * nchan * npol, (void*) data_int.data(), &status);
	}





	stream.close();
	std::remove(stream_file.c_str());
	delete[] data_double;

	return true;
}

bool PSRFITS_Writer::append_subint_search(double* data_double, const size_t nbin, const size_t nchan, const size_t npol, bool cmp)
{
    if (!fptr) 
	{
        std::cerr << "FITS file not initialized." << std::endl;
        return false;
    }

	int nsblk = 4096;
	int nbits = 8;

	size_t nstot = nbin;
	if (cmp)
		nstot = nstot * 2;

	long nsubint = nstot / nsblk;
	if ((nstot/nchan/npol) % nsblk != 0)
		nsubint += 1;

    // Fixed parameters for search-mode
	char freq_form[32], wts_form[32], offs_form[32], scl_form[32], data_form[32];
	snprintf(freq_form, sizeof(freq_form), "%dD", int(nchan));
	snprintf(wts_form,  sizeof(wts_form),  "%dE", int(nchan));
	snprintf(offs_form, sizeof(offs_form), "%dE", int(nchan * npol)); // usually nchan * 1
	snprintf(scl_form,  sizeof(scl_form),  "%dE", int(nchan * npol));
	snprintf(data_form, sizeof(data_form), "%dB", int((nsblk*nbits/8) * nchan * npol));

	const char* ttype[] = { "TSUBINT", "OFFS_SUB", "DAT_FREQ", "DAT_WTS", "DAT_OFFS", "DAT_SCL", "DATA" };
	const char* tform[] = { "1D", "1D", freq_form, wts_form, offs_form, scl_form, data_form };
	const char* tunit[] = { "s", "s", "MHz", "", "", "", ""};

	// Create the binary table
    fits_create_tbl(fptr, BINARY_TBL, nsubint, 7,
                    const_cast<char**>(ttype), // names
                    const_cast<char**>(tform), //sizes and dtypes
                    const_cast<char**>(tunit), // units
                    "SUBINT", &status);

	int naxis = 3;
	long naxes[3] = {long(nchan), long(npol), long(nsblk * nbits/8)};
	fits_write_tdim(fptr, 7, naxis, naxes, &status);
	
	

	// Additional keys
    fits_write_key(fptr, TSTRING, "EPOCHS", (void*) "STT_MJD", "Epoch convention (VALID, MIDTIME, STT_MJD)", &status);
    fits_write_key(fptr, TSTRING, "INT_TYPE", (void*) "TIME", "Time axis (TIME, BINPHSPERI, BINLNGASC, etc)", &status);
    fits_write_key(fptr, TSTRING, "INT_UNIT", (void*) "SEC", "Unit of time axis (SEC, PHS (0-1), DEG)", &status);
    fits_write_key(fptr, TSTRING, "SCALE", (void*) "Counts", "Intensity units (FluxDen/RefFlux/Jansky)", &status);
    fits_write_key(fptr, TSTRING, "POL_TYPE", (void*) "", "Polarisation identifier (e.g., AABBCRCI, AA+BB)", &status);
    fits_write_key(fptr, TINT, "NPOL", new int (1), "Number of polarisations", &status); // Assuming 1 pol
    fits_write_key(fptr, TDOUBLE, "TBIN", new double(header->tau*1.0e-3), "[s] Time per bin/sample", &status);
    fits_write_key(fptr, TINT, "NBIN", new int(1), "Nr of bins (PSR/CAL mode; else 1)", &status); 
    fits_write_key(fptr, TDOUBLE, "PHS_OFFS", new double(0.0), "Phase offset of bin 0 for gated data", &status); 
    fits_write_key(fptr, TINT, "NBITS", &nbits, "Nr of bits/datum (SEARCH mode data, else 1)", &status); 
    fits_write_key(fptr, TDOUBLE, "ZERO_OFF", new double(0.0), "Zero offset for SEARCH-mode data", &status);
    fits_write_key(fptr, TINT, "SIGNINT", new int(1), "1 for signed ints in SEARCH-mode data, else 0", &status); 
    fits_write_key(fptr, TINT, "NCHAN", (void*) &nchan, "Number of channels/sub-bands in this file", &status); 
	double dB = std::abs(header->fmax - header->fmin) / double(nchan);
    fits_write_key(fptr, TDOUBLE, "CHAN_BW", &dB, "[MHz] Channel/sub-band width", &status);
    fits_write_key(fptr, TDOUBLE, "DM", &(header->dm), "[cm-3 pc] DM used for dedispersion", &status);
    fits_write_key(fptr, TDOUBLE, "RM", new double(0.0), "[rad m-2] RM for post-detection deFaraday", &status);
    fits_write_key(fptr, TINT, "NSBLK", &nsblk, "Samples/row (SEARCH mode, else 1)", &status); 
    fits_write_key(fptr, TINT, "NSTOT", &nstot, "Total number of samples (SEARCH mode, else 1)", &status); 


    // Prepare frequency array (linear spacing)
    std::vector<double> dat_freq(nchan);
    double df = (header->fmax - header->fmin) / nchan;
    for (size_t i = 0; i < nchan; ++i) 
        dat_freq[i] = header->fmin + (i + 0.5) * df;

    // Prepare mask (DAT_WTS)
    std::vector<float> dat_wts(nchan);
    if (profile->mask) 
	{
        for (size_t i = 0; i < nchan; ++i) 
            dat_wts[i] = static_cast<float>(profile->mask[i]);

		// Clamp to [0,1] per PSRFITS spec
        for (size_t i = 0; i < nchan; ++i) 
            dat_wts[i] = std::clamp(dat_wts[i], 0.0f, 1.0f);
    } 
	else 
	{
        std::fill(dat_wts.begin(), dat_wts.end(), 1.0f);
    }


    // Prepare offset and double arrays for data
    std::vector<float> dat_offs(nchan * npol);
    std::vector<float> dat_scl(nchan * npol);
    std::vector<char> data_int(nsblk / (8/nbits) * nchan * npol);

	size_t actually_read = 0;


	for (int row = 1; row < nsubint+1; ++row)
	{

		// Check how many time steps are available
		actually_read = std::min(size_t(nsblk), nbin - nsblk*row);
		actually_read = std::min(actually_read, nbin);
		
		quantize(data_double + (row-1)*actually_read*nchan*npol, 
				actually_read, nchan, 
				data_int.data(), nbits,
				dat_scl.data(), dat_offs.data());


		// TSUBINT and OFFS_SUB 
		double tsubint = header->tau * actually_read * 1.0e-3; // Total subint duration
		double offs_sub = (double(row) - 0.5) * tsubint; // Center of subint

		fits_write_col(fptr, TDOUBLE, 1, row, 1, 1, &tsubint, &status);
		fits_write_col(fptr, TDOUBLE, 2, row, 1, 1, &offs_sub, &status);

		// DAT_FREQ, DAT_WTS
		fits_write_col(fptr, TDOUBLE, 3, row, 1, nchan, dat_freq.data(), &status);
		fits_write_col(fptr, TFLOAT, 4, row, 1, nchan, dat_wts.data(), &status);

		// DAT_OFFS, DAT_SCL
		fits_write_col(fptr, TFLOAT, 5, row, 1, nchan * npol, dat_offs.data(), &status);
		fits_write_col(fptr, TFLOAT, 6, row, 1, nchan * npol, dat_scl.data(), &status);

		// DATA
		fits_write_col(fptr, TBYTE, 7, row, 1, (actually_read*nbits/8) * nchan * npol, (void*) data_int.data(), &status);
	}

	return true;
}
