#include "PSRFITS_Writer.h"
#include "aux_math.h"
#include "fitsio.h"

#include <cstddef>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

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

    std::tm* utc_tm = std::gmtime(&raw_time);
    char buffer[24];
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", utc_tm);
    return std::string(buffer);
}

char* convert_str(std::string str, size_t len)
{
    static char result[256];
    std::copy(str.begin(), str.end(), result);
    std::fill(result + str.length(), result + len, ' ');
    return result;
}

/*-----------------------
 * Bit magic function
 *------------------------
 */
inline void quantize(
        double* data,
        size_t ntime,
        size_t nchan,
        size_t npol,
        char* quantized,
        size_t nbit,
        float* dat_scl,
        float* dat_offs,
        bool cmp = false)
{
	if (!data)
		return;

    size_t c = cmp ? 2 : 1;
    size_t nsamp_total = ntime * nchan * npol * c;

    const double bit_range = static_cast<double>((1ULL << nbit) - 1);
    const double inf = std::numeric_limits<double>::infinity();

    size_t stats_size = nchan * npol * c;
    thread_local static std::vector<double> min_val, max_val;
    if (min_val.size() != stats_size)
    {
        min_val.resize(stats_size, inf);
        max_val.resize(stats_size, -inf);
    }
    std::fill(min_val.begin(), min_val.end(), inf);
    std::fill(max_val.begin(), max_val.end(), -inf);

    for (size_t t = 0; t < ntime; ++t) {
        for (size_t p = 0; p < npol; ++p) {
            for (size_t f = 0; f < nchan; ++f) {
                for (size_t comp = 0; comp < c; ++comp) {
                    size_t idx = ((t * npol + p) * nchan + f) * c + comp;
                    double val = data[idx];
                    size_t stat_idx = (f * npol + p) * c + comp;
                    if (val < min_val[stat_idx]) min_val[stat_idx] = val;
                    if (val > max_val[stat_idx]) max_val[stat_idx] = val;
                }
            }
        }
    }

    for (size_t f = 0; f < nchan; ++f) {
        for (size_t p = 0; p < npol; ++p) {
            for (size_t comp = 0; comp < c; ++comp) {
                size_t stat_idx = (f * npol + p) * c + comp;
                dat_offs[stat_idx] = static_cast<float>((min_val[stat_idx] + max_val[stat_idx]) / 2.0);
                double range = max_val[stat_idx] - min_val[stat_idx];
                dat_scl[stat_idx] = (range == 0.0) ? 1.0f : static_cast<float>(range / bit_range);
            }
        }
    }

    size_t nbytes = (nsamp_total * nbit + 7) / 8;
    std::fill(quantized, quantized + nbytes, 0);

    if (nbit == 16)
    {
        for (size_t t = 0; t < ntime; ++t)
        {
            for (size_t p = 0; p < npol; ++p)
            {
                for (size_t f = 0; f < nchan; ++f)
                {
                    for (size_t comp = 0; comp < c; ++comp)
                    {
                        size_t i = ((t * npol + p) * nchan + f) * c + comp;
                        size_t stat_idx = (f * npol + p) * c + comp;

                        double norm = (data[i] - dat_offs[stat_idx]) / dat_scl[stat_idx];
                        int16_t q = static_cast<int16_t>(
								std::clamp(
									std::llround(norm), 
									-32768LL, 32767LL));
                        uint16_t u = static_cast<uint16_t>(q);
                        quantized[i * 2 + 0] = u & 0xFF;
                        quantized[i * 2 + 1] = (u >> 8) & 0xFF;
                    }
                }
            }
        }
    }
    else
    {
        int samples_per_byte = 8 / static_cast<int>(nbit);
        long long left = -(1LL << (nbit-1));
        long long right = (1LL << (nbit-1)) - 1LL;

        for (size_t t = 0; t < ntime; ++t)
        {
            for (size_t p = 0; p < npol; ++p)
            {
                for (size_t f = 0; f < nchan; ++f)
                {
                    for (size_t comp = 0; comp < c; ++comp)
                    {
                        size_t i = ((t * npol + p) * nchan + f) * c + comp;
                        size_t stat_idx = (f * npol + p) * c + comp;

                        double norm = (data[i] - dat_offs[stat_idx]) / dat_scl[stat_idx];
                        uint8_t sample = static_cast<uint8_t>(
                                std::clamp(std::llround(norm), left, right));

                        size_t byte_idx = i / samples_per_byte;
                        int pos_in_byte = i % samples_per_byte;
                        int shift = 8 - nbit * (pos_in_byte + 1);
                        quantized[byte_idx] |= (sample << shift);
                    }
                }
            }
        }
    }
}

void PSRFITS_Writer::check_status(std::string operation)
{
    if (status)
    {
        char errtext[FLEN_STATUS];
        fits_get_errstatus(status, errtext);
        fprintf(stderr, "FITS Error during %s: %s\n", operation.c_str(), errtext);
    }
}

PSRFITS_Writer::PSRFITS_Writer(std::string filename)
    : fptr(nullptr), status(0)
{
    fits_create_file(&fptr, ("!" + filename + ".fits").c_str(), &status);
    fits_create_img(fptr, BYTE_IMG, 0, 0, &status);
    check_status("Creating FITS file");
}

PSRFITS_Writer::~PSRFITS_Writer()
{
    if (fptr)
    {
        fits_close_file(fptr, &status);
        check_status("Closing FITS file");
    }
}

bool PSRFITS_Writer::createPrimaryHDU(std::string obs_mode, const BaseHeader* header)
{
	hdr = header[0];
	hdr.MODE = obs_mode;

    fits_write_key(fptr, TSTRING, "COMMENT", (void*)"FITS (Flexible Image Transport System) format defined in Astronomy and", nullptr, &status);
    fits_write_key(fptr, TSTRING, "COMMENT", (void*)"Astrophysics Supplement Series v44/p363, v44/p371, v73/p359, v73/p365.", nullptr, &status);
    fits_write_key(fptr, TSTRING, "COMMENT", (void*)"Contact the NASA Science Office of Standards and Technology for the", nullptr, &status);
    fits_write_key(fptr, TSTRING, "COMMENT", (void*)"FITS Definition document #100 and other FITS information.", nullptr, &status);

    fits_write_key(fptr, TSTRING, "HDRVER", (void*)"6.1", "Header version", &status);
    fits_write_key(fptr, TSTRING, "FITSTYPE", (void*)"PSRFITS", "FITS definition for pulsar data files", &status);

    std::string date = getCurrentUTCTime();
    fits_write_key(fptr, TSTRING, "DATE", (void*)date.c_str(), "File creation UTC date", &status);

    fits_write_key(fptr, TSTRING, "OBSERVER", (void*)hdr.observer.c_str(), "Observer name(s)", &status);
    fits_write_key(fptr, TSTRING, "PROJID", (void*)hdr.proj_id.c_str(), "Project name", &status);
    fits_write_key(fptr, TSTRING, "TELESCOP", (void*)hdr.telescop.c_str(), "Telescope name", &status);

    // REFACTORED: Use stack variables instead of new
    float ant_x = hdr.ant_x;
    float ant_y = hdr.ant_y;
    float ant_z = hdr.ant_z;
    fits_write_key(fptr, TFLOAT, "ANT_X", &ant_x, "[m] Antenna ITRF X-coordinate (D)", &status);
    fits_write_key(fptr, TFLOAT, "ANT_Y", &ant_y, "[m] Antenna ITRF Y-coordinate (D)", &status);
    fits_write_key(fptr, TFLOAT, "ANT_Z", &ant_z, "[m] Antenna ITRF Z-coordinate (D)", &status);

    fits_write_key(fptr, TSTRING, "FRONTEND", (void*)"", "Receiver ID", &status);

    // REFACTORED: Use stack variable
    int nrcvr = hdr.npol;
    fits_write_key(fptr, TINT, "NRCVR", &nrcvr, "Number of receiver polarisation channels", &status);

    fits_write_key(fptr, TSTRING, "FD_POLN", (void*)hdr.fd_poln.c_str(), "LIN or CIRC", &status);
    fits_write_key(fptr, TSTRING, "BACKEND", (void*)hdr.backend.c_str(), "Backend ID", &status);

    // REFACTORED: Use stack variable
    float equinox = hdr.equinox;
    fits_write_key(fptr, TFLOAT, "EQUINOX", &equinox, "Equinox of coords (e.g. 2000.0)", &status);

    fits_write_key(fptr, TSTRING, "RA", (void*)hdr.RA.c_str(), "Right ascension (hh:mm:ss.ssss)", &status);
    fits_write_key(fptr, TSTRING, "DEC", (void*)hdr.DEC.c_str(), "Declination (-dd:mm:ss.sss)", &status);

    // REFACTORED: Use stack variables
    float bmaj = 0.0f;
    float bmin = 0.0f;
    float bpa = 0.0f;
    fits_write_key(fptr, TFLOAT, "BMAJ", &bmaj, "[deg] Beam major axis length", &status);
    fits_write_key(fptr, TFLOAT, "BMIN", &bmin, "[deg] Beam minor axis length", &status);
    fits_write_key(fptr, TFLOAT, "BPA", &bpa, "[deg] Beam position angle", &status);

    fits_write_key(fptr, TSTRING, "TRK_MODE", (void*)"", "Track mode (TRACK, SCANGC, SCANLAT)", &status);

    fits_write_key(fptr, TSTRING, "SRC_NAME", (void*)header->name.c_str(), "", &status);
    fits_write_key(fptr, TSTRING, "OBS_MODE", (void*)obs_mode.c_str(), "Observation mode (PSR, CAL, SEARCH)", &status);

    std::string utc_obs = mjd2utc(header->t0);
    fits_write_key(fptr, TSTRING, "DATE-OBS", (void*) utc_obs.c_str(), "UTC date of observation (YYYY-MM-DDThh:mm:ss)", &status);

    double obsfreq = (header->fmax + header->fmin)/2.0;
    double obsbw = std::abs(header->fmax - header->fmin);
    int imjd = int(header->t0);
    double smjd = (header->t0 - double(imjd))*86400.0L;

    fits_write_key(fptr, TDOUBLE, "OBSFREQ", &obsfreq, "[MHz] Centre frequency for observation", &status);
    fits_write_key(fptr, TDOUBLE, "OBSBW", &obsbw, "[MHz] Bandwidth for observation", &status);
    fits_write_key(fptr, TINT, "OBSNCHAN", (void*)&(header->nchann), "Number of frequency channels", &status);

    // REFACTORED: Use stack variable
    double chan_dm = 0.0;
    fits_write_key(fptr, TDOUBLE, "CHAN_DM", &chan_dm, "[cm-3 pc] DM used for on-line dedispersion", &status);

    fits_write_key(fptr, TINT, "STT_IMJD", &imjd, "[days] Start MJD (UTC)", &status);
    fits_write_key(fptr, TDOUBLE, "STT_SMJD", &smjd, "[s] Start time (sec past UTC 00h)", &status);

    // REFACTORED: Use stack variable
    double stt_offs = 0.0;
    fits_write_key(fptr, TDOUBLE, "STT_OFFS", &stt_offs, "[s] Start time offset", &status);

    check_status("Writing PRIMARY HDU");
    return true;
}

bool PSRFITS_Writer::append_subint(double* data, double* mask)
{
	if (!fptr)
	{
		std::cerr << "FITS file not initialized." << std::endl;
		return false;
	}

	size_t nbits = 0;
	if (hdr.MODE == "SEARCH") nbits = 8;
	else if (hdr.MODE == "PSR") nbits = 16;

	size_t nchan = hdr.nchann;
	size_t npol = hdr.npol;
	size_t nbin = hdr.obs_window;
	long nsubint = hdr.nsubint;
	size_t nstot = hdr.OBS_SIZE;

	double tau = hdr.tau;
	double dm = hdr.dm;

	int nsblk = 4096;
	int c = hdr.cmplx ? 2 : 1;

	if (hdr.MODE == "PSR")
		nsblk = 1;

	if (hdr.MODE == "SEARCH")
		nbin = 1;

	char freq_form[32], wts_form[32], offs_form[32], scl_form[32], data_form[32];
	snprintf(freq_form, sizeof(freq_form), "%dD", int(nchan));
	snprintf(wts_form,  sizeof(wts_form),  "%dE", int(nchan));
	snprintf(offs_form, sizeof(offs_form), "%dE", int(nchan * npol));
	snprintf(scl_form,  sizeof(scl_form),  "%dE", int(nchan * npol));

	if (hdr.MODE == "PSR")
		snprintf(data_form, sizeof(data_form), "%dI", int(nbin * nchan * npol));
	else if (hdr.MODE == "SEARCH")
		snprintf(data_form, sizeof(data_form), "%dB", int((nsblk * nbits / 8) * nchan * npol * c));

	const char* ttype[] = { "TSUBINT", "OFFS_SUB", "DAT_FREQ", "DAT_WTS", "DAT_OFFS", "DAT_SCL", "PERIOD", "DATA" };
	const char* tform[] = { "1D", "1D", freq_form, wts_form, offs_form, scl_form, "1D", data_form };
	const char* tunit[] = { "s", "s", "MHz", "", "", "", "s", ""};

	fits_create_tbl(fptr, BINARY_TBL, 1, 8,
			const_cast<char**>(ttype),
			const_cast<char**>(tform),
			const_cast<char**>(tunit),
			"SUBINT", &status);

	if (hdr.cmplx)
	{
		int naxis = 4;
		long naxes[4] = {long(c), long(nchan), long(npol), long(nsblk * nbits / 8)};
		fits_write_tdim(fptr, 8, naxis, naxes, &status);
	}
	else
	{
		int naxis = 3;
		if (hdr.MODE == "SEARCH")
		{
			long naxes[3] = {long(nchan), long(npol), long(nsblk * nbits / 8)};
			fits_write_tdim(fptr, 8, naxis, naxes, &status);
		}
		else if (hdr.MODE == "PSR")
		{
			long naxes[3] = {long(nbin), long(nchan), long(npol)};
			fits_write_tdim(fptr, 8, naxis, naxes, &status);
		}

	}

	double fmin = hdr.freqs.at(0);
	double fmax = hdr.freqs.at(nchan-1);
	double dB = std::abs(fmax - fmin) / double(nchan);

	fits_write_key(fptr, TSTRING, "EPOCHS", (void*) "STT_MJD", "Epoch convention (VALID, MIDTIME, STT_MJD)", &status);
	fits_write_key(fptr, TSTRING, "INT_TYPE", (void*) "TIME", "Time axis (TIME, BINPHSPERI, BINLNGASC, etc)", &status);
	fits_write_key(fptr, TSTRING, "INT_UNIT", (void*) "SEC", "Unit of time axis (SEC, PHS (0-1), DEG)", &status);
	fits_write_key(fptr, TSTRING, "SCALE", (void*) "FluxDen", "Intensity units (FluxDen/RefFlux/Jansky)", &status);
	fits_write_key(fptr, TSTRING, "POL_TYPE", (void*) "NONE", "Polarisation identifier (e.g., AABBCRCI, AA+BB)", &status);



	// REFACTORED: Use stack variables
	int npol_val = static_cast<int>(npol);
	int nbin_val = static_cast<int>(nbin);
	int nchan_val = static_cast<int>(nchan);
    int nsblk_val = static_cast<int>(nsblk);
    long nstot_val = static_cast<long>(nstot);
    long nbits_val = static_cast<long>(nbits);
	double tbin_val = tau * 1.0e-3;
	double phs_offs = 0.0;
	double dm_val = dm;
	double zero_val = 0.0;
	int zero_int = 0;
	int one_int = 1;


	fits_write_key(fptr, TINT, "NPOL", &npol_val, "Number of polarisations", &status);
	fits_write_key(fptr, TDOUBLE, "TBIN", &tbin_val, "[s] Time per bin/sample", &status);
	fits_write_key(fptr, TINT, "NBIN", &nbin_val, "Nr of bins (PSR/CAL mode; else 1)", &status);
	fits_write_key(fptr, TDOUBLE, "PHS_OFFS", &phs_offs, "Phase offset of bin 0 for gated data", &status);
    fits_write_key(fptr, TINT, "NBITS", &nbits_val, "Nr of bits/datum (SEARCH mode data, else 1)", &status);
	fits_write_key(fptr, TINT, "SIGNINT", &one_int, "1 for signed ints in SEARCH-mode data, else 0", &status);
	fits_write_key(fptr, TINT, "NSUBOFFS", &zero_int, "Subint offset (Contiguous SEARCH-mode files)", &status);
	fits_write_key(fptr, TINT, "NCHAN", &nchan_val, "Number of channels/sub-bands in this file", &status);
	fits_write_key(fptr, TINT, "NCH_STRT", &zero_int, "Channel/sub-band offset for split files", &status);
	fits_write_key(fptr, TDOUBLE, "CHAN_BW", &dB, "[MHz] Channel/sub-band width", &status);
	fits_write_key(fptr, TDOUBLE, "DM", &dm_val, "[cm-3 pc] DM used for dedispersion", &status);
	fits_write_key(fptr, TDOUBLE, "RM", &zero_val, "[rad m-2] RM for post-detection deFaraday", &status);
	fits_write_key(fptr, TINT, "NCHNOFFS", &zero_int, "Channel/sub-band offset for split files", &status);
    fits_write_key(fptr, TINT, "NSBLK", &nsblk_val, "Samples/row (SEARCH mode, else 1)", &status);
    fits_write_key(fptr, TINT, "NSTOT", &nstot_val, "Total number of samples (SEARCH mode, else 1)", &status);
	fits_write_key(fptr, TINT, "EXTVER", &one_int, "auto assigned by template parser ", &status);

	if (hdr.MODE == "SEARCH" && hdr.cmplx)
		fits_write_key(fptr, TINT, "CMPLX", &one_int, "is data complex (1/0)", &status);



	check_status("Creating SUBINT bin table");

	std::vector<float> dat_wts(nchan);
	if (mask)
	{
		for (size_t i = 0; i < nchan; ++i)
			dat_wts[i] = static_cast<float>(mask[i]);

		for (size_t i = 0; i < nchan; ++i)
			dat_wts[i] = std::clamp(dat_wts[i], 0.0f, 1.0f);
	}
	else
	{
		std::fill(dat_wts.begin(), dat_wts.end(), 1.0f);
	}

	std::vector<float> dat_offs(nchan * npol*c);
	std::vector<float> dat_scl(nchan * npol*c);
	std::vector<char> data_int;

	if (hdr.MODE == "PSR")
		data_int.resize((nbits/8) * nbin * nchan * npol);
	else if (hdr.MODE == "SEARCH")
		data_int.resize(nsblk / (8 / nbits) * nchan * npol * c);


	// Iterate over subintegrations
	size_t actually_read = 0;
	size_t buf_pos = 0;
	double *current = nullptr;
	for (int row = 1; row < nsubint + 1; ++row)
	{
		if (hdr.MODE == "PSR")
			actually_read = nbin;
		else if (hdr.MODE == "SEARCH")
			actually_read = std::min(size_t(nsblk), nstot - buf_pos);

		current = data + buf_pos * npol * nchan * c ;
		quantize(
				current,
				actually_read,
				nchan,
				npol,
				data_int.data(),
				nbits,
				dat_scl.data(),
				dat_offs.data(),
				hdr.cmplx);

		buf_pos += actually_read;

		if (hdr.MODE == "PSR")
		{
			const std::vector<size_t> dims = {nbin, nchan};
			math::layout_c_to_f((int16_t*) data_int.data(), dims);
		}


		double tsubint = tau * actually_read * 1.0e-3;
		double offs_sub = (double(row) - 0.5) * tsubint;
		double period_val = hdr.period;

		if (hdr.t_subint.size() > 0)
			offs_sub = hdr.t_subint.at(row-1);



		fits_write_col(fptr, TDOUBLE, 1, row, 1, 1, &tsubint, &status);
		fits_write_col(fptr, TDOUBLE, 2, row, 1, 1, &offs_sub, &status);
		fits_write_col(fptr, TDOUBLE, 3, row, 1, nchan, hdr.freqs.data(), &status);
		fits_write_col(fptr, TFLOAT, 4, row, 1, nchan, dat_wts.data(), &status);
		fits_write_col(fptr, TFLOAT, 5, row, 1, nchan * npol * c, dat_offs.data(), &status);
		fits_write_col(fptr, TFLOAT, 6, row, 1, nchan * npol * c, dat_scl.data(), &status);
		fits_write_col(fptr, TDOUBLE, 7, row, 1, 1, &period_val, &status);

		if (hdr.MODE == "PSR")
			fits_write_col(fptr, TSHORT, 8, row, 1, nbin * nchan * npol, data_int.data(), &status);
		else if (hdr.MODE == "SEARCH")
			fits_write_col(fptr, TBYTE, 8, row, 1,
					(actually_read * nbits / 8) * nchan * npol * c,
					static_cast<void*>(data_int.data()), &status);
	}


	check_status("Writing SUBINT bintable");

	return true;
}





bool PSRFITS_Writer::append_subint(std::string stream_file, double* mask)
{
	if (!fptr)
	{
		std::cerr << "FITS file not initialized." << std::endl;
		return false;
	}

	size_t nbits = 0;
	if (hdr.MODE == "SEARCH") nbits = 8;
	else if (hdr.MODE == "PSR") nbits = 16;

	size_t nchan = hdr.nchann;
	size_t npol = hdr.npol;
	size_t nbin = hdr.obs_window;
	long nsubint = 0;
	size_t nstot = 0;

	double tau = hdr.tau;
	double dm = hdr.dm;

	int nsblk = 4096;
	int c = hdr.cmplx ? 2 : 1;

	if (hdr.MODE == "PSR")
		nsblk = 1;

	if (hdr.MODE == "SEARCH")
		nbin = 1;

    std::ifstream stream(stream_file, std::ios::binary);
    stream.seekg(0, std::ios::end);
    nstot = static_cast<long>(stream.tellg()) / sizeof(double);
    nstot = nstot / (nchan * npol * c);
    stream.seekg(0, std::ios::beg);

	if (hdr.MODE == "SEARCH")
	{
		nsubint = nstot / (nsblk * nbits / 8);
		if (nstot % (nsblk * nbits / 8) != 0)
			nsubint += 1;
	}
	else if (hdr.MODE == "PSR")
	{
		nsubint = nstot / nbin;
		if (nstot % nbin != 0)
			nsubint += 1;
	}

	char freq_form[32], wts_form[32], offs_form[32], scl_form[32], data_form[32];
	snprintf(freq_form, sizeof(freq_form), "%dD", int(nchan));
	snprintf(wts_form,  sizeof(wts_form),  "%dE", int(nchan));
	snprintf(offs_form, sizeof(offs_form), "%dE", int(nchan * npol));
	snprintf(scl_form,  sizeof(scl_form),  "%dE", int(nchan * npol));

	if (hdr.MODE == "PSR")
		snprintf(data_form, sizeof(data_form), "%dI", int(nbin * nchan * npol));
	else if (hdr.MODE == "SEARCH")
		snprintf(data_form, sizeof(data_form), "%dB", int((nsblk * nbits / 8) * nchan * npol * c));

	const char* ttype[] = { "TSUBINT", "OFFS_SUB", "DAT_FREQ", "DAT_WTS", "DAT_OFFS", "DAT_SCL", "PERIOD", "DATA" };
	const char* tform[] = { "1D", "1D", freq_form, wts_form, offs_form, scl_form, "1D", data_form };
	const char* tunit[] = { "s", "s", "MHz", "", "", "", "s", ""};

	fits_create_tbl(fptr, BINARY_TBL, 1, 8,
			const_cast<char**>(ttype),
			const_cast<char**>(tform),
			const_cast<char**>(tunit),
			"SUBINT", &status);

	if (hdr.cmplx)
	{
		int naxis = 4;
		long naxes[4] = {long(c), long(nchan), long(npol), long(nsblk * nbits / 8)};
		fits_write_tdim(fptr, 8, naxis, naxes, &status);
	}
	else
	{
		int naxis = 3;
		if (hdr.MODE == "SEARCH")
		{
			long naxes[3] = {long(nchan), long(npol), long(nsblk * nbits / 8)};
			fits_write_tdim(fptr, 8, naxis, naxes, &status);
		}
		else if (hdr.MODE == "PSR")
		{
			long naxes[3] = {long(nbin), long(nchan), long(npol)};
			fits_write_tdim(fptr, 8, naxis, naxes, &status);
		}

	}

	double fmin = hdr.freqs.at(0);
	double fmax = hdr.freqs.at(nchan-1);
	double dB = std::abs(fmax - fmin) / double(nchan);

	fits_write_key(fptr, TSTRING, "EPOCHS", (void*) "STT_MJD", "Epoch convention (VALID, MIDTIME, STT_MJD)", &status);
	fits_write_key(fptr, TSTRING, "INT_TYPE", (void*) "TIME", "Time axis (TIME, BINPHSPERI, BINLNGASC, etc)", &status);
	fits_write_key(fptr, TSTRING, "INT_UNIT", (void*) "SEC", "Unit of time axis (SEC, PHS (0-1), DEG)", &status);
	fits_write_key(fptr, TSTRING, "SCALE", (void*) "FluxDen", "Intensity units (FluxDen/RefFlux/Jansky)", &status);
	fits_write_key(fptr, TSTRING, "POL_TYPE", (void*) "NONE", "Polarisation identifier (e.g., AABBCRCI, AA+BB)", &status);



	// REFACTORED: Use stack variables
	int npol_val = static_cast<int>(npol);
	int nbin_val = static_cast<int>(nbin);
	int nchan_val = static_cast<int>(nchan);
    int nsblk_val = static_cast<int>(nsblk);
    long nstot_val = static_cast<long>(nstot);
    long nbits_val = static_cast<long>(nbits);
	double tbin_val = tau * 1.0e-3;
	double phs_offs = 0.0;
	double dm_val = dm;
	double zero_val = 0.0;
	int zero_int = 0;
	int one_int = 1;


	fits_write_key(fptr, TINT, "NPOL", &npol_val, "Number of polarisations", &status);
	fits_write_key(fptr, TDOUBLE, "TBIN", &tbin_val, "[s] Time per bin/sample", &status);
	fits_write_key(fptr, TINT, "NBIN", &nbin_val, "Nr of bins (PSR/CAL mode; else 1)", &status);
	fits_write_key(fptr, TDOUBLE, "PHS_OFFS", &phs_offs, "Phase offset of bin 0 for gated data", &status);
    fits_write_key(fptr, TINT, "NBITS", &nbits_val, "Nr of bits/datum (SEARCH mode data, else 1)", &status);
	fits_write_key(fptr, TINT, "SIGNINT", &one_int, "1 for signed ints in SEARCH-mode data, else 0", &status);
	fits_write_key(fptr, TINT, "NSUBOFFS", &zero_int, "Subint offset (Contiguous SEARCH-mode files)", &status);
	fits_write_key(fptr, TINT, "NCHAN", &nchan_val, "Number of channels/sub-bands in this file", &status);
	fits_write_key(fptr, TINT, "NCH_STRT", &zero_int, "Channel/sub-band offset for split files", &status);
	fits_write_key(fptr, TDOUBLE, "CHAN_BW", &dB, "[MHz] Channel/sub-band width", &status);
	fits_write_key(fptr, TDOUBLE, "DM", &dm_val, "[cm-3 pc] DM used for dedispersion", &status);
	fits_write_key(fptr, TDOUBLE, "RM", &zero_val, "[rad m-2] RM for post-detection deFaraday", &status);
	fits_write_key(fptr, TINT, "NCHNOFFS", &zero_int, "Channel/sub-band offset for split files", &status);
    fits_write_key(fptr, TINT, "NSBLK", &nsblk_val, "Samples/row (SEARCH mode, else 1)", &status);
    fits_write_key(fptr, TINT, "NSTOT", &nstot_val, "Total number of samples (SEARCH mode, else 1)", &status);
	fits_write_key(fptr, TINT, "EXTVER", &one_int, "auto assigned by template parser ", &status);

	if (hdr.MODE == "SEARCH" && hdr.cmplx)
		fits_write_key(fptr, TINT, "CMPLX", &one_int, "is data complex (1/0)", &status);



	check_status("Creating SUBINT bin table");

	std::vector<float> dat_wts(nchan);
	if (mask)
	{
		for (size_t i = 0; i < nchan; ++i)
			dat_wts[i] = static_cast<float>(mask[i]);

		for (size_t i = 0; i < nchan; ++i)
			dat_wts[i] = std::clamp(dat_wts[i], 0.0f, 1.0f);
	}
	else
	{
		std::fill(dat_wts.begin(), dat_wts.end(), 1.0f);
	}

	std::vector<float> dat_offs(nchan * npol*c);
	std::vector<float> dat_scl(nchan * npol*c);
	std::vector<char> data_int;
    std::vector<double> data_double;

	if (hdr.MODE == "PSR")
	{
		data_int.resize((nbits/8) * nbin * nchan * npol);
		data_double.resize(nbin * nchan * npol);
	}
	else if (hdr.MODE == "SEARCH")
	{
		data_int.resize(nsblk / (8 / nbits) * nchan * npol * c);
		data_double.resize(nsblk * nchan * npol * c);
	}


	// Iterate over subintegrations
	size_t actually_read = 0;
	size_t to_read = 0;
	for (int row = 1; row < nsubint + 1; ++row)
	{

		if (hdr.MODE == "PSR")
			to_read = nbin*nchan*npol;
		else if (hdr.MODE == "SEARCH")
			to_read = nsblk*nchan*npol*c;

		stream.read(reinterpret_cast<char*>(data_double.data()),
				sizeof(double) * to_read);

        actually_read = static_cast<size_t>(stream.gcount());
        actually_read /= (sizeof(double) * nchan * npol * c);


        if (actually_read < to_read)
        {
            std::fill(data_double.begin() + actually_read * nchan * npol * c,
                     data_double.end(), *(data_double.end()-1));
        }

		quantize(
				data_double.data(),
				actually_read,
				nchan,
				npol,
				data_int.data(),
				nbits,
				dat_scl.data(),
				dat_offs.data(),
				hdr.cmplx);

		if (hdr.MODE == "PSR")
		{
			const std::vector<size_t> dims = {actually_read, nchan};
			math::layout_c_to_f((int16_t*) data_int.data(), dims);
		}


		double tsubint = tau * actually_read * 1.0e-3;
		double offs_sub = (double(row) - 0.5) * tsubint;
		double period_val = hdr.period;

		if (hdr.t_subint.size() > 0)
			offs_sub = hdr.t_subint.at(row-1);



		fits_write_col(fptr, TDOUBLE, 1, row, 1, 1, &tsubint, &status);
		fits_write_col(fptr, TDOUBLE, 2, row, 1, 1, &offs_sub, &status);
		fits_write_col(fptr, TDOUBLE, 3, row, 1, nchan, hdr.freqs.data(), &status);
		fits_write_col(fptr, TFLOAT, 4, row, 1, nchan, dat_wts.data(), &status);
		fits_write_col(fptr, TFLOAT, 5, row, 1, nchan * npol * c, dat_offs.data(), &status);
		fits_write_col(fptr, TFLOAT, 6, row, 1, nchan * npol * c, dat_scl.data(), &status);
		fits_write_col(fptr, TDOUBLE, 7, row, 1, 1, &period_val, &status);

		if (hdr.MODE == "PSR")
			fits_write_col(fptr, TSHORT, 8, row, 1, actually_read * nchan * npol, data_int.data(), &status);
		else if (hdr.MODE == "SEARCH")
			fits_write_col(fptr, TBYTE, 8, row, 1,
					(actually_read * nbits / 8) * nchan * npol * c,
					static_cast<void*>(data_int.data()), &status);
	}


	check_status("Writing SUBINT bintable");

	stream.close();
	std::remove(stream_file.c_str());

	return true;
}

bool PSRFITS_Writer::append_history(std::string dds_mtd, double* mask)
{
    if (!fptr)
    {
        std::cerr << "FITS file not initialized." << std::endl;
        return false;
    }

	size_t nchan = hdr.nchann;
	size_t npol = hdr.npol;
	size_t nbin = hdr.obs_window;
	long nsubint = hdr.nsubint;

	double tau = hdr.tau;
	double dm = hdr.dm;

    const char* ttype[] = {
        "DATE_PRO", "PROC_CMD", "SCALE", "POL_TYPE", "NSUB", "NPOL", "NBIN",
        "TBIN", "CTR_FREQ", "NCHAN", "CHAN_BW", "DM", "RM", "RM_CORR",
        "DEDISP", "DDS_MTHD", "SC_MTHD", "CAL_MTHD", "CAL_FILE", "RFI_MTHD",
        "RM_MODEL", "AUX_RM_C", "DM_MODEL", "AUX_DM_C", "NBIN_PRD", "REF_FREQ",
        "PR_CORR", "FD_CORR", "BE_CORR"
    };

    const char* tform[] = {
        "24A", "256A", "8A", "8A", "1J", "1I", "1I", "1D", "1D", "1J", "1D",
        "1D", "1D", "1I", "1I", "32A", "32A", "32A", "256A", "32A", "32A",
        "1I", "32A", "1I", "1I", "1D", "1I", "1I", "1I"
    };

    const char *tunit[] = {"", "", "", "", "", "", "", "s", "MHz", "", "MHz",
                           "CM-3 PC", "RAD M-2", "", "", "", "", "", "", "",
                           "", "", "", "", "", "", "", "", ""};

    fits_create_tbl(fptr, BINARY_TBL, 1, 29,
                    const_cast<char**>(ttype),
                    const_cast<char**>(tform),
                    const_cast<char**>(tunit),
                    "HISTORY", &status);

    check_status("Creating HISTORY binary table");

    double ctr_freq = 0.0;
    double fmin = hdr.freqs.at(0);
    double fmax = hdr.freqs.at(nchan - 1);
    double bw = fmax - fmin;
    double chan_bw = bw / double(nchan);

    std::string rfi_mtd = "";

    if (mask)
    {
        for (size_t i = 0; i < nchan; ++i)
            ctr_freq = mask[i] * hdr.freqs[i];

        ctr_freq = ctr_freq / double(nchan);
        rfi_mtd = "spectral kurtisis";
    }
    else
    {
        ctr_freq = (fmin + fmax) / 2.0;
    }
    bw = std::abs(bw);
    chan_bw = std::abs(chan_bw);

    int dedisp = 0;
    if (dds_mtd == "none" || dds_mtd == "")
        dedisp = 0;
    else if (dds_mtd == "coherent" || dds_mtd == "incoherent")
        dedisp = 1;
    else
        std::cout << "Unknown type of dedispersion. Setting flag to dedispersed" << std::endl;

    // REFACTORED: Use stack variables instead of new
    long nsubint_val = static_cast<long>(nsubint);
    long nchan_val = static_cast<long>(nchan);
    double tau_val = tau * 1e-3;
    double dm_val = dm;
    double zero_val = 0.0;
    int zero_int = 0;
    int one_int = 1;
    double fcomp_val = hdr.fcomp;

    fits_write_col(fptr, TBYTE,     1, 1, 1, 24, convert_str(getCurrentUTCTime(), 24), &status);
    fits_write_col(fptr, TBYTE,     2, 1, 1, 256, convert_str("This file was created with AnTi-pipeline2", 256), &status);
    fits_write_col(fptr, TBYTE,     3, 1, 1, 8, convert_str("FluxDen", 8), &status);
    fits_write_col(fptr, TBYTE,     4, 1, 1, 8, convert_str("NONE", 8), &status);
    fits_write_col(fptr, TLONG,     5, 1, 1, 1, &nsubint_val, &status);
    fits_write_col(fptr, TINT,      6, 1, 1, 1, &npol, &status);
    fits_write_col(fptr, TINT,      7, 1, 1, 1, &nbin, &status);
    fits_write_col(fptr, TDOUBLE,   8, 1, 1, 1, &tau_val, &status);
    fits_write_col(fptr, TDOUBLE,   9, 1, 1, 1, &ctr_freq, &status);
    fits_write_col(fptr, TLONG,    10, 1, 1, 1, &nchan_val, &status);
    fits_write_col(fptr, TDOUBLE,  11, 1, 1, 1, &chan_bw, &status);
    fits_write_col(fptr, TDOUBLE,  12, 1, 1, 1, &dm_val, &status);
    fits_write_col(fptr, TDOUBLE,  13, 1, 1, 1, &zero_val, &status);
    fits_write_col(fptr, TINT,     14, 1, 1, 1, &zero_int, &status);
    fits_write_col(fptr, TINT,     15, 1, 1, 1, &dedisp, &status);
    fits_write_col(fptr, TBYTE,    16, 1, 1, 32, convert_str(dds_mtd, 32), &status);
    fits_write_col(fptr, TBYTE,    17, 1, 1, 32, convert_str("NONE", 32), &status);
    fits_write_col(fptr, TBYTE,    18, 1, 1, 32, convert_str("NONE", 32), &status);
    fits_write_col(fptr, TBYTE,    19, 1, 1, 256, convert_str("NONE", 256), &status);
    fits_write_col(fptr, TBYTE,    20, 1, 1, 32, convert_str(rfi_mtd, 32), &status);
    fits_write_col(fptr, TBYTE,    21, 1, 1, 32, convert_str("NONE", 32), &status);
    fits_write_col(fptr, TINT,     22, 1, 1, 1, &zero_int, &status);
    fits_write_col(fptr, TBYTE,    23, 1, 1, 32, convert_str("NONE", 32), &status);
    fits_write_col(fptr, TINT,     24, 1, 1, 1, &zero_int, &status);
    fits_write_col(fptr, TINT,     25, 1, 1, 1, &one_int, &status);
    fits_write_col(fptr, TDOUBLE,  26, 1, 1, 1, &fcomp_val, &status);
    fits_write_col(fptr, TINT,     27, 1, 1, 1, &zero_int, &status);
    fits_write_col(fptr, TINT,     28, 1, 1, 1, &zero_int, &status);
    fits_write_col(fptr, TINT,     29, 1, 1, 1, &zero_int, &status);

    fits_write_key(fptr, TINT, "EXTVER", &one_int, "auto assigned by template parser ", &status);

    check_status("Writing history");
    return true;


}

bool PSRFITS_Writer::append_bandpass(double* fr)
{
	int npol = hdr.npol;
	int nchan = hdr.nchann;

	char offs_form[32], scl_form[32], data_form[32];
	snprintf(offs_form, sizeof(offs_form), "%dE", npol);
	snprintf(scl_form, sizeof(scl_form), "%dE", npol);
	snprintf(data_form, sizeof(scl_form), "%dI", npol*nchan);

    const char* ttype[] = {"DAT_OFFS", "DAT_SCL", "DATA"};
    const char* tform[] = {offs_form, scl_form, data_form};
    const char* tunit[] = {"", "", ""};


	fits_write_key(fptr, TINT, "NCH_ORIG", &nchan, "Number of channels in original bandpass", &status);
	fits_write_key(fptr, TINT, "BP_NPOL", &npol, "Number of polarizations in bandpass", &status);

    fits_create_tbl(fptr, BINARY_TBL, 1, 3,
                    const_cast<char**>(ttype),
                    const_cast<char**>(tform),
                    const_cast<char**>(tunit),
                    "BANDPASS", &status);

	check_status("Creating BANDPASS table");

    int naxis = 2;
    long naxes[2] = {long(nchan), long(npol)};
    fits_write_tdim(fptr, 3, naxis, naxes, &status);

	std::vector<float> dat_offs, dat_scl;
	std::vector<int16_t> data_int;

	dat_offs.resize(npol);
	dat_scl.resize(npol);
	data_int.resize(npol*nchan);

	quantize(
			fr,
			nchan*npol,
			npol,
			1,
			(char*) data_int.data(),
			16,
			dat_scl.data(),
			dat_offs.data());

	fits_write_col(fptr, TFLOAT, 1, 1, 1, npol, dat_offs.data(), &status);
	fits_write_col(fptr, TFLOAT, 2, 1, 1, npol, dat_scl.data(), &status);
	fits_write_col(fptr, TSHORT,   3, 1, 1, npol*nchan, data_int.data(), &status);

	check_status("Writing BANDPASS table");


	return true;
}

bool PSRFITS_Writer::append_history(
        const size_t nsubint,
        const size_t nbin,
        const size_t npol,
        const size_t nchan,
        const double dm,
        const double *freqs,
        const double fcomp,
        const double tau,
        std::string dds_mtd,
        const double* mask)
{
    if (!fptr)
    {
        std::cerr << "FITS file not initialized." << std::endl;
        return false;
    }

    const char* ttype[] = {
        "DATE_PRO", "PROC_CMD", "SCALE", "POL_TYPE", "NSUB", "NPOL", "NBIN",
        "TBIN", "CTR_FREQ", "NCHAN", "CHAN_BW", "DM", "RM", "RM_CORR",
        "DEDISP", "DDS_MTHD", "SC_MTHD", "CAL_MTHD", "CAL_FILE", "RFI_MTHD",
        "RM_MODEL", "AUX_RM_C", "DM_MODEL", "AUX_DM_C", "NBIN_PRD", "REF_FREQ",
        "PR_CORR", "FD_CORR", "BE_CORR"
    };

    const char* tform[] = {
        "24A", "256A", "8A", "8A", "1J", "1I", "1I", "1D", "1D", "1J", "1D",
        "1D", "1D", "1I", "1I", "32A", "32A", "32A", "256A", "32A", "32A",
        "1I", "32A", "1I", "1I", "1D", "1I", "1I", "1I"
    };

    const char *tunit[] = {"", "", "", "", "", "", "", "s", "MHz", "", "MHz",
                           "CM-3 PC", "RAD M-2", "", "", "", "", "", "", "",
                           "", "", "", "", "", "", "", "", ""};

    fits_create_tbl(fptr, BINARY_TBL, 1, 29,
                    const_cast<char**>(ttype),
                    const_cast<char**>(tform),
                    const_cast<char**>(tunit),
                    "HISTORY", &status);

    check_status("Creating HISTORY binary table");

    double ctr_freq = 0.0;
    double fmin = freqs[0];
    double fmax = freqs[nchan - 1];
    double bw = fmax - fmin;
    double chan_bw = bw / double(nchan);

    std::string rfi_mtd = "";

    if (mask)
    {
        for (size_t i = 0; i < nchan; ++i)
            ctr_freq = mask[i] * freqs[i];

        ctr_freq = ctr_freq / double(nchan);
        rfi_mtd = "spectral kurtisis";
    }
    else
    {
        ctr_freq = (fmin + fmax) / 2.0;
    }
    bw = std::abs(bw);
    chan_bw = std::abs(chan_bw);

    int dedisp = 0;
    if (dds_mtd == "none" || dds_mtd == "")
        dedisp = 0;
    else if (dds_mtd == "coherent" || dds_mtd == "incoherent")
        dedisp = 1;
    else
        std::cout << "Unknown type of dedispersion. Setting flag to dedispersed" << std::endl;

    // REFACTORED: Use stack variables instead of new
    long nsubint_val = static_cast<long>(nsubint);
    long nchan_val = static_cast<long>(nchan);
    double tau_val = tau * 1e-3;
    double dm_val = dm;
    double zero_val = 0.0;
    int zero_int = 0;
    int one_int = 1;
    double fcomp_val = fcomp;

    fits_write_col(fptr, TBYTE,     1, 1, 1, 24, convert_str(getCurrentUTCTime(), 24), &status);
    fits_write_col(fptr, TBYTE,     2, 1, 1, 256, convert_str("This file was created with AnTi-pipeline2", 256), &status);
    fits_write_col(fptr, TBYTE,     3, 1, 1, 8, convert_str("FluxDen", 8), &status);
    fits_write_col(fptr, TBYTE,     4, 1, 1, 8, convert_str("NONE", 8), &status);
    fits_write_col(fptr, TLONG,     5, 1, 1, 1, &nsubint_val, &status);
    fits_write_col(fptr, TINT,      6, 1, 1, 1, (void*)&npol, &status);
    fits_write_col(fptr, TINT,      7, 1, 1, 1, (void*)&nbin, &status);
    fits_write_col(fptr, TDOUBLE,   8, 1, 1, 1, &tau_val, &status);
    fits_write_col(fptr, TDOUBLE,   9, 1, 1, 1, (void*)&ctr_freq, &status);
    fits_write_col(fptr, TLONG,    10, 1, 1, 1, (void*)&nchan_val, &status);
    fits_write_col(fptr, TDOUBLE,  11, 1, 1, 1, (void*)&chan_bw, &status);
    fits_write_col(fptr, TDOUBLE,  12, 1, 1, 1, (void*)&dm_val, &status);
    fits_write_col(fptr, TDOUBLE,  13, 1, 1, 1, &zero_val, &status);
    fits_write_col(fptr, TINT,     14, 1, 1, 1, &zero_int, &status);
    fits_write_col(fptr, TINT,     15, 1, 1, 1, (void*)&dedisp, &status);
    fits_write_col(fptr, TBYTE,    16, 1, 1, 32, convert_str(dds_mtd, 32), &status);
    fits_write_col(fptr, TBYTE,    17, 1, 1, 32, convert_str("NONE", 32), &status);
    fits_write_col(fptr, TBYTE,    18, 1, 1, 32, convert_str("NONE", 32), &status);
    fits_write_col(fptr, TBYTE,    19, 1, 1, 256, convert_str("NONE", 256), &status);
    fits_write_col(fptr, TBYTE,    20, 1, 1, 32, convert_str(rfi_mtd, 32), &status);
    fits_write_col(fptr, TBYTE,    21, 1, 1, 32, convert_str("NONE", 32), &status);
    fits_write_col(fptr, TINT,     22, 1, 1, 1, &zero_int, &status);
    fits_write_col(fptr, TBYTE,    23, 1, 1, 32, convert_str("NONE", 32), &status);
    fits_write_col(fptr, TINT,     24, 1, 1, 1, &zero_int, &status);
    fits_write_col(fptr, TINT,     25, 1, 1, 1, &one_int, &status);
    fits_write_col(fptr, TDOUBLE,  26, 1, 1, 1, (void*)&fcomp_val, &status);
    fits_write_col(fptr, TINT,     27, 1, 1, 1, &zero_int, &status);
    fits_write_col(fptr, TINT,     28, 1, 1, 1, &zero_int, &status);
    fits_write_col(fptr, TINT,     29, 1, 1, 1, &zero_int, &status);

    fits_write_key(fptr, TINT, "EXTVER", &one_int, "auto assigned by template parser ", &status);

    check_status("Writing history");
    return true;
}

bool PSRFITS_Writer::append_subint_fold(
        double *data_double,
        double *dat_freq,
        double *mask,
        const size_t nbin,
        const size_t nchan,
        const size_t npol,
        const double period,
        const double dm,
        const double fcomp,
        const double tau,
        std::string dds_mtd)
{
    if (!fptr)
    {
        std::cerr << "FITS file not initialized." << std::endl;
        return false;
    }

    char freq_form[32], wts_form[32], offs_form[32], scl_form[32], data_form[32];
    snprintf(freq_form, sizeof(freq_form), "%dD", int(nchan));
    snprintf(wts_form,  sizeof(wts_form),  "%dE", int(nchan));
    snprintf(offs_form, sizeof(offs_form), "%dE", int(nchan * npol));
    snprintf(scl_form,  sizeof(scl_form),  "%dE", int(nchan * npol));
    snprintf(data_form, sizeof(data_form), "%dI", int(nbin * nchan * npol));

    const char* ttype[] = { "TSUBINT", "OFFS_SUB", "DAT_FREQ", "DAT_WTS", "DAT_OFFS", "DAT_SCL", "PERIOD", "DATA" };
    const char* tform[] = { "1D", "1D", freq_form, wts_form, offs_form, scl_form, "1D", data_form };
    const char* tunit[] = { "s", "s", "MHz", "", "", "", "s", ""};

    fits_create_tbl(fptr, BINARY_TBL, 1, 8,
                    const_cast<char**>(ttype),
                    const_cast<char**>(tform),
                    const_cast<char**>(tunit),
                    "SUBINT", &status);

    int naxis = 3;
    long naxes[3] = {long(nbin), long(nchan), long(npol)};
    fits_write_tdim(fptr, 8, naxis, naxes, &status);

    double fmin = dat_freq[0];
    double fmax = dat_freq[nchan-1];
    double dB = std::abs(fmax - fmin) / double(nchan);

    fits_write_key(fptr, TSTRING, "EPOCHS", (void*) "STT_MJD", "Epoch convention (VALID, MIDTIME, STT_MJD)", &status);
    fits_write_key(fptr, TSTRING, "INT_TYPE", (void*) "TIME", "Time axis (TIME, BINPHSPERI, BINLNGASC, etc)", &status);
    fits_write_key(fptr, TSTRING, "INT_UNIT", (void*) "SEC", "Unit of time axis (SEC, PHS (0-1), DEG)", &status);
    fits_write_key(fptr, TSTRING, "SCALE", (void*) "FluxDen", "Intensity units (FluxDen/RefFlux/Jansky)", &status);
    fits_write_key(fptr, TSTRING, "POL_TYPE", (void*) "NONE", "Polarisation identifier (e.g., AABBCRCI, AA+BB)", &status);



    // REFACTORED: Use stack variables
    int npol_val = static_cast<int>(npol);
    int nbin_val = static_cast<int>(nbin);
    int nchan_val = static_cast<int>(nchan);
    double tbin_val = tau * 1.0e-3;
    double phs_offs = 0.0;
    double dm_val = dm;
    double zero_val = 0.0;
    int zero_int = 0;
    int one_int = 1;


    fits_write_key(fptr, TINT, "NPOL", &npol_val, "Number of polarisations", &status);
    fits_write_key(fptr, TDOUBLE, "TBIN", &tbin_val, "[s] Time per bin/sample", &status);
    fits_write_key(fptr, TINT, "NBIN", &nbin_val, "Nr of bins (PSR/CAL mode; else 1)", &status);
    fits_write_key(fptr, TDOUBLE, "PHS_OFFS", &phs_offs, "Phase offset of bin 0 for gated data", &status);
    fits_write_key(fptr, TINT, "SIGNINT", &one_int, "1 for signed ints in SEARCH-mode data, else 0", &status);
    fits_write_key(fptr, TINT, "NSUBOFFS", &zero_int, "Subint offset (Contiguous SEARCH-mode files)", &status);
    fits_write_key(fptr, TINT, "NCHAN", (void*)&nchan_val, "Number of channels/sub-bands in this file", &status);
    fits_write_key(fptr, TINT, "NCH_STRT", &zero_int, "Channel/sub-band offset for split files", &status);
    fits_write_key(fptr, TDOUBLE, "CHAN_BW", &dB, "[MHz] Channel/sub-band width", &status);
    fits_write_key(fptr, TDOUBLE, "DM", (void*)&dm_val, "[cm-3 pc] DM used for dedispersion", &status);
    fits_write_key(fptr, TDOUBLE, "RM", &zero_val, "[rad m-2] RM for post-detection deFaraday", &status);
    fits_write_key(fptr, TINT, "NCHNOFFS", &zero_int, "Channel/sub-band offset for split files", &status);
    fits_write_key(fptr, TINT, "EXTVER", &one_int, "auto assigned by template parser ", &status);

    check_status("Creating SUBINT (dynamic profile) bin table");

    std::vector<float> dat_offs(nchan * npol);
    std::vector<float> dat_scl(nchan * npol);
    std::vector<char> data_int(2 * nbin * nchan * npol);

    quantize(
            data_double,
            nbin,
            nchan,
            npol,
            data_int.data(),
            16,
            dat_scl.data(),
            dat_offs.data());

    const std::vector<size_t> dims = {nbin, nchan};
    math::layout_c_to_f((int16_t*) data_int.data(), dims);

    std::vector<float> dat_wts(nchan);
    if (mask)
    {
        for (size_t i = 0; i < nchan; ++i)
            dat_wts[i] = static_cast<float>(mask[i]);

        for (size_t i = 0; i < nchan; ++i)
            dat_wts[i] = std::clamp(dat_wts[i], 0.0f, 1.0f);
    }
    else
    {
        std::fill(dat_wts.begin(), dat_wts.end(), 1.0f);
    }

    size_t subint_index = 0;
    long row = static_cast<long>(subint_index + 1);

    double tsubint = tau * nbin * 1.0e-3;
    double offs_sub = (subint_index + 0.5) * tsubint;
    double period_val = period;

    fits_write_col(fptr, TDOUBLE, 1, row, 1, 1, &tsubint, &status);
    fits_write_col(fptr, TDOUBLE, 2, row, 1, 1, &offs_sub, &status);
    fits_write_col(fptr, TDOUBLE, 3, row, 1, nchan, dat_freq, &status);
    fits_write_col(fptr, TFLOAT, 4, row, 1, nchan, dat_wts.data(), &status);
    fits_write_col(fptr, TFLOAT, 5, row, 1, nchan * npol, dat_offs.data(), &status);
    fits_write_col(fptr, TFLOAT, 6, row, 1, nchan * npol, dat_scl.data(), &status);
    fits_write_col(fptr, TDOUBLE, 7, row, 1, 1, &period_val, &status);
    fits_write_col(fptr, TSHORT, 8, row, 1, nbin * nchan * npol, data_int.data(), &status);


    check_status("Writing SUBINT bintable (folded pulse)");

    append_history(1, nbin, npol, nchan, dm, dat_freq, fcomp, tau, dds_mtd);

    return true;
}

bool PSRFITS_Writer::append_subint_stream(
        std::string stream_file,
        double *dat_freq,
        double *mask,
        const size_t nchan,
        const size_t npol,
        const double dm,
        const double fcomp,
        const double tau,
        const std::string dds_mtd,
        const bool cmp)
{
    if (!fptr)
    {
        std::cerr << "FITS file not initialized." << std::endl;
        return false;
    }

    double fmin = dat_freq[0];
    double fmax = dat_freq[nchan-1];

    int nsblk = 4096;
    int nbits = 8;
    int c = cmp ? 2 : 1;

    std::ifstream stream(stream_file, std::ios::binary);
    stream.seekg(0, std::ios::end);
    long nstot = static_cast<long>(stream.tellg()) / sizeof(double);
    nstot = nstot / (nchan * npol * c);
    stream.seekg(0, std::ios::beg);

    long nsubint = nstot / (nsblk * nbits / 8);
    if (nstot % (nsblk * nbits / 8) != 0)
        nsubint += 1;

    char freq_form[32], wts_form[32], offs_form[32], scl_form[32], data_form[32];
    snprintf(freq_form, sizeof(freq_form), "%dD", int(nchan));
    snprintf(wts_form,  sizeof(wts_form),  "%dE", int(nchan));
    snprintf(offs_form, sizeof(offs_form), "%dE", int(nchan * npol * c));
    snprintf(scl_form,  sizeof(scl_form),  "%dE", int(nchan * npol * c));
    snprintf(data_form, sizeof(data_form), "%dB", int((nsblk * nbits / 8) * nchan * npol * c));

    const char* ttype[] = { "TSUBINT", "OFFS_SUB", "DAT_FREQ", "DAT_WTS", "DAT_OFFS", "DAT_SCL", "DATA" };
    const char* tform[] = { "1D", "1D", freq_form, wts_form, offs_form, scl_form, data_form };
    const char* tunit[] = { "s", "s", "MHz", "", "", "", ""};

    fits_create_tbl(fptr, BINARY_TBL, nsubint, 7,
                    const_cast<char**>(ttype),
                    const_cast<char**>(tform),
                    const_cast<char**>(tunit),
                    "SUBINT", &status);

    if (cmp)
    {
        int naxis = 4;
        long naxes[4] = {long(c), long(nchan), long(npol), long(nsblk * nbits / 8)};
        fits_write_tdim(fptr, 7, naxis, naxes, &status);
    }
    else
    {
        int naxis = 3;
        long naxes[3] = {long(nchan), long(npol), long(nsblk * nbits / 8)};
        fits_write_tdim(fptr, 7, naxis, naxes, &status);
    }

    // REFACTORED: Use stack variables
    int nsubint_val = static_cast<int>(nsubint);
    int npol_val = static_cast<int>(npol);
    int nchan_val = static_cast<int>(nchan);
    int nsblk_val = nsblk;
    long nstot_val = nstot;
    int nbits_val = nbits;
    int is_cmplx = cmp ? 1 : 0;
    int zero_int = 0;
    int one_int = 1;
    double tbin_val = tau * 1.0e-3;
    double zero_offset = 0.0;
    double phs_offs = 0.0;
    double dB = std::abs(fmax - fmin) / double(nchan);
    double dm_val = dm;
    double zero_double = 0.0;

    fits_write_key(fptr, TINT, "NSUBINT", &nsubint_val, "Number of Sub-Integrations", &status);
    fits_write_key(fptr, TSTRING, "EPOCHS", (void*) "STT_MJD", "Epoch convention (VALID, MIDTIME, STT_MJD)", &status);
    fits_write_key(fptr, TSTRING, "INT_TYPE", (void*) "TIME", "Time axis (TIME, BINPHSPERI, BINLNGASC, etc)", &status);
    fits_write_key(fptr, TSTRING, "INT_UNIT", (void*) "SEC", "Unit of time axis (SEC, PHS (0-1), DEG)", &status);
    fits_write_key(fptr, TSTRING, "SCALE", (void*) "FluxDen", "Intensity units (FluxDen/RefFlux/Jansky)", &status);
    fits_write_key(fptr, TSTRING, "POL_TYPE", (void*) "NONE", "Polarisation identifier (e.g., AABBCRCI, AA+BB)", &status);
    fits_write_key(fptr, TINT, "NPOL", (void*)&npol_val, "Number of polarisations", &status);
    fits_write_key(fptr, TDOUBLE, "TBIN", &tbin_val, "[s] Time per bin/sample", &status);
    fits_write_key(fptr, TINT, "NBIN", &one_int, "Nr of bins (PSR/CAL mode; else 1)", &status);
    fits_write_key(fptr, TDOUBLE, "PHS_OFFS", &phs_offs, "Phase offset of bin 0 for gated data", &status);
    fits_write_key(fptr, TINT, "NBITS", &nbits_val, "Nr of bits/datum (SEARCH mode data, else 1)", &status);
    fits_write_key(fptr, TDOUBLE, "ZERO_OFF", &zero_offset, "Zero offset for SEARCH-mode data", &status);
    fits_write_key(fptr, TINT, "SIGNINT", &one_int, "1 for signed ints in SEARCH-mode data, else 0", &status);
    fits_write_key(fptr, TINT, "NSUBOFFS", &zero_int, "Subint offset (Contiguous SEARCH-mode files)", &status);
    fits_write_key(fptr, TINT, "NCHAN", (void*)&nchan_val, "Number of channels/sub-bands in this file", &status);
    fits_write_key(fptr, TINT, "NCH_STRT", &zero_int, "Channel/sub-band offset for split files", &status);
    fits_write_key(fptr, TDOUBLE, "CHAN_BW", &dB, "[MHz] Channel/sub-band width", &status);
    fits_write_key(fptr, TDOUBLE, "DM", (void*)&dm_val, "[cm-3 pc] DM used for dedispersion", &status);
    fits_write_key(fptr, TDOUBLE, "RM", &zero_double, "[rad m-2] RM for post-detection deFaraday", &status);
    fits_write_key(fptr, TINT, "NCHNOFFS", &zero_int, "Channel/sub-band offset for split files", &status);
    fits_write_key(fptr, TINT, "NSBLK", &nsblk_val, "Samples/row (SEARCH mode, else 1)", &status);
    fits_write_key(fptr, TINT, "NSTOT", &nstot_val, "Total number of samples (SEARCH mode, else 1)", &status);
    fits_write_key(fptr, TINT, "EXTVER", &one_int, "auto assigned by template parser ", &status);
    fits_write_key(fptr, TINT, "CMPLX", &is_cmplx, "is data complex (1/0)", &status);

    std::vector<float> dat_wts(nchan);
    if (mask)
    {
        for (size_t i = 0; i < nchan; ++i)
            dat_wts[i] = static_cast<float>(mask[i]);

        for (size_t i = 0; i < nchan; ++i)
            dat_wts[i] = std::clamp(dat_wts[i], 0.0f, 1.0f);
    }
    else
    {
        std::fill(dat_wts.begin(), dat_wts.end(), 1.0f);
    }

    std::vector<float> dat_offs(nchan * npol * c);
    std::vector<float> dat_scl(nchan * npol * c);
    std::vector<char> data_int((nsblk * nbits / 8) * nchan * npol * c);

    // REFACTORED: Use std::vector instead of raw new/delete
    std::vector<double> data_double(nsblk * nchan * npol * c);
    size_t actually_read = 0;

    for (int row = 1; row <= nsubint; ++row)
    {
        stream.read(reinterpret_cast<char*>(data_double.data()),
                    sizeof(double) * nsblk * nchan * npol * c);

        actually_read = static_cast<size_t>(stream.gcount());
        actually_read /= (sizeof(double) * nchan * npol * c);

        if (actually_read < size_t(nsblk))
        {
            std::fill(data_double.begin() + actually_read * nchan * npol * c,
                     data_double.end(), 0.0);
        }

        quantize(
                data_double.data(),
                actually_read,
                nchan,
                npol,
                data_int.data(),
                8,
                dat_scl.data(),
                dat_offs.data(),
                cmp);

        double tsubint = tau * actually_read * 1.0e-3;
        double offs_sub = (double(row) - 0.5) * tsubint;

        fits_write_col(fptr, TDOUBLE, 1, row, 1, 1, &tsubint, &status);
        fits_write_col(fptr, TDOUBLE, 2, row, 1, 1, &offs_sub, &status);
        fits_write_col(fptr, TDOUBLE, 3, row, 1, nchan, dat_freq, &status);
        fits_write_col(fptr, TFLOAT, 4, row, 1, nchan, dat_wts.data(), &status);
        fits_write_col(fptr, TFLOAT, 5, row, 1, nchan * npol * c, dat_offs.data(), &status);
        fits_write_col(fptr, TFLOAT, 6, row, 1, nchan * npol * c, dat_scl.data(), &status);
        fits_write_col(fptr, TBYTE, 7, row, 1,
                      (actually_read * nbits / 8) * nchan * npol * c,
                      static_cast<void*>(data_int.data()), &status);
    }

    stream.close();
    std::remove(stream_file.c_str());

    append_history(
            static_cast<size_t>(nsubint), 1, npol, nchan,
            dm, dat_freq, fcomp,
            tau, dds_mtd);

    return true;
}

bool PSRFITS_Writer::append_subint_search(
        double* data_double,
        double* dat_freq,
        double *mask,
        const size_t nbin,
        const size_t nchan,
        const size_t npol,
        const double dm,
        const double fcomp,
        const double tau,
        const std::string dds_mtd,
        const bool cmp)
{
    if (!fptr)
    {
        std::cerr << "FITS file not initialized." << std::endl;
        return false;
    }

    double fmin = dat_freq[0];
    double fmax = dat_freq[nchan-1];

    int nsblk = 4096;
    int nbits = 8;
    int c = cmp ? 2 : 1;

    size_t nstot = nbin;
    long nsubint = nstot / nsblk;
    if (nstot % nsblk != 0)
        nsubint += 1;

    char freq_form[32], wts_form[32], offs_form[32], scl_form[32], data_form[32];
    snprintf(freq_form, sizeof(freq_form), "%dD", int(nchan));
    snprintf(wts_form,  sizeof(wts_form),  "%dE", int(nchan));
    snprintf(offs_form, sizeof(offs_form), "%dE", int(nchan * npol * c));
    snprintf(scl_form,  sizeof(scl_form),  "%dE", int(nchan * npol * c));
    snprintf(data_form, sizeof(data_form), "%dB", int((nsblk * nbits / 8) * nchan * npol * c));

    const char* ttype[] = { "TSUBINT", "OFFS_SUB", "DAT_FREQ", "DAT_WTS", "DAT_OFFS", "DAT_SCL", "DATA" };
    const char* tform[] = { "1D", "1D", freq_form, wts_form, offs_form, scl_form, data_form };
    const char* tunit[] = { "s", "s", "MHz", "", "", "", ""};

    fits_create_tbl(fptr, BINARY_TBL, nsubint, 7,
                    const_cast<char**>(ttype),
                    const_cast<char**>(tform),
                    const_cast<char**>(tunit),
                    "SUBINT", &status);

    if (cmp)
    {
        int naxis = 4;
        long naxes[4] = {long(c), long(nchan), long(npol), long(nsblk * nbits / 8)};
        fits_write_tdim(fptr, 7, naxis, naxes, &status);
    }
    else
    {
        int naxis = 3;
        long naxes[3] = {long(nchan), long(npol), long(nsblk * nbits / 8)};
        fits_write_tdim(fptr, 7, naxis, naxes, &status);
    }

    // REFACTORED: Use stack variables
    int npol_val = static_cast<int>(npol);
    int nchan_val = static_cast<int>(nchan);
    int nsblk_val = nsblk;
    long nstot_val = static_cast<long>(nstot);
    int nbits_val = nbits;
    int is_cmplx = cmp ? 1 : 0;
    int zero_int = 0;
    int one_int = 1;
    double tbin_val = tau * 1.0e-3;
    double zero_offset = 0.0;
    double phs_offs = 0.0;
    double dB = std::abs(fmax - fmin) / double(nchan);
    double dm_val = dm;
    double zero_double = 0.0;

    fits_write_key(fptr, TSTRING, "EPOCHS", (void*) "STT_MJD", "Epoch convention (VALID, MIDTIME, STT_MJD)", &status);
    fits_write_key(fptr, TSTRING, "INT_TYPE", (void*) "TIME", "Time axis (TIME, BINPHSPERI, BINLNGASC, etc)", &status);
    fits_write_key(fptr, TSTRING, "INT_UNIT", (void*) "SEC", "Unit of time axis (SEC, PHS (0-1), DEG)", &status);
    fits_write_key(fptr, TSTRING, "SCALE", (void*) "FluxDen", "Intensity units (FluxDen/RefFlux/Jansky)", &status);
    fits_write_key(fptr, TSTRING, "POL_TYPE", (void*) "NONE", "Polarisation identifier (e.g., AABBCRCI, AA+BB)", &status);
    fits_write_key(fptr, TINT, "NPOL", &npol_val, "Number of polarisations", &status);
    fits_write_key(fptr, TDOUBLE, "TBIN", &tbin_val, "[s] Time per bin/sample", &status);
    fits_write_key(fptr, TINT, "NBIN", &one_int, "Nr of bins (PSR/CAL mode; else 1)", &status);
    fits_write_key(fptr, TDOUBLE, "PHS_OFFS", &phs_offs, "Phase offset of bin 0 for gated data", &status);
    fits_write_key(fptr, TINT, "NBITS", &nbits_val, "Nr of bits/datum (SEARCH mode data, else 1)", &status);
    fits_write_key(fptr, TDOUBLE, "ZERO_OFF", &zero_offset, "Zero offset for SEARCH-mode data", &status);
    fits_write_key(fptr, TINT, "SIGNINT", &one_int, "1 for signed ints in SEARCH-mode data, else 0", &status);
    fits_write_key(fptr, TINT, "NSUBOFFS", &zero_int, "Subint offset (Contiguous SEARCH-mode files)", &status);
    fits_write_key(fptr, TINT, "NCHAN", (void*)&nchan_val, "Number of channels/sub-bands in this file", &status);
    fits_write_key(fptr, TDOUBLE, "CHAN_BW", &dB, "[MHz] Channel/sub-band width", &status);
    fits_write_key(fptr, TDOUBLE, "DM", (void*)&dm_val, "[cm-3 pc] DM used for dedispersion", &status);
    fits_write_key(fptr, TDOUBLE, "RM", &zero_double, "[rad m-2] RM for post-detection deFaraday", &status);
    fits_write_key(fptr, TINT, "NCHNOFFS", &zero_int, "Channel/sub-band offset for split files", &status);
    fits_write_key(fptr, TINT, "NSBLK", &nsblk_val, "Samples/row (SEARCH mode, else 1)", &status);
    fits_write_key(fptr, TINT, "NSTOT", &nstot_val, "Total number of samples (SEARCH mode, else 1)", &status);
    fits_write_key(fptr, TINT, "EXTVER", &one_int, "auto assigned by template parser ", &status);
    fits_write_key(fptr, TINT, "CMPLX", &is_cmplx, "is data complex (1/0)", &status);

    std::vector<float> dat_wts(nchan);
    if (mask)
    {
        for (size_t i = 0; i < nchan; ++i)
            dat_wts[i] = static_cast<float>(mask[i]);

        for (size_t i = 0; i < nchan; ++i)
            dat_wts[i] = std::clamp(dat_wts[i], 0.0f, 1.0f);
    }
    else
    {
        std::fill(dat_wts.begin(), dat_wts.end(), 1.0f);
    }

    std::vector<float> dat_offs(nchan * npol * c);
    std::vector<float> dat_scl(nchan * npol * c);
    std::vector<char> data_int(nsblk / (8 / nbits) * nchan * npol * c);

    size_t actually_read = 0;
    size_t buf_pos = 0;

    for (int row = 1; row < nsubint + 1; ++row)
    {
        actually_read = std::min(size_t(nsblk), nstot - buf_pos);

        quantize(
                data_double + buf_pos * npol * nchan * c,
                actually_read,
                nchan,
                npol,
                data_int.data(),
                8,
                dat_scl.data(),
                dat_offs.data(),
                cmp);

        buf_pos += actually_read;

        double tsubint = tau * actually_read * 1.0e-3;
        double offs_sub = (double(row) - 0.5) * tsubint;

        fits_write_col(fptr, TDOUBLE, 1, row, 1, 1, &tsubint, &status);
        fits_write_col(fptr, TDOUBLE, 2, row, 1, 1, &offs_sub, &status);
        fits_write_col(fptr, TDOUBLE, 3, row, 1, nchan, dat_freq, &status);
        fits_write_col(fptr, TFLOAT, 4, row, 1, nchan, dat_wts.data(), &status);
        fits_write_col(fptr, TFLOAT, 5, row, 1, nchan * npol * c, dat_offs.data(), &status);
        fits_write_col(fptr, TFLOAT, 6, row, 1, nchan * npol * c, dat_scl.data(), &status);
        fits_write_col(fptr, TBYTE, 7, row, 1,
                      (actually_read * nbits / 8) * nchan * npol * c,
                      static_cast<void*>(data_int.data()), &status);
    }

    append_history(
            static_cast<size_t>(nsubint), 1, npol, nchan,
            dm, dat_freq, fcomp,
            tau, dds_mtd);

    return true;
}
