#include "Profile.h"
#include "aux_math.h"
#include "PSRFITS_Writer.h"
#include "tempo2.h"
#include "tempo2pred.h"

#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>


double Profile::get_redshift (std::string par_path, std::string site)
{

	if (!psr) load_psr(par_path, site);

	// reading pulsar parameters
	// Initialize pulsar and observation
	observation* obs = &(psr->obsn[0]);	

	double v_total[3];
	for (int i = 0; i < 3; ++i)
		v_total[i] = obs->earth_ssb[i+3] + obs->siteVel[i];

	// Project onto pulsar direction
	redshift = 0.0;
	for (int i = 0; i < 3; ++i)
		redshift += v_total[i] * psr->posPulsar[i];

	// This is z ≈ v_radial / c (special relativistic + kinematic Doppler)

	return redshift;
}



void Profile::calc_shift_int(int *shift, double DM, double fcomp, double *freqs, size_t nchann, double tau, double beta)
{

	double *dt = nullptr;
	dt = new double[nchann];

	double f, fc;

	fc = fcomp * (1.0 + beta);

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
	{
		f = freqs[i] * (1.0 + beta);
		
		dt[i] = 4.148808e6 * DM * (1.0/f/f - 1.0/fc/fc);
	}

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i) 
		shift[i] = static_cast<int> (dt[i] / tau + 0.5);

	delete[] dt;
}

void Profile::calc_shift_phase(fftw_complex *dphase, double DM, double fcomp, double *freqs, size_t nchann, double beta)
{
	double fmin = freqs[0];
	double fmax = freqs[nchann-1];
	double sign = fmin  > fmax ? 1.0 : -1.0;
	double phase, phase0;

	fcomp = (1+beta)*fcomp;
	fmin  = (1+beta)*fmin;

	phase0 = sign * 2.0e3 * M_PI * 4.148808e6 * DM * 
		std::pow(fcomp-fmin, 2) /(fmin * fmin * fcomp);

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
	{
		phase = sign * 2.0e3 * M_PI * 4.148808e6 * DM * (freqs[i]-fmin) * (freqs[i]-fmin) /
			(fmin * fmin * (freqs[i])) - phase0;

		dphase[i][0] = std::cos(phase);
		dphase[i][1] = std::sin(phase);
	}
}

void Profile::shift_window_incoherent(const double* in, double* out, const int* shift, const size_t nchann, const size_t obs_window, const double* mask)
{
	for (size_t t = 0; t < obs_window; ++t) 
	{
		#pragma omp simd
		for (size_t i = 0; i < nchann; ++i) 
			out[t * nchann + i] = in[(t+shift[i])%obs_window  * nchann + i];

		if (mask)
			math::vec_prod(out + t*nchann, const_cast<double*>(mask), nchann);
	}
}


void Profile::shift_window_coherent(fftw_plan fft, fftw_plan ifft, fftw_complex* f_space, fftw_complex* dphase, size_t nchann)
{
 	//clock_t t0, t1;;
	//t0 = std::clock();

	fftw_execute(fft);
		//t1 = std::clock();
		//std::cout << float(t1 - t0) / CLOCKS_PER_SEC << std::endl;
		//t0 = t1;


	math::vec_prod(f_space, dphase, nchann);
		//t1 = std::clock();
		//std::cout << float(t1 - t0) / CLOCKS_PER_SEC << std::endl;
		//t0 = t1;
	fftw_execute(ifft);
		//t1 = std::clock();
		//std::cout << float(t1 - t0) / CLOCKS_PER_SEC << std::endl;
		//t0 = t1;
}

void Profile::detect(fftw_complex* t_space, double* sum, size_t nchann)
{
	double re, im;

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
	{
		re = t_space[i][0]/double(nchann);
		im = t_space[i][1]/double(nchann);

		sum[i] = re*re + im*im;
	}
}

size_t Profile::fill_2d(double* data, size_t &nchann, size_t &buf_pos, size_t &buf_max, size_t &buf_size) 
{
	if (!reader || !reader->is_open) 
        throw std::runtime_error("Reader not initialized or file not open");

	size_t valid_samples = buf_max - buf_pos;
	size_t bytes_to_copy = valid_samples * nchann * sizeof(double);
	std::memmove(data, data + buf_pos * nchann, bytes_to_copy);

	buf_pos = 0;
	buf_max = valid_samples;

	size_t filled = reader-> fill_2d(data + buf_max*nchann, buf_size - buf_max, nchann);

	buf_max += filled;


    return filled;
}

size_t Profile::fill_1d(double* data, size_t& buf_pos, size_t& buf_max, size_t& buf_size) 
{
    if (!reader || !reader->is_open) 
	{
        throw std::runtime_error("Reader not initialized or file not open");
	}

	size_t valid_samples = buf_max - buf_pos;
	size_t bytes_to_copy = valid_samples * sizeof(double);
	std::memmove(data, data + buf_pos, bytes_to_copy);

	buf_pos = 0;
	buf_max = valid_samples;

	size_t filled = reader->fill_1d(data + buf_max, buf_size - buf_max);
	buf_max += filled;

    return filled;
}

void Profile::check_incoherent(size_t nchann)
{
	if (!reader || !reader->is_open) 
	{
		throw std::runtime_error("Reader not initialized or file not open");
	}

	if (hdr->nchann != nchann && hdr->nchann != 1)
	{
		throw std::runtime_error("File was recorded with different number of freqs");
	}

	if (hdr->fmin == 0.0 || hdr->fmax == 0.0)
	{
		throw std::runtime_error("Frequency information was not provided");
	}

	if (sum != nullptr)
	{
		throw std::runtime_error("The file already contains frequency averaged data");
	}

	if (!reader->allow_2d())
	{
		throw std::runtime_error("The reader prohibited to use dynamic spectrum");
	}

	// If the file was recorded in baseband
	// change its characteristics to simulated
	// spectrum recording data
	if (hdr->nchann == 1)
	{
		hdr->nchann = nchann;
		hdr-> tau = 1.0e-3 * nchann / hdr->sampling;

		hdr->tau = hdr->cmplx ? hdr->tau : 2.0*hdr->tau; 

		double fmin = hdr->fmin;
		double fmax = hdr->fmax;
		double df = (fmax - fmin) / double(nchann);

		if (hdr->freqs.size() > 0) 
			hdr->freqs.resize(nchann);

		for (size_t i = 0; i < nchann; ++i)
			hdr->freqs[i] = fmin + df * (double(i) + .5);
	}
	if (hdr->fcomp == 0.0)
	{
		hdr->fcomp = std::max(hdr->fmin, hdr->fmax);
	}

	hdr->dds_mthd = "incoherent";
}

void Profile::check_coherent()
{
	if (!reader || !reader->is_open) 
	{
		throw std::runtime_error("Reader not initialized or file not open");
	}
	if (hdr->fmin == 0.0 || hdr->fmax == 0.0)
		throw std::runtime_error("Frequency information was not provided");
	if (hdr->nchann != 1 || reader->allow_1d())
		throw std::runtime_error("Coherent dedispersion is unavailable for this file");
	if (sum != nullptr)
		throw std::runtime_error("The file already contains frequency averaged data");

	if (hdr->fcomp == 0.0)
	{
		hdr->fcomp = std::max(hdr->fmin, hdr->fmax);
	}

	if (hdr-> nchann == 1)
		hdr-> tau = 1.0e-3 / hdr->sampling;
	else
		throw std::runtime_error("The file type prohibits coherent dedispersion");

	hdr->tau = hdr->cmplx ? hdr->tau : 2.0*hdr->tau; 
	hdr->dds_mthd = "coherent";
}

void Profile::save_raw(std::string mode, std::string stream_file)
{
	if (!raw && stream_file == "")
		throw std::runtime_error("No profile to save");

	if (mode == "PSR" && hdr->cmplx)
		throw std::runtime_error("PSR file can not be complex");

	size_t nbin_orig = hdr->obs_window;
	if (nbin_orig == 0)
		hdr->obs_window = sumidx;

	PSRFITS_Writer writer(output_dir + "raw_" + reader->filename);

	writer.createPrimaryHDU(mode, hdr);

	if (stream_file != "")
		writer.append_subint(stream_file, mask);
	else
		writer.append_subint(raw, mask);

	writer.append_history("", mask);

	if (fr) writer.append_bandpass(fr);

	hdr->obs_window = nbin_orig;
}

void Profile::save_dyn(std::string mode, std::string stream_file)
{
	if (!dyn && stream_file == "")
		throw std::runtime_error("No profile to save");

	bool cmplx_orig = hdr->cmplx;
	bool cmplx = false;
	if (hdr->dds_mthd == "coherent")
		cmplx = true;

	if (mode == "PSR" && cmplx)
		throw std::runtime_error("PSR file can not be complex");

	size_t nbin_orig = hdr->obs_window;
	if (nbin_orig == 0)
		hdr->obs_window = sumidx;

	hdr->cmplx = cmplx;



	PSRFITS_Writer writer(output_dir + "dyn_" + reader->filename);

	writer.createPrimaryHDU(mode, hdr);

	if (stream_file != "")
		writer.append_subint(stream_file, mask);
	else
		writer.append_subint(dyn, mask);

	writer.append_history(hdr->dds_mthd, mask);

	if (fr) writer.append_bandpass(fr);

	hdr->cmplx = cmplx_orig;
	hdr->obs_window = nbin_orig;
}

void Profile::save_sum(std::string mode, std::string stream_file)
{
	if (!sum && stream_file == "")
		throw std::runtime_error("No profile to save");

	size_t nbin_orig = hdr->obs_window;
	if (nbin_orig == 0)
		hdr->obs_window = sumidx;

	size_t nchann_orig = hdr->nchann;
	hdr->nchann = 1;

	PSRFITS_Writer writer(output_dir + "sum_" + reader->filename);

	writer.createPrimaryHDU(mode, hdr);

	if (stream_file != "")
		writer.append_subint(stream_file, nullptr);
	else
		writer.append_subint(sum, nullptr);

	writer.append_history(hdr->dds_mthd, mask);

	hdr->nchann = nchann_orig;
	hdr->obs_window = nbin_orig;
}


void Profile::save_filt()
{
	size_t nbin_orig = hdr->obs_window;
	hdr->obs_window = 1;

	PSRFITS_Writer writer(output_dir + "wts_" + reader->filename);

	writer.createPrimaryHDU("SEARCH", hdr);

	writer.append_subint(nullptr, mask);
	writer.append_bandpass(fr);

	hdr->obs_window = nbin_orig;
}

void Profile::load_predictor(std::string pred_file)
{
	pred = new T2Predictor{};
	// T2predict takes char* as input file path
	T2Predictor_Init(pred);                // optional, but good practice

	if (T2Predictor_Read(pred, const_cast<char*>(pred_file.c_str())) != 0)
        throw std::runtime_error("Prediction file can not be loaded");

	if (hdr->t0 < T2Predictor_GetStartMJD(pred) ||
			hdr->t0 > T2Predictor_GetEndMJD(pred))
        throw std::runtime_error("Date of observation is out of range of predictor dates: " + std::to_string(hdr->t0) + " vs (" + std::to_string(T2Predictor_GetStartMJD(pred)) + ", " + std::to_string(T2Predictor_GetEndMJD(pred)) + ")");
}

void Profile::load_psr(std::string par_path, std::string site)
{
	psr = new pulsar{};

	initialiseOne(psr, 1, 0); // minimal init, with warnings enabled
	psr->nobs = 1;
	observation* obs = &(psr->obsn[0]);	

	char t2_path[500];
	strncpy(t2_path, par_path.c_str(), par_path.length());
	t2_path[par_path.length()] = '\0';
	readParfile(psr, &t2_path, nullptr, 1); /* Read .par file to define the pulsar's initial parameters */  

	// Set site arrival time and observatory
	const char* obs_code = site.c_str();
	obs->sat = hdr->t0;
	strcpy(obs->telID, obs_code);

	psr->t2cMethod = T2C_TEMPO;
	obs->clockCorr = 1;	
	obs->delayCorr = 1;	

	readEphemeris(psr, 1, 0);	// fill Earth SSB posvel
	get_obsCoord(psr, 1);		// fill siteVel
	vectorPulsar(psr, 1);	// fill pulsar position
}

void Profile::fill_PSR()
{
	// was data de-dispersed and averaged
	// over frequency channels
	if (hdr->MODE != "PSR")
		throw std::runtime_error("Attempting to load not PSR-mode file");

	bool dd = ! (hdr->dds_mthd == "");
	bool avg = (hdr->nchann == 1);

	size_t nsub = hdr->nsubint;
	size_t nbin = hdr->obs_window;
	size_t npol = hdr->nchann;
	size_t nchann = hdr->nchann;

	size_t buf_pos = 0;
	size_t buf_max = 0;
	size_t buf_size = nsub * nbin * npol * nchann;

	// read data from the file
	double *buff = nullptr;
	buff = new double[buf_size];
	size_t filled = fill_2d(buff, nchann, buf_pos, buf_max,buf_size);


	if (dd && avg)
		this->sum = buff;
	else if (dd && !avg)
		this->dyn = buff;
	else if (!dd && !avg)
		this->raw = buff;
	else if (!dd && avg || filled == 0)
	{
		delete [] buff;
		throw std::runtime_error("Unable to load SEARCH-file data");
	}
}

void Profile::fill_SEARCH()
{
	if (! reader->check_fit())
		throw std::runtime_error("Attempting to load large SEARCH-mode file (file is bigger than the buffer)");

	bool dd = ! (hdr->dds_mthd == "");
	bool avg = (hdr->nchann == 1);

	size_t npol = hdr->nchann;
	size_t nchann = hdr->nchann;
	size_t time_steps = std::min(hdr->OBS_SIZE, hdr->CUT_SIZE);

	size_t buf_pos = 0;
	size_t buf_max = 0;
	size_t buf_size = time_steps * npol * nchann;

	// read data from the file
	double *buff = nullptr;
	buff = new double[buf_size];

	size_t filled = 0;
	if (reader->allow_1d())
		filled = fill_1d(buff, buf_pos, buf_max,buf_size);
	else if (reader->allow_2d())
		filled = fill_2d(buff, nchann, buf_pos, buf_max,buf_size);


	if (dd && avg)
		sum = buff;
	else if (dd && !avg)
		dyn = buff;
	else if (!dd && !avg)
		raw = buff;
	else if (!dd && avg || filled == 0)
	{
		delete [] buff;
		throw std::runtime_error("Unable to load SEARCH-file data");
	}

}

void Profile::normilize(double BL_window)
{
	size_t bl_win_points = 0;

	if (BL_window > 0.0)
		bl_win_points = BL_window / (hdr->tau*1.0e-3);
	else
	{
		bl_win_points = hdr->obs_window / 3;
		bl_win_points = std::max(100ul, bl_win_points);
	}

	// Apply the normalization routine 

	math::subtract_baseline(sum, hdr->obs_window, bl_win_points);
	math::normalize_std(sum, hdr->obs_window, bl_win_points);
}
