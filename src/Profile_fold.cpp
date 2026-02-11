#include "Profile.h"
#include "aux_math.h"
#include "PSRFITS_Writer.h"
#include "tempo2pred.h"  // API for TEMPO2 prediction files

#include <iostream>
#include <iomanip>
#include <cstring>


void Profile::dedisperse_incoherent(double DM, size_t nchann)
{
	check_incoherent(nchann);

	size_t obs_window;
	double tau;

	double fcomp;
	int *shift;

	fcomp = hdr->fcomp;
	tau = hdr->tau;
	nchann = hdr->nchann;
	obs_window = hdr->obs_window;


	shift = new int[nchann];
	dyn = new double[obs_window*nchann];
	sum = new double[obs_window];


	calc_shift_int(shift, DM, fcomp, hdr->freqs, nchann, tau, redshift);
	shift_window_incoherent(raw, dyn, shift, nchann, obs_window, mask);


	for (size_t t = 0; t < obs_window; ++t) 
		sum[t] = std::accumulate(dyn + t*nchann, dyn + (t+1)*nchann, 0.0);


	if (save_dyn)
	{
		PSRFITS_Writer writer(output_dir + "dyn_" + reader->filename);
		writer.createPrimaryHDU("PSR", hdr);
		writer.append_subint_fold(
				dyn, hdr->freqs, nullptr, 
				obs_window, nchann, hdr->npol, 
				hdr->period, DM, fcomp, tau,
				"incoherent");
	}

	if (save_sum)
	{
		PSRFITS_Writer writer(output_dir + "sum_" + reader->filename);
		writer.createPrimaryHDU("PSR", hdr);
		writer.append_subint_fold(
				sum, 
				new double(fcomp), nullptr, 
				obs_window, 1, hdr->npol, 
				hdr->period, DM, fcomp, tau,
				"incoherent");
	}

	delete[] shift;
}



void Profile::dedisperse_coherent(double DM, size_t nchann)
{
	throw std::runtime_error("The function development is in progress");
}


void Profile::fold_dyn(double P, size_t nchann)
{
	if (!reader || !reader->is_open) 
		throw std::runtime_error("Reader not initialized or file not open");


	size_t obs_window;
	double tau;

	double *buff = nullptr, *buff_curr = nullptr;
	size_t buf_pos, buf_max, buf_size;

	// vars for time correction
	size_t rev = 0;
	sumidx = 0;
	double diff;


	if (hdr->nchann == 1)
	{
		hdr-> tau = 
			2.0e-3 * nchann / hdr->sampling;
		hdr-> nchann = nchann; 
	}
	else
	{
		throw std::runtime_error("Profile was recorded with different number of frequency channels");
	}

	tau = hdr->tau;
	obs_window = size_t(P*1e3 / tau);
	buf_size = 2*obs_window;


	hdr->obs_window = obs_window;

	if (obs_window * hdr->tau > P*1e3)
		throw std::runtime_error("Observational window must be less than period!");


	raw = new double [obs_window*nchann];
	buff = new double [buf_size * nchann];

	std::fill(raw, raw + obs_window*nchann, 0.0);

	buf_pos = 0;
	buf_max = 0;
	rev = 0;

	while(true)
	{
		if (buf_pos + obs_window >= buf_max)
			fill_2d(buff, nchann, buf_pos, buf_max, buf_size);

		if (buf_pos + obs_window >= buf_max)
			break; // EOF is reached
			


		buff_curr = buff + buf_pos * nchann;
		math::vec_add(raw, buff_curr, obs_window*nchann);

		rev += 1;

		//correct for integer number of points in the observarional window
		diff = (rev*P - reader->point2time(sumidx))*1e3;
		buf_pos += static_cast<size_t> (diff/tau + .5);
		sumidx += static_cast<size_t> (diff/tau + .5);

		std::cout << "\r\033[K"; // move to the beginning of the line and clear the line
		std::cout << "rev: " << rev << " diff: " << std::setw(5);
		std::cout << int((rev*P - reader->point2time(sumidx))*1e6) << " us" << std::flush;
	}
	std::cout << std::endl;

	#pragma omp simd
	for(size_t i = 0; i < obs_window*nchann; ++i)
		raw[i] = raw[i] / double(rev);

	if (save_raw)
	{
		PSRFITS_Writer writer(output_dir + "raw_" + reader->filename);
		writer.createPrimaryHDU("PSR", hdr);
		writer.append_subint_fold(
				raw, hdr->freqs, mask, 
				obs_window, nchann, hdr->npol, 
				hdr->period, 0.0, hdr->fcomp, tau, "");
	}

	delete[] buff;
	buff = nullptr;
}	

void Profile::fold_dyn(std::string pred_file, size_t nchann)
{
	if (!reader || !reader->is_open) 
        throw std::runtime_error("Reader not initialized or file not open");


	// T2predict takes char* as input file path
	char* pred_file_c = new char[pred_file.length() + 1];
	strcpy(pred_file_c, pred_file.c_str());


	T2Predictor *pred = nullptr;
	pred = new T2Predictor();

	if (T2Predictor_Read(pred, pred_file_c) != 0)
        throw std::runtime_error("Prediction file can not be loaded");

	if (hdr->t0 < T2Predictor_GetStartMJD(pred) ||
			hdr->t0 > T2Predictor_GetEndMJD(pred))
        throw std::runtime_error("Date of observation is out of range of predictor dates: " + std::to_string(hdr->t0) + " vs (" + std::to_string(T2Predictor_GetStartMJD(pred)) + ", " + std::to_string(T2Predictor_GetEndMJD(pred)) + ")");

	std::cout << "Integrating pulse using prediction file for " << 
    T2Predictor_GetSiteName(pred) << " telescope" << std::endl;


	double P;
	size_t obs_window;
	double tau;
	long double phase, t0, phase0;

	double *buff = nullptr, *buff_curr = nullptr;
	size_t buf_pos, buf_max, buf_size;

	// vars for time correction
	size_t rev = 0;
	double diff;
	sumidx = 0;

	// vars for frequency correction
	double fmin, fmax, fcomp;

	fmin = hdr->fmin;
	fmax = hdr->fmax;
	fcomp = hdr->fcomp;

	if (fcomp == 0.0)
	{
		fcomp = std::max(fmin, fmax);
		hdr->fcomp = fcomp;
	}

	if (hdr->nchann == 1)
	{
		hdr-> tau = 
			2.0e-3 * nchann / hdr->sampling;
		hdr-> nchann = nchann; 
	}
	else
	{
		throw std::runtime_error("Profile was recorded with different number of frequency channels");
	}



	// Make obsservational window corresponding to the initial period value
	t0 = hdr->t0;
	phase0 = fmodl(T2Predictor_GetPhase(pred, t0, fcomp), 1.0L);
	P = 1.0 / T2Predictor_GetFrequency(pred, t0, fcomp);
	tau = hdr->tau;
	obs_window = size_t(P*1e3 / tau);
	hdr->obs_window = obs_window;
	buf_size = 2*obs_window;



	if (obs_window * tau > P*1e3)
		throw std::runtime_error("Observational window must be less than period!");


	raw = new double [obs_window*nchann];
	buff = new double [buf_size * nchann];

	std::fill(raw, raw + obs_window*nchann, 0.0);

	buf_pos = 0;
	buf_max = 0;
	rev = 0;

	while(true)
	{
		if (buf_pos + obs_window >= buf_max)
			fill_2d(buff, nchann, buf_pos, buf_max, buf_size); 


		if (buf_pos + obs_window >= buf_max)
			break; // EOF is reached
			


		buff_curr = buff + buf_pos * nchann;
		
		math::vec_add(raw, buff_curr, obs_window*nchann);


		buf_pos += obs_window;
		sumidx += obs_window;	

		rev += 1;

		//correct for integer number of points in the observarional window
		phase = fmodl(T2Predictor_GetPhase(pred, t0 + reader->point2time(sumidx)/86400.0L, fcomp) - phase0, 1.0L);
		diff = (1-phase) * P * 1.0e3;
		if (diff > .5)
			diff -= P*1.0e3;

		buf_pos += static_cast<int> (diff/tau + .5);
		sumidx  += static_cast<int> (diff/tau + .5);

		std::cout << "\r\033[K"; // move to the beginning of the line and clear the line
		std::cout << "rev: " << rev << " diff: " << std::setw(5);
		std::cout << int((rev*P - reader->point2time(sumidx))*1e6) << " us" << std::flush;

	}
	std::cout<<std::endl;

	#pragma omp simd
	for(size_t i = 0; i < obs_window*nchann; ++i)
		raw[i] = raw[i] / double(rev);

	if (save_raw)
	{
		PSRFITS_Writer writer(output_dir + "raw_" + reader->filename);
		writer.createPrimaryHDU("PSR", hdr);
		writer.append_subint_fold(
				raw, hdr->freqs, mask, 
				obs_window, nchann, 1, 
				hdr->period, 0.0, hdr->fcomp, tau, "");
	}

	delete[] buff;
	buff = nullptr;
}
