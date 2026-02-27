#include "Profile.h"
#include "aux_math.h"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector>


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


	if (!dyn) dyn = new double[obs_window*nchann];
	if (!sum) sum = new double[obs_window];


	if (DM > 0.0)
	{
		shift = new int[nchann];

		calc_shift_int(shift, DM, fcomp, hdr->freqs.data(), nchann, tau, redshift);
		shift_window_incoherent(raw, dyn, shift, nchann, obs_window, mask);

		delete[] shift;
	}

	for (size_t t = 0; t < obs_window; ++t) 
		sum[t] = std::accumulate(dyn + t*nchann, dyn + (t+1)*nchann, 0.0);
}



void Profile::dedisperse_coherent(double DM, size_t nchann)
{
	throw std::runtime_error("The function development is in progress");
}


void Profile::fold_dyn(double P, size_t nchann)
{
	check_incoherent(nchann);

	size_t obs_window;
	size_t npol;
	double tau;

	double *buff = nullptr, *buff_curr = nullptr;
	size_t buf_pos, buf_max, buf_size;

	// vars for time correction
	size_t rev = 0;
	double diff = 0.0;
	int diff_int = 0;


	if (hdr->nchann == 1)
	{
		hdr-> tau = 
			2.0e-3 * nchann / hdr->sampling;
		hdr-> nchann = nchann; 
	}
	else if (hdr->nchann != nchann)
	{
		throw std::runtime_error("Profile was recorded with different number of frequency channels");
	}

	tau = hdr->tau;
	obs_window = hdr->obs_window;
	npol = hdr->npol;
	
	if (obs_window <= 1) 
	{
		obs_window = size_t(P*1e3 / tau);
		hdr->obs_window = obs_window;
	}
	buf_size = 2*obs_window;


	if (obs_window * hdr->tau > P*1e3)
		throw std::runtime_error("Observational window must be less than period!");
	if (obs_window == 0)
		throw std::runtime_error("Observational window is zero. Check input period and/or observational window");


	raw = new double [obs_window * nchann * npol];
	buff = new double [buf_size * nchann * npol];

	std::fill(raw, raw + obs_window*nchann*npol, 0.0);

	buf_pos = 0;
	buf_max = 0;
	rev = 0;
	sumidx = 0;

	while(true)
	{
		if (buf_pos + obs_window >= buf_max)
			fill_2d(buff, nchann, buf_pos, buf_max, buf_size);

		if (buf_pos + obs_window >= buf_max)
			break; // EOF is reached

		buff_curr = buff + buf_pos * nchann * npol;

		if (fr)
		{
			for (size_t i = 0; i < obs_window; ++i)
			{
				math::vec_sub(buff_curr + i*nchann, fr, nchann);
				math::vec_div(buff_curr + i*nchann, fr, nchann);
			}
		}

		math::vec_add(raw, buff_curr, obs_window*nchann*npol);

		rev += 1;
		sumidx += obs_window;
		buf_pos += obs_window;


		//correct for integer number of points in the observarional window
		diff = (rev*P - reader->point2time(sumidx))*1e3;
		diff_int = int(diff/tau + .5);

		// time stamps may jump between periods, so the additional
		// jump may worsen the coherence
		if (std::abs(rev*P - reader->point2time(sumidx + diff_int)) < std::abs(diff))
		{
			buf_pos += diff_int;
			sumidx  += diff_int;
		}

		std::cout << "\r\033[K"; // move to the beginning of the line and clear the line
		std::cout << "rev: " << rev << " diff: " << std::setw(5);
		std::cout << diff_int * tau * 1e3 << " us" << std::flush;
	}

	#pragma omp simd
	for(size_t i = 0; i < obs_window*nchann*npol; ++i)
		raw[i] = raw[i] / double(rev);

	hdr->nsubint = 1;
	hdr->period = P;
	hdr->obs_window = obs_window;

	delete[] buff;
	buff = nullptr;
}	





void Profile::fold_dyn(std::string pred_file, size_t nchann)
{
	check_incoherent(nchann);

	if (!pred) load_predictor(pred_file);

	std::cout << "Integrating pulse using prediction file for " << 
    T2Predictor_GetSiteName(pred) << " telescope" << std::endl;

	double P;
	size_t obs_window;
	size_t npol;
	double tau;
	long double phase, t0, phase0;

	double *buff = nullptr, *buff_curr = nullptr;
	size_t buf_pos, buf_max, buf_size;

	// vars for time correction
	size_t rev = 0;
	double diff;

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

	// Produce obsservational window corresponding to the initial period value
	// if the file has no own obs_window (search mode)
	t0 = hdr->t0;
	tau = hdr->tau;
	phase0 = fmodl(T2Predictor_GetPhase(pred, t0, fcomp), 1.0L);
	P = 1.0 / T2Predictor_GetFrequency(pred, t0, fcomp);

	if (hdr->period == 0.0)
		hdr->obs_window = size_t(P*1e3 / tau);

	obs_window = hdr->obs_window;
	npol = hdr->npol;
	buf_size = 2*obs_window;

	if (obs_window * tau > P*1e3)
		throw std::runtime_error("Observational window must be less than period!");


	raw = new double [obs_window*nchann*npol];
	buff = new double [buf_size * nchann * npol];

	std::fill(raw, raw + obs_window*nchann*npol, 0.0);

	buf_pos = 0;
	buf_max = 0;
	rev = 0;
	sumidx = 0;

	while(true)
	{
		if (buf_pos + obs_window >= buf_max)
			fill_2d(buff, nchann, buf_pos, buf_max, buf_size); 


		if (buf_pos + obs_window >= buf_max)
			break; // EOF is reached
			

		buff_curr = buff + buf_pos * nchann;

		if (fr)
		{
			for (size_t i = 0; i < obs_window; ++i)
			{
				math::vec_sub(buff_curr + i*nchann, fr, nchann);
				math::vec_div(buff_curr + i*nchann, fr, nchann);
			}
		}

		math::vec_add(raw, buff_curr, obs_window*nchann*npol);

		buf_pos += obs_window;
		sumidx += obs_window;	

		rev += 1;

		//correct for integer number of points in the observarional window
		phase = fmodl(T2Predictor_GetPhase(pred, t0 + (long double) reader->point2time(sumidx)/86400.0L, fcomp) - phase0, 1.0L);

		diff = (1.0-phase);
		if (diff > .5) diff -= 1.0;

		diff *= P*1.0e3;

		buf_pos += static_cast<int> (diff/tau + .5);
		sumidx  += static_cast<int> (diff/tau + .5);

		std::cout << "\r\033[K"; // move to the beginning of the line and clear the line
		std::cout << "rev: " << rev << " diff: " << std::setw(5);
		std::cout << int((rev*P - reader->point2time(sumidx))*1e6) << " us" << std::flush;

	}
	std::cout<<std::endl;

	#pragma omp simd
	for(size_t i = 0; i < obs_window*nchann*npol; ++i)
		raw[i] = raw[i] / double(rev);

	hdr->nsubint = 1;
	hdr->period = P;
	hdr->obs_window = obs_window;

	delete[] buff;
	buff = nullptr;
}
