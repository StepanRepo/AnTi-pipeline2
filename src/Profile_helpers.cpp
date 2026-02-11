#include "Profile.h"
#include "aux_math.h"

#include <cstddef>
#include <cstring>
#include <iostream>

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
		{
			#pragma omp simd
			for (size_t i = 0; i < nchann; ++i) 
				out[t * nchann + i] *= mask[i];
		}
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

		if (hdr->freqs) delete[] hdr->freqs;
		hdr->freqs = new double[nchann];

		for (size_t i = 0; i < nchann; ++i)
			hdr->freqs[i] = fmin + df * (double(i) + .5);
	}
	if (hdr->fcomp == 0.0)
	{
		hdr->fcomp = std::max(hdr->fmin, hdr->fmax);
	}
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
}
