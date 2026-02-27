#include "Profile.h"
#include "aux_math.h"
#include "PSRFITS_Writer.h"

#include <iostream>
#include <algorithm>

std::string Profile::dedisperse_incoherent_stream(
		double DM, size_t nchann,
		bool is_save_raw, bool is_save_dyn, bool is_save_sum
		)
{
	check_incoherent(nchann);

	size_t obs_window;
	double tau;
	size_t n_DM;
	double fcomp, fmin, fmax;
	size_t buf_pos, buf_max;

	int *shift = nullptr;
	double *pre = nullptr, *post = nullptr;

	std::ofstream raw_output, dyn_output, sum_output;
	std::srand(time(NULL));
	std::string id = "";
	id += std::to_string(std::rand());
	id += ".bin";

	if (is_save_raw)
		raw_output = std::ofstream(output_dir + "raw_" + id);
	if (is_save_dyn)
		dyn_output = std::ofstream(output_dir + "dyn_" + id);
	if (is_save_sum)
		sum_output = std::ofstream(output_dir + "sum_" + id);


	fmin = hdr->fmin;
	fmax = hdr->fmax;
	fcomp = hdr->fcomp;
	tau = hdr->tau;


	double dtmax = 4.15e6 * DM * std::abs(1/fmin/fmin - 1/fmax/fmax);
	n_DM = static_cast<size_t>(dtmax / tau) + 1;
	n_DM += n_DM % 2;

	// set 256 MiB buffer as standard size
	obs_window = std::max(n_DM, (256ul << 20)/nchann/hdr->npol/sizeof(double)); 
	obs_window = std::min(obs_window, hdr->OBS_SIZE);
	obs_window += n_DM;


	shift = new int[nchann];
	pre = new double[obs_window*nchann];
	post = new double[obs_window*nchann];
	sum = new double[obs_window - n_DM];

	calc_shift_int(shift, DM, fcomp, hdr->freqs.data(), nchann, tau, redshift);

	buf_pos = 0;
	buf_max = 0;
	sumidx = 0;

	bool eof = false;
	while(true && !eof)
	{
		if (buf_pos + obs_window >= buf_max)
			fill_2d(pre, nchann, buf_pos, buf_max, obs_window);

		if (buf_max < obs_window)
		{
			eof = true; // EOF is reached

			// zero padding to save the last part of the file
			std::fill(pre + buf_max*nchann, pre + obs_window*nchann, 0.0);
		}

		if (buf_max < n_DM)
			continue;


		if (fr)
		{
			for (size_t i = buf_pos; i < buf_max; ++i)
			{
				math::vec_sub(pre + i*nchann, fr, nchann);
				math::vec_div(pre + i*nchann, fr, nchann);
			}
		}

		shift_window_incoherent(pre, post, shift, nchann, obs_window, mask);


		for (size_t t = 0; t < obs_window - n_DM; ++t) 
			sum[t] = std::accumulate(post + t*nchann, post + (t+1)*nchann, 0.0);

		// save processed buffer
		// regecting last n_DM inputs


		if (is_save_raw)
			raw_output.write(reinterpret_cast<const char*>(pre),
					nchann * (buf_max - n_DM) * sizeof(double));

		if (is_save_dyn)
			dyn_output.write(reinterpret_cast<const char*>(post),
					nchann * (buf_max - n_DM) * sizeof(double));

		if (is_save_sum)
			sum_output.write(reinterpret_cast<const char*>(sum),
					(buf_max - n_DM) * sizeof(double));

		buf_pos = buf_max - n_DM;
		sumidx += buf_pos;
		std::cout << "t = " << reader->point2time(sumidx) << " ms" << std::endl;
	}

	if (sumidx == 0)
	{
		if (is_save_raw)
		{
			raw_output.close();
			std::remove((output_dir + "raw_" + id).c_str());
		}

		if (is_save_dyn)
		{
			dyn_output.close();
			std::remove((output_dir + "dyn_" + id).c_str());
		}

		if (is_save_sum)
		{
			sum_output.close();
			std::remove((output_dir + "sum_" + id).c_str());
		}

		id = "";
	}
	else 
	{
		if (is_save_raw)
			raw_output.close();

		if (is_save_dyn)
			dyn_output.close();

		if (is_save_sum)
			sum_output.close();
	}

	delete[] pre;
	pre = nullptr;
	delete[] post;
	post = nullptr;
	delete[] sum;
	sum = nullptr;
	delete[] shift;
	shift = nullptr;

	return id;
}

std::string Profile::dedisperse_coherent_stream(
		double DM, size_t nchann,
		bool is_save_raw, bool is_save_dyn, bool is_save_sum
		)
{
	check_coherent();

	std::ofstream raw_output, dyn_output, sum_output;

	double *buff;
	size_t n_DM, obs_window;
	size_t buf_pos, buf_max;

	double fcomp, fmin, fmax;
	double tau;

	fftw_complex* dphase;
	fftw_complex *f_space, *t_space;
	fftw_plan fft, ifft;


	/******************************************
	 * This part allows to plot output spectrum
	 * for the processed chunk. It is used for debugging
	 ******************************************
	 */
	// fftw_plan p;
	// fftw_complex *f_small, *t_small;
	// size_t freq_num = 2048;
	// double *spec;
	// f_small = (fftw_complex*)(fftw_malloc(sizeof(fftw_complex) * (freq_num)));
	// t_small = (fftw_complex*)(fftw_malloc(sizeof(fftw_complex) * (freq_num)));
	// spec = (double*)(fftw_malloc(sizeof(double) * (freq_num)));
	// p  = fftw_plan_dft_1d(freq_num, t_small, f_small, FFTW_FORWARD, FFTW_ESTIMATE);
	// std::ofstream output(output_dir + "spectrum.bin");
	/* ========== */

	fmin = hdr->fmin;
	fmax = hdr->fmax;
	fcomp = hdr->fcomp;
	tau = hdr->tau;


	double dtmax = 4.15e6 * DM * (std::pow(std::min(fmax, fmin), -2.0) - std::pow(std::max(fmax, fmin), -2.0)); // ms
	n_DM = static_cast<size_t>(dtmax/tau);
	n_DM += n_DM % 2;
	
	if (nchann <= 2*n_DM)
        throw std::runtime_error("The number of channels is too small for coherent dedispersion. Set at least 2^" + std::to_string(size_t(std::log2(n_DM)) + 2));
	obs_window = 2*nchann;


	dphase = (fftw_complex*)(fftw_malloc(sizeof(fftw_complex) * (nchann)));
	calc_shift_phase(dphase, DM, fcomp, hdr->freqs.data(), nchann, redshift);

	if (mask)
		math::vec_prod(dphase, mask, nchann);


	buff    = (double*) (fftw_malloc(sizeof(double) * obs_window));
	sum     = (double*) (fftw_malloc(sizeof(double) * nchann));
	t_space = (fftw_complex*) (fftw_malloc(sizeof(fftw_complex) * (nchann)));
	f_space = (fftw_complex*) (fftw_malloc(sizeof(fftw_complex) * (nchann+1)));

	if (hdr->cmplx)
		fft = fftw_plan_dft_1d(nchann, (fftw_complex*) buff, f_space, FFTW_FORWARD, FFTW_ESTIMATE);
	else
		fft  = fftw_plan_dft_r2c_1d(obs_window, buff, f_space, FFTW_ESTIMATE);


	ifft = fftw_plan_dft_1d(nchann, f_space+1, t_space, FFTW_BACKWARD, FFTW_ESTIMATE);


	//std::srand(time(NULL));
	std::string id = "";
	id += std::to_string(std::rand());
	id += ".bin";

	if (is_save_raw)
		raw_output = std::ofstream(output_dir + "raw_" + id);

	if (is_save_dyn)
		dyn_output = std::ofstream(output_dir + "dyn_" + id);

	if (is_save_sum)
		sum_output = std::ofstream(output_dir + "sum_" + id);

	std::fill(buff, buff + 2*n_DM, 0.0);
	buf_max = 2*n_DM; 
	buf_pos = 0;
	sumidx = 0;


	bool eof = false;
	bool beginning = true;
	size_t N = 0;
	while(!eof)
	{
		if (buf_pos + obs_window >= buf_max)
			fill_1d(buff, buf_pos, buf_max, obs_window);

		if (buf_max < obs_window)
		{
			eof = true; // EOF is reached
			std::fill(buff + buf_max, buff + obs_window, 0.0);
		}

		if (beginning)
		{
			beginning = false;
			buf_pos = 2*n_DM;
		}

		N = buf_max/2 - n_DM - buf_pos/2;

		if (hdr->cmplx)
			shift_window_coherent(fft, ifft, f_space, dphase, nchann);
		else
			shift_window_coherent(fft, ifft, f_space+1, dphase, nchann);


		detect(t_space, sum, nchann);


		if (is_save_raw)
			raw_output.write(reinterpret_cast<const char*>(buff + buf_pos),
					(2*N) * sizeof(double));

		if (is_save_dyn)
			dyn_output.write(reinterpret_cast<const char*>(t_space + buf_pos/2),
					(N) * sizeof(fftw_complex));

		if (is_save_sum)
			sum_output.write(reinterpret_cast<const char*>(sum + buf_pos/2),
					(N) * sizeof(double));

		buf_pos = buf_max - 2*n_DM;
		sumidx += N;
		std::cout << "t = " << reader->point2time(sumidx) << " ms" << std::endl;

		/******************************************
		 * This part allows to plot output spectrum
		 * for the processed chunk. It is used for debugging
		 * (uncomment the section below to use it 
		 * AND the same section above)
		 ******************************************
		 */
		// double re, im;
		// for (size_t i = 0; i < (buf_max/2 - n_DM)/freq_num; ++i)
		// {
		// #pragma omp simd
		// 	for (size_t k = 0; k < freq_num; ++k)
		// 	{
		// 		t_small[k][0] = t_space[i*freq_num + k][0];
		// 		t_small[k][1] = t_space[i*freq_num + k][1];
		// 	}

		// 	fftw_execute(p);

		// #pragma omp simd
		// 	for (size_t k = 0; k < freq_num; ++k)
		// 	{
		// 		re = f_small[k][0];
		// 		im = f_small[k][1];

		// 		spec[k] = re*re + im*im;
		// 	}

		// 	output.write(reinterpret_cast<const char*>(spec),
		// 			(freq_num) * sizeof(double));
		// }
		/* ========== */
	}

	if (sumidx == 0)
	{
		if (is_save_raw)
		{
			raw_output.close();
			std::remove((output_dir + "raw_" + id).c_str());
		}

		if (is_save_dyn)
		{
			dyn_output.close();
			std::remove((output_dir + "dyn_" + id).c_str());
		}

		if (is_save_sum)
		{
			sum_output.close();
			std::remove((output_dir + "sum_" + id).c_str());
		}

		id = "";
	}
	else 
	{
		if (is_save_raw)
			raw_output.close();

		if (is_save_dyn)
			dyn_output.close();

		if (is_save_sum)
			sum_output.close();
	}

	fftw_destroy_plan(fft);
	fftw_destroy_plan(ifft);
	fftw_free(buff);
	fftw_free(sum);
	fftw_free(dphase);
	fftw_free(f_space);
	fftw_free(t_space);

	return id;
}
