#include "Profile.h"
#include "aux_math.h"
#include "PSRFITS_Writer.h"

#include <cstddef>
#include <iostream>
#include <iomanip>


void Profile::matched_filter(double* data, size_t N, double threshold, std::vector<size_t>& pos, std::vector<double>& power)
{
    thread_local static std::vector<signed char> mask, edges;

    if (mask.size() != N) mask.resize(N);
    if (edges.size() != N) edges.resize(N);

	std::vector<size_t> rises(0), falls(0); 


	for (size_t i = 0; i < N; ++i)
		mask[i] = data[i] < threshold ? 0 : 1;

	for (size_t i = 0; i < N-1; ++i)
		edges[i] = mask[i+1] - mask[i];
	edges[N-1] = 0.0;

	for (size_t i = 0; i < N; ++i)
	{
		if (edges[i] > 0.0) rises.push_back(i);
		if (edges[i] < 0.0) falls.push_back(i);
	}

	if (rises.size() > falls.size())
		rises.pop_back();

	if (rises.size() < falls.size())
		falls.erase(falls.begin());

	pos.resize(2*rises.size());
	power.resize(rises.size());

	size_t a, b;
	for(size_t i = 0; i < rises.size(); ++i)
	{
		a = rises[i];
		b = falls[i];
		//power[i] = math::mean(data + a+1, b-a);
		power[i] = (b-a > 0) ? *std::max_element(data+a, data+b) : 0.0;
	}


	size_t counter = 0;
	for (size_t i = 0; i < rises.size(); ++i)
	{
		if (power[i] < threshold) continue;

		pos[2*counter] = rises[i];
		pos[2*counter+1] = falls[i];
		power[counter] = power[i];
		counter ++;
	}

	pos.resize(2*counter);
	power.resize(counter);
}

std::string Profile::csv_result (size_t left, size_t right, double power, size_t id) const
{
	std::ostringstream row("");
	row << std::fixed << std::setprecision(14); // f5.14 (down to ns in MJD)  

	long double mjd  = hdr->t0 + reader->point2time(left)/86400.0L;
	double sec = double(right+left)/2.0 * hdr->tau*1e-3;
	double width = double(right-left) * hdr->tau*1e-3;


	row << reader -> filename  << "; ";
	row << id << "; ";

	row << std::fixed << std::setprecision(16); 
	row << mjd << "; ";

	row << std::fixed << std::setprecision(6); 
	row << sec << "; ";

	row << std::fixed << std::setprecision(9); // f.9 
	row << width << "; ";
	row << std::fixed << std::setprecision(3); // f.9 
	row << power << "; ";

	return row.str();
}

std::string Profile::dedisperse_coherent_search(
		double DM, size_t nchann, 
		double BL_window_s, double threshold, 
		std::string conv_type, double fwhm)
{

	check_coherent();

	double *buff;
	size_t n_DM, obs_window;
	size_t BL_window;
	size_t buf_pos, buf_max;

	double fcomp, fmin, fmax;
	double tau;
	long double t0;

	fftw_complex *dphase;
	fftw_complex *f_space, *t_space;
	fftw_complex *conv_f_space;
	fftw_complex *ker_f = nullptr;
	double 		 *sum_dm1, *conv_t_space;
	fftw_plan 	 fft, ifft;
	fftw_plan 	 conv_fft, conv_ifft;

	fmin  = hdr->fmin;
	fmax  = hdr->fmax;
	fcomp = hdr->fcomp;
	tau   = hdr->tau;
	t0    = hdr->t0;

	double dtmax = 4.15e6 * DM * (std::pow(std::min(fmax, fmin), -2) - std::pow(std::max(fmax, fmin), -2));
	n_DM = static_cast<size_t>(dtmax/tau);
	n_DM += n_DM % 2;
	
	if (nchann <= n_DM)
        throw std::runtime_error("The number of channels is too small for coherent dedispersion. Set at least 2^" + std::to_string(size_t(std::log2(n_DM)) + 1));
	obs_window = 2*nchann;


	dphase  = (fftw_complex*)(fftw_malloc(sizeof(fftw_complex) * (nchann)));
	calc_shift_phase(dphase, DM, fcomp, hdr->freqs, nchann, redshift);


	// Modify phase shift with sensitivity 
	// correction for every channel
	if (mask)
	{
		math::vec_prod(dphase, mask, nchann);
	}

	// Set up the convolution kernel
	if(conv_type == "gaussian")
	{

		double* ker_t = new double[nchann];
		std::fill(ker_t, ker_t + nchann, 0);
		size_t n = size_t((5.0e3*fwhm)/tau) + 1;
		math::gaussian_kernel(ker_t, n, fwhm*1e3/tau);


		ker_f   = (fftw_complex*)(fftw_malloc(sizeof(fftw_complex) * (nchann)));
		std::fill((double*) ker_f, ((double*) ker_f) + 2*nchann, 0.0);
		for (size_t i = 0; i < nchann; ++i) ker_f[i][0] = ker_t[i];

		fft = fftw_plan_dft_1d(nchann, ker_f, ker_f, FFTW_BACKWARD, FFTW_ESTIMATE);
		fftw_execute(fft);
		fftw_destroy_plan(fft);

		math::vec_prod(ker_f, 1.0/nchann, nchann);

		delete[] ker_t;
		ker_t = nullptr;
	}




	buff    = (double*) (fftw_malloc(sizeof(double) * obs_window));
	sum_dm1 = (double*) (fftw_malloc(sizeof(double) * nchann));
	t_space = (fftw_complex*) (fftw_malloc(sizeof(fftw_complex) * (nchann)));
	f_space = (fftw_complex*) (fftw_malloc(sizeof(fftw_complex) * (nchann+1)));
	conv_t_space = (double*) (fftw_malloc(sizeof(double) * nchann));
	conv_f_space = (fftw_complex*) (fftw_malloc(sizeof(fftw_complex) * (nchann+1)));

	fft  = fftw_plan_dft_r2c_1d(obs_window, buff, f_space, FFTW_ESTIMATE);
	ifft = fftw_plan_dft_1d(nchann, f_space+1, t_space, FFTW_BACKWARD, FFTW_ESTIMATE);
	conv_fft = fftw_plan_dft_r2c_1d(nchann, sum_dm1, conv_f_space, FFTW_ESTIMATE);
	conv_ifft = fftw_plan_dft_c2r_1d(nchann, conv_f_space, conv_t_space, FFTW_ESTIMATE);


	std::srand(time(NULL));
	std::string id = "";
	id += std::to_string(std::rand());
	id += ".csv";
	std::ofstream csv(output_dir + id);

	// write table header
	csv << "file; num; MJD; sec; width; power;" << std::endl;
	size_t empty_size =  csv.tellp();

	buf_pos = 0;
	buf_max = 0;
	sumidx = 0;

	std::vector<size_t> pulses_dm1;
	std::vector<double> power_dm1;
	size_t N = 0;
	bool eof = false;
	bool beginning = true;
	bool is_pulse = false;
	size_t n_found = 0;

	BL_window = size_t(BL_window_s/hdr->tau * 1.0e3);


	// Pad the beginning of the buffer with zeros
	// to preserve the beginning of the file
	std::fill(buff, buff + 2*n_DM, 0.0);
	buf_max = 2*n_DM;

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
		shift_window_coherent(fft, ifft, f_space+1, dphase, nchann);
		detect(t_space, sum_dm1, nchann);


		// Convolve the de-disperced window
		if (ker_f)
		{
			shift_window_coherent(conv_fft, conv_ifft, conv_f_space+1, ker_f, nchann);
		}
		else if (conv_type == "box")
		{
			math::box_conv(sum_dm1, conv_t_space, size_t(fwhm * 1e3/tau + .5),  nchann);
		}
		else
		{
			conv_t_space = sum_dm1;
		}

		// Find pulses with overlapping windows
		math::subtract_baseline(conv_t_space + buf_pos/2, N, BL_window);
		math::normalize_std(conv_t_space + buf_pos/2, N, BL_window);
		matched_filter(conv_t_space + buf_pos/2, N, threshold, pulses_dm1, power_dm1);
		


		/*
		// Debug output
		if (n_found == 0)
		{
			std::ofstream test0(output_dir + "conv.bin");
			test0.write((char*) (conv_t_space + n_DM), N*sizeof(double));
			test0.close();

			std::ofstream test1(output_dir + "sum.bin");
			test1.write((char*) (sum_dm1 + n_DM), N*sizeof(double));
			test1.close();
		}
		*/


		// Save profiles of the search
		is_pulse = pulses_dm1.size() > 0;
		//is_pulse = true;

		if (is_pulse)
			n_found ++;

		hdr->t0 = t0 + reader->point2time(sumidx) / 86400.0L;

		// Save info of the search
		for(size_t i = 0; i < power_dm1.size(); ++i)
			csv << csv_result(pulses_dm1[2*i], pulses_dm1[2*i+1], power_dm1[i], i) << '\n';


		if (save_raw && is_pulse) 
		{
			PSRFITS_Writer writer1(output_dir + "raw_" + reader->filename + "_" + std::to_string(n_found));
			writer1.createPrimaryHDU("SEARCH", hdr);
			writer1.append_subint_search(
					buff + buf_pos, 
					hdr->freqs, nullptr,
					2*N, 1, 1, 
					0.0, fcomp, tau/2.0, "");
		}

		if (save_dyn && is_pulse) 
		{
			PSRFITS_Writer writer1(output_dir + "dyn_" + reader->filename + "_" + std::to_string(n_found));
			writer1.createPrimaryHDU("SEARCH", hdr);
			writer1.append_subint_search(
					(double*) (t_space + buf_pos/2), 
					new double((fmin+fmax)/2.0), nullptr,
					N, 1, 1, 
					DM, fcomp, tau, "coherent", true);
		}

		if (save_sum && is_pulse) 
		{

			PSRFITS_Writer writer0(output_dir + "conv_" + reader->filename + "_" + std::to_string(n_found));
			writer0.createPrimaryHDU("SEARCH", hdr);
			writer0.append_subint_search(
					conv_t_space + buf_pos/2, 
					new double((fmin+fmax)/2.0), nullptr,
					N, 1, 1, 
					DM, fcomp, tau, "coherent");

			PSRFITS_Writer writer1(output_dir + "sum_" + reader->filename + "_" + std::to_string(n_found));
			writer1.createPrimaryHDU("SEARCH", hdr);
			writer1.append_subint_search(
					sum_dm1 + buf_pos/2, 
					new double((fmin+fmax)/2.0), nullptr,
					N, 1, 1, 
					DM, fcomp, tau, "coherent");
		}

		buf_pos = buf_max - 2*n_DM;
		sumidx += N;



		if (verbose > 0)
		{
			std::cout << "\x1b[2K\r" ;
			std::cout << "t = " << reader->point2time(sumidx) << " s";
			std::cout << std::flush;
		}

	}

	if (verbose > 0)
	{
		std::cout << std::endl;
		std::cout << "Found " << n_found << " windows" << std::endl;
	}


	hdr->t0 = t0;

	if (ker_f)
		delete[] ker_f;

	delete[] buff;
	delete[] sum_dm1;
	delete[] t_space;
	delete[] f_space;

	if (size_t(csv.tellp()) > empty_size)
	{
		csv.close();
		return id;
	}
	else 
	{
		csv.close();
		std::remove((output_dir + id).c_str());
		return "";
	}
}

std::string Profile::dedisperse_incoherent_search(
		double DM, 
		size_t nchann, 
		double  BL_window_s, 
		double threshold,
		std::string conv_type, 
		double fwhm
		)
{

	check_incoherent(nchann);

	size_t BL_window;
	size_t obs_window;
	double tau;
	size_t n_DM;
	double fcomp, fmin, fmax;
	size_t buf_pos, buf_max;

	int *shift = nullptr;
	double *pre = nullptr, *post = nullptr;


	size_t ker_len = 0;
	double* ker_t = nullptr;
	double *conv  = nullptr;


	long double t0;

	fmin  = hdr->fmin;
	fmax  = hdr->fmax;
	fcomp = hdr->fcomp;
	tau   = hdr->tau;
	t0    = hdr->t0;

	double dtmax = 4.15e6 * DM * std::abs(1/fmin/fmin - 1/fmax/fmax);
	n_DM = static_cast<size_t>(dtmax/tau);
	n_DM += n_DM % 2;

	// set 256 MiB buffer as standard size
	obs_window = std::max(n_DM, (256ul << 20)/nchann/hdr->npol/sizeof(double)); 
	obs_window = std::min(obs_window, hdr->OBS_SIZE);

	obs_window += n_DM;

	ker_len = size_t((5.0e3*fwhm)/tau) + 1;

	shift = new int[nchann];
	pre = new double[obs_window*nchann];
	post = new double[obs_window*nchann];
	sum = new double[obs_window - n_DM];

	conv = new double[obs_window - n_DM + ker_len - 1];
	// Set up the convolution kernel
	if(conv_type == "gaussian")
	{

		ker_t = new double[nchann];
		std::fill(ker_t, ker_t + nchann, 0);
		math::gaussian_kernel(ker_t, ker_len, fwhm*1e3/tau);
	}
	else {
		throw ;
	}


	calc_shift_int(shift, DM, fcomp, hdr->freqs, nchann, tau, redshift);
	



	std::srand(time(NULL));
	std::string id = "";
	id += std::to_string(std::rand());
	id += ".csv";
	std::ofstream csv(output_dir + id);

	// write table header
	csv << "file; num; MJD; sec; width; power;" << std::endl;
	size_t empty_size =  csv.tellp();

	buf_pos = 0;
	buf_max = 0;
	sumidx = 0;

	std::vector<size_t> pulses_dm1;
	std::vector<double> power_dm1;
	size_t N = 0;
	bool eof = false;
	bool is_pulse = false;
	size_t n_found = 0;

	BL_window = size_t(BL_window_s/hdr->tau * 1.0e3);
	size_t zeroth_lag = (ker_len - 1) / 2;

	while(!eof)
	{
		if (buf_pos + obs_window >= buf_max)
			fill_2d(pre, nchann, buf_pos, buf_max, obs_window);

		if (buf_max < obs_window)
		{
			eof = true; // EOF is reached

			// zero padding to save the last part of the file
			std::fill(pre + buf_max*nchann, pre + obs_window*nchann, 0.0);
		}


		shift_window_incoherent(pre, post, shift, nchann, obs_window, mask);
		
		N = buf_max - n_DM;

		for (size_t t = 0; t < N; ++t) 
			sum[t] = std::accumulate(post + t*nchann, post + (t+1)*nchann, 0.0);


		if(conv_type == "gaussian")
			math::ccf(sum, ker_t, N, ker_len, conv);
		else
			throw;



		// Find pulses with overlapping windows
		math::subtract_baseline(conv + zeroth_lag, N, BL_window);
		math::normalize_std(conv + zeroth_lag, N, BL_window);
		matched_filter(conv + zeroth_lag, N, threshold, pulses_dm1, power_dm1);
		


		
		/*
		// Debug output
		if (n_found == 0)
		{

			std::ofstream test2(output_dir + "ker.bin");
			test2.write((char*) (ker_t), ker_len*sizeof(double));
			test2.close();

			std::ofstream test0(output_dir + "conv.bin");
			test0.write((char*) (conv + zeroth_lag), N*sizeof(double));
			test0.close();

			std::ofstream test1(output_dir + "sum.bin");
			test1.write((char*) (sum), N*sizeof(double));
			test1.close();
		}
		*/
		

		// Save profiles of the search
		is_pulse = pulses_dm1.size() > 0;
		//is_pulse = true;

		if (is_pulse)
			n_found ++;

		hdr->t0 = t0 + reader->point2time(sumidx) / 86400.0L;

		// Save info of the search
		for(size_t i = 0; i < power_dm1.size(); ++i)
			csv << csv_result(pulses_dm1[2*i], pulses_dm1[2*i+1], power_dm1[i], i) << '\n';


		/*
		if (save_raw && is_pulse) 
		{
			PSRFITS_Writer writer1(output_dir + "raw_" + reader->filename + "_" + std::to_string(n_found));
			writer1.createPrimaryHDU("SEARCH", hdr);
			writer1.append_subint_search(
					pre, 
					hdr->freqs, nullptr,
					N, nchann, 1, 
					0.0, fcomp, tau, "");
		}

		if (save_dyn && is_pulse) 
		{
			PSRFITS_Writer writer1(output_dir + "dyn_" + reader->filename + "_" + std::to_string(n_found));
			writer1.createPrimaryHDU("SEARCH", hdr);
			writer1.append_subint_search(
					(double*) post, 
					new double((fmin+fmax)/2.0), nullptr,
					N, nchann, 1, 
					DM, fcomp, tau, "coherent", true);
		}

		if (save_sum && is_pulse) 
		{

			PSRFITS_Writer writer0(output_dir + "conv_" + reader->filename + "_" + std::to_string(n_found));
			writer0.createPrimaryHDU("SEARCH", hdr);
			writer0.append_subint_search(
					conv + ker_len-1, 
					new double((fmin+fmax)/2.0), nullptr,
					N, 1, 1, 
					DM, fcomp, tau, "coherent");

			PSRFITS_Writer writer1(output_dir + "sum_" + reader->filename + "_" + std::to_string(n_found));
			writer1.createPrimaryHDU("SEARCH", hdr);
			writer1.append_subint_search(
					sum, 
					new double((fmin+fmax)/2.0), nullptr,
					N, 1, 1, 
					DM, fcomp, tau, "coherent");
		}
		*/

		buf_pos = buf_max - n_DM;
		sumidx += N;



		if (verbose > 0)
		{
			std::cout << "\x1b[2K\r" ;
			std::cout << "t = " << reader->point2time(sumidx) << " s";
			std::cout << std::flush;
		}

	}

	if (verbose > 0)
	{
		std::cout << std::endl;
		std::cout << "Found " << n_found << " windows" << std::endl;
	}


	hdr->t0 = t0;


	delete[] shift; shift = nullptr;
	delete[] pre;   pre   = nullptr; 
	delete[] post;  post  = nullptr;
	delete[] sum;   sum   = nullptr;
	delete[] conv;  conv  = nullptr;
	delete[] ker_t; ker_t = nullptr;

	if (size_t(csv.tellp()) > empty_size)
	{
		csv.close();
		return id;
	}
	else 
	{
		csv.close();
		std::remove((output_dir + id).c_str());
		return "";
	}
}
