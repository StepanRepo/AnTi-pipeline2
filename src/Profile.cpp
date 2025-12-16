#include "Profile.h"
#include "PRAO_adc.h"   // full definition of PRAO_adc
#include "IAA_vdif.h"   // full definition of IAA_vdif
#include "aux_math.h"
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <numeric>

#include <ctime>

#include "tempo2.h"
#include"tempo2pred.h"  // API for TEMPO2 prediction files


# define M_PI           3.14159265358979323846
# define C           299792.458

void calc_shift_int(int *shift, double DM, double fcomp, double fmin, double fmax, size_t nchann, double tau, double beta = 0.0)
{

	double *freqs = nullptr;
	double *dt = nullptr;
	double df = (fmax - fmin)  / nchann;

	freqs = new double[nchann];
	dt    = new double[nchann];

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		freqs[i] = (fmin + df * (static_cast<double>(i) + .5)) * (1.0 + beta);

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		dt[i] = 4.148808e6 * DM * (1.0/freqs[i]/freqs[i] - 1.0/fcomp/fcomp);

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i) 
		shift[i] = static_cast<int> (dt[i] / tau + 0.5);

	delete[] freqs;
	delete[] dt;
}

void calc_shift_phase(fftw_complex *dphase, double DM, double fcomp, double fmin, double fmax, size_t nchann, double beta = 0.0)
{
	double *freqs = nullptr;
	double df = (fmax - fmin)  / nchann;
	double sign = fmin > fmax ? 1.0 : -1.0;
	double phase, phase0;

	freqs = new double[nchann];

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		freqs[i] = df * (static_cast<double>(i) + .5);

	fcomp = (1+beta)*fcomp;
	fmin  = (1+beta)*fmin;


	phase0 = sign * 2.0e3 * M_PI * 4.148808e6 * DM * 
		std::pow(fcomp-fmin, 2) /(fmin * fmin * fcomp);

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
	{
		phase = sign * 2.0e3 * M_PI * 4.148808e6 * DM * freqs[i] * freqs[i] /
			(fmin * fmin * (fmin + freqs[i])) - phase0;

		dphase[i][0] = std::cos(phase);
		dphase[i][1] = std::sin(phase);
	}

	delete[] freqs;
}

void shift_window_incoherent(const double* in, double* out, const int* shift, const size_t nchann, const size_t obs_window, const double* mask = nullptr)
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
 	clock_t t0, t1;;
	t0 = std::clock();

	fftw_execute(fft);
		t1 = std::clock();
		std::cout << float(t1 - t0) / CLOCKS_PER_SEC << std::endl;
		t0 = t1;


	math::vec_prod(f_space, dphase, nchann);
		t1 = std::clock();
		std::cout << float(t1 - t0) / CLOCKS_PER_SEC << std::endl;
		t0 = t1;
	fftw_execute(ifft);
		t1 = std::clock();
		std::cout << float(t1 - t0) / CLOCKS_PER_SEC << std::endl;
		t0 = t1;
}

void detect(fftw_complex* t_space, double* sum, size_t nchann)
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
		power[i] = math::mean(data + a, b-a);
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

std::string Profile::csv_result (size_t left, size_t right, double power) const
{
	std::ostringstream row("");
	row << std::fixed << std::setprecision(19); // f5.14 (down to ns in MJD)  

	double mjd_left  = hdr->t0 + reader->point2time(sumidx + left)/86400.0;
	double mjd_right = hdr->t0 + reader->point2time(sumidx + right)/86400.0;

	row << mjd_left << "; ";
	row << mjd_right << "; ";
	row << power << "; ";

	return row.str();
}





Profile::Profile(
		const std::string& filename, 
		const std::string& format, 
		size_t buffer_size,
		bool save_raw_in, bool save_dyn_in, bool save_sum_in,
		std::string output_dir_in)
{
    if (format == "PRAO_adc") 
	{
        reader = new PRAO_adc(filename, buffer_size);
    } 
	else if (format == "IAA_vdif") 
	{
        reader = new IAA_vdif(filename, buffer_size);
    }
   	else 
	{
        throw std::invalid_argument("Unsupported format: " + format);
    }

    if (!reader || !reader->is_open) 
	{
        throw std::runtime_error("Reader not initialized or file not open");
	}

	raw = nullptr;
	dyn = nullptr;
	sum = nullptr;
	fr = nullptr;
	mask = nullptr;

	save_raw = save_raw_in;
	save_dyn = save_dyn_in;
	save_sum = save_sum_in;
	output_dir = output_dir_in;

	hdr = reader->header_ptr;

	redshift = 0.0;
	sumidx = 0;
}

size_t Profile::fill_2d(double* data, size_t& nchann, size_t& buf_pos, size_t& buf_max, size_t& buf_size) 
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


	if (hdr->nchann == 1)
	{
		hdr->nchann = nchann;
		hdr-> tau = 2.0e-3 * nchann / hdr->sampling;
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
	if (hdr->nchann != 1)
		throw std::runtime_error("Coherent dedispersion is unavailable for this file");
	if (sum != nullptr)
		throw std::runtime_error("The file already contains frequency averaged data");

	if (hdr->fcomp == 0.0)
	{
		hdr->fcomp = std::max(hdr->fmin, hdr->fmax);
	}

	if (hdr-> nchann == 1)
		hdr-> tau = 2.0e-3 / hdr->sampling;
}

void Profile::dedisperse_incoherent(double DM, size_t nchann)
{
	check_incoherent(nchann);

	size_t obs_window;
	double tau;

	double fcomp, fmin, fmax;
	int *shift;

	fmin = hdr->fmin;
	fmax = hdr->fmax;
	fcomp = hdr->fcomp;
	tau = hdr->tau;
	nchann = hdr->nchann;
	obs_window = hdr->obs_window;




	shift = new int[nchann];
	dyn = new double[obs_window*nchann];
	sum = new double[obs_window];


	calc_shift_int(shift, DM, fcomp, fmin, fmax, nchann, tau, redshift);
	shift_window_incoherent(raw, dyn, shift, nchann, obs_window, mask);


	for (size_t t = 0; t < obs_window; ++t) 
		sum[t] = std::accumulate(dyn + t*nchann, dyn + (t+1)*nchann, 0.0);


	
	if (save_dyn)
	{
		PSRFITS_Writer writer(output_dir + "dyn_" + reader->filename);
		writer.createPrimaryHDU("PSR", hdr);
		writer.append_subint_fold(
				dyn, nullptr, 
				obs_window, nchann, 1, 
				hdr->period, DM, fmin, fmax, tau);
	}

	if (save_sum)
	{
		PSRFITS_Writer writer(output_dir + "sum_" + reader->filename);
		writer.createPrimaryHDU("PSR", hdr);
		writer.append_subint_fold(
				sum, nullptr, 
				obs_window, 1, 1, 
				hdr->period, DM, fmin, fmax, tau);
	}

	delete[] shift;
}

std::string Profile::dedisperse_incoherent_stream(double DM, size_t nchann)
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

	if (save_raw)
		raw_output = std::ofstream(output_dir + "raw_" + id);
	if (save_dyn)
		dyn_output = std::ofstream(output_dir + "dyn_" + id);
	if (save_sum)
		sum_output = std::ofstream(output_dir + "sum_" + id);


	fmin = hdr->fmin;
	fmax = hdr->fmax;
	fcomp = hdr->fcomp;
	tau = hdr->tau;


	double dtmax = 4.15e6 * DM * std::abs(1/fmin/fmin - 1/fmax/fmax);
	n_DM = static_cast<size_t>(dtmax / tau) + 1;
	n_DM += n_DM % 2;

	// set 256 MiB buffer as standard size
	obs_window = std::max(n_DM, (256ul << 20)/nchann/sizeof(double)); 
	obs_window += n_DM;


	shift = new int[nchann];
	pre = new double[obs_window*nchann];
	post = new double[obs_window*nchann];
	sum = new double[obs_window - n_DM];

	calc_shift_int(shift, DM, fcomp, fmin, fmax, nchann, tau, redshift);

	buf_pos = 0;
	buf_max = 0;
	sumidx = 0;

	while(true)
	{
		if (buf_pos + obs_window >= buf_max)
			fill_2d(pre, nchann, buf_pos, buf_max, obs_window);

		if (buf_max < obs_window)
			break; // EOF is reached
				   //
	
		shift_window_incoherent(pre, post, shift, nchann, obs_window, mask);


		for (size_t t = 0; t < obs_window - n_DM; ++t) 
			sum[t] = std::accumulate(post + t*nchann, post + (t+1)*nchann, 0.0);

		// save processed buffer
		// regecting last n_DM inputs

		if (save_raw)
			raw_output.write(reinterpret_cast<const char*>(pre),
					nchann * (buf_max - n_DM) * sizeof(double));

		if (save_dyn)
			dyn_output.write(reinterpret_cast<const char*>(post),
					nchann * (buf_max - n_DM) * sizeof(double));

		if (save_sum)
			sum_output.write(reinterpret_cast<const char*>(sum),
					(buf_max - n_DM) * sizeof(double));

		buf_pos = buf_max - n_DM;
		sumidx += buf_pos;
		std::cout << "t = " << reader->point2time(sumidx) << " ms" << std::endl;
	}

	// zero padding to save the last part of the file
	std::fill(pre + buf_max*nchann, pre + obs_window*nchann, 0.0);
	shift_window_incoherent(pre, post, shift, nchann, obs_window, mask);


	for (size_t t = 0; t < obs_window - n_DM; ++t) 
		sum[t] = std::accumulate(post + t*nchann, post + (t+1)*nchann, 0.0);

	buf_pos = buf_max - n_DM;
	sumidx += buf_pos;
	std::cout << "t = " << reader->point2time(sumidx) << " s" << std::endl;


	if (save_raw)
		raw_output.write(reinterpret_cast<const char*>(pre),
				nchann * (buf_max - n_DM) * sizeof(double));

	if (save_dyn)
		dyn_output.write(reinterpret_cast<const char*>(post),
				nchann * (buf_max - n_DM) * sizeof(double));

	if (save_sum)
		sum_output.write(reinterpret_cast<const char*>(sum),
				(buf_max - n_DM) * sizeof(double));



	// Convert save buffers into readable psrfits files
	if (save_raw)
	{
		raw_output.close();

		PSRFITS_Writer writer(output_dir + "raw_" + reader->filename);
		writer.createPrimaryHDU("SEARCH", hdr);
		writer.append_subint_stream(
				output_dir + "raw_" + id, mask, 
				nchann, 1, 
				DM, fmin, fmax, tau);
	}

	if (save_dyn)
	{
		dyn_output.close();

		PSRFITS_Writer writer(output_dir + "dyn_" + reader->filename);
		writer.createPrimaryHDU("SEARCH", hdr);
		writer.append_subint_stream(
				output_dir + "dyn_" + id, nullptr, 
				nchann, 1, 
				DM, fmin, fmax, tau);
	}

	if (save_sum)
	{
		sum_output.close();

		PSRFITS_Writer writer(output_dir + "sum_" + reader->filename);
		writer.createPrimaryHDU("SEARCH", hdr);
		writer.append_subint_stream(
				output_dir + "sum_" + id, nullptr, 
				1, 1, 
				DM, fmin, fmax, tau);
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

std::string Profile::dedisperse_incoherent_search(double DM, size_t nchann, double  BL_window_s, double threshold)
{

	check_incoherent(nchann);

	size_t obs_window, BL_window;
	double tau;
	size_t n_DM;
	double fcomp, fmin, fmax;
	size_t buf_pos, buf_max;

	bool is_pulse = false;
	bool eof = false;

	int *shift = nullptr;
	double *pre = nullptr, *post = nullptr;
	double *sum_dm0 = nullptr, *sum_dm1 = nullptr;


	std::srand(time(NULL));
	std::string id = "";
	id += std::to_string(std::rand());
	id += ".csv";
	std::ofstream csv(output_dir + id);

	// write table header
	csv << "# ===== Search Results =====" << std::endl;
	csv << "# left; right; power; dm0_lefts; dm0_rights; dm0_powers;" << std::endl;
	size_t empty_size =  csv.tellp();


	fmin = hdr->fmin;
	fmax = hdr->fmax;
	fcomp = hdr->fcomp;
	tau = hdr->tau;


	double dtmax = 4.15e6 * DM * std::abs(1/fmin/fmin - 1/fmax/fmax);
	n_DM = static_cast<size_t>(dtmax / tau) + 1;
	n_DM += n_DM % 2;

	// set 256 MiB buffer as standard size
	obs_window = std::max(n_DM, (256ul << 20)/nchann/sizeof(double)); 
	obs_window += n_DM;

	shift = new int[nchann];
	pre = new double[obs_window*nchann];
	post = new double[obs_window*nchann];
	sum_dm1 = new double[obs_window - n_DM];
	sum_dm0 = new double[obs_window - n_DM];

	calc_shift_int(shift, DM, fcomp, fmin, fmax, nchann, tau, redshift);

	buf_pos = 0;
	buf_max = 0;
	sumidx = 0;

	std::vector<size_t> pulses_dm0, pulses_dm1;
	std::vector<double> power_dm0, power_dm1;
	size_t N = 0;

	BL_window = size_t(BL_window_s/hdr->tau * 1.0e3);
	size_t n_found = 0;

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

		N = buf_max - n_DM;
		shift_window_incoherent(pre, post, shift, nchann, obs_window, mask);

		for (size_t t = 0; t < N; ++t) 
		{
			if (mask)
			{
				math::vec_prod(pre+t*nchann, mask, nchann);
			}
			sum_dm0[t] = math::mean(pre + t*nchann, nchann);
		}

		for (size_t t = 0; t < N; ++t) 
			sum_dm1[t] = math::mean(post + t*nchann, nchann);



		math::subtract_baseline(sum_dm0, N, BL_window);
		math::subtract_baseline(sum_dm1, N, BL_window);

		
		std::ofstream test0(output_dir + "test0.bin");
		test0.write((char*) sum_dm0, N*sizeof(double));
		test0.close();
		std::ofstream test1(output_dir + "test1.bin");
		test1.write((char*) sum_dm1, N*sizeof(double));
		test1.close();
		


		matched_filter(sum_dm0, N, threshold, pulses_dm0, power_dm0);
		matched_filter(sum_dm1, N, threshold, pulses_dm1, power_dm1);

		// save info of the search
		for(size_t i = 0; i < power_dm1.size(); ++i)
			csv << csv_result(pulses_dm1[2*i], pulses_dm1[2*i+1], power_dm1[i]) << '\n';


		// save profiles of the search
		is_pulse = pulses_dm1.size() > 0;
		is_pulse = true;

		if (is_pulse)
			n_found ++;


		if (save_raw && is_pulse) 
		{
			PSRFITS_Writer writer1(output_dir + "raw_" + reader->filename + "_" + std::to_string(n_found));
			writer1.createPrimaryHDU("SEARCH", hdr);
			writer1.append_subint_search(
					pre, nullptr,
					buf_max - n_DM, nchann, 1, 
					DM, fmin, fmax, tau);
		}

		if (save_dyn && is_pulse) 
		{
			PSRFITS_Writer writer1(output_dir + "dyn_" + reader->filename + "_" + std::to_string(n_found));
			writer1.createPrimaryHDU("SEARCH", hdr);
			writer1.append_subint_search(
					post, nullptr,
					buf_max - n_DM, nchann, 1, 
					DM, fmin, fmax, tau);
		}

		if (save_sum && is_pulse) 
		{
			PSRFITS_Writer writer0(output_dir + "sum0_" + reader->filename + "_" + std::to_string(n_found));
			writer0.createPrimaryHDU("SEARCH", hdr);
			writer0.append_subint_search(
					sum_dm0, nullptr,
					buf_max - n_DM, 1, 1, 
					0, fmin, fmax, tau);

			PSRFITS_Writer writer1(output_dir + "sum1_" + reader->filename + "_" + std::to_string(n_found));
			writer1.createPrimaryHDU("SEARCH", hdr);
			writer1.append_subint_search(
					sum_dm1, nullptr,
					buf_max - n_DM, 1, 1, 
					DM, fmin, fmax, tau);
		}

		buf_pos = buf_max - n_DM;
		sumidx += buf_pos;
		std::cout << "t = " << reader->point2time(sumidx) << " ms" << std::endl;
	}


	delete[] shift;
	delete[] pre;
	delete[] post;
	delete[] sum_dm1;
	delete[] sum_dm0;

	if (size_t(csv.tellp()) > empty_size)
	{
		csv.close();
		return id;
	}
	else 
	{
		csv.close();
		std::remove(id.c_str());
		return "";
	}
}


void Profile::dedisperse_coherent(double DM, size_t nchann)
{
		throw std::runtime_error("The function development is in progress");
		return;
}

std::string Profile::dedisperse_coherent_stream(double DM, size_t nchann)
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
	fftw_plan p;
	fftw_complex *f_small, *t_small;
	size_t freq_num = 2048;
	double *spec;
	f_small = (fftw_complex*)(fftw_malloc(sizeof(fftw_complex) * (freq_num)));
	t_small = (fftw_complex*)(fftw_malloc(sizeof(fftw_complex) * (freq_num)));
	spec = (double*)(fftw_malloc(sizeof(double) * (freq_num)));
	p  = fftw_plan_dft_1d(freq_num, t_small, f_small, FFTW_FORWARD, FFTW_ESTIMATE);
	std::ofstream output(output_dir + "test.bin");

	fmin = hdr->fmin;
	fmax = hdr->fmax;
	fcomp = hdr->fcomp;
	tau = hdr->tau;


	double dtmax = 4.15e6 * DM * (std::pow(std::min(fmax, fmin), -2) - std::pow(std::max(fmax, fmin), -2));
	n_DM = static_cast<size_t>(dtmax/tau);
	n_DM += n_DM % 2;
	
	if (nchann <= n_DM)
        throw std::runtime_error("The number of channels is too small for coherent dedispersion. Set at least 2^" + std::to_string(size_t(std::log2(n_DM)) + 1));
	obs_window = 2*nchann;


	dphase = (fftw_complex*)(fftw_malloc(sizeof(fftw_complex) * (nchann)));
	calc_shift_phase(dphase, DM, fcomp, fmin, fmax, nchann, redshift);

	if (mask)
		math::vec_prod(dphase, mask, nchann);


	buff    = (double*) (fftw_malloc(sizeof(double) * obs_window));
	sum     = (double*) (fftw_malloc(sizeof(double) * nchann));
	t_space = (fftw_complex*) (fftw_malloc(sizeof(fftw_complex) * (nchann)));
	f_space = (fftw_complex*) (fftw_malloc(sizeof(fftw_complex) * (nchann+1)));

	fft  = fftw_plan_dft_r2c_1d(obs_window, buff, f_space, FFTW_ESTIMATE);
	ifft = fftw_plan_dft_1d(nchann, f_space+1, t_space, FFTW_BACKWARD, FFTW_ESTIMATE);


	//std::srand(time(NULL));
	std::string id = "";
	id += std::to_string(std::rand());
	id += ".bin";

	if (save_raw)
		raw_output = std::ofstream(output_dir + "raw_" + id);

	if (save_dyn)
		dyn_output = std::ofstream(output_dir + "dyn_" + id);

	if (save_sum)
		sum_output = std::ofstream(output_dir + "sum_" + id);

	std::fill(buff, buff + 2*n_DM, 0.0);
	buf_max = 2*n_DM; 
	buf_pos = 0;
	sumidx = 0;


	bool eof = false;
	while(true && !eof)
	{
		if (buf_pos + obs_window >= buf_max)
			fill_1d(buff, buf_pos, buf_max, obs_window);

		if (buf_max < obs_window)
		{
			eof = true; // EOF is reached
			std::fill(buff + buf_max, buff + obs_window, 0.0);
		}

		shift_window_coherent(fft, ifft, f_space+1, dphase, nchann);
		detect(t_space, sum, nchann);


		if (save_raw)
			raw_output.write(reinterpret_cast<const char*>(buff + 2*n_DM),
					(buf_max - 2*n_DM) * sizeof(double));

		if (save_dyn)
			dyn_output.write(reinterpret_cast<const char*>(t_space + n_DM),
					(buf_max/2 - n_DM) * sizeof(fftw_complex));

		if (save_sum)
			sum_output.write(reinterpret_cast<const char*>(sum + n_DM),
					(buf_max/2 - n_DM) * sizeof(double));

		buf_pos = buf_max - 2*n_DM;
		sumidx += buf_pos/2;
		std::cout << "t = " << reader->point2time(sumidx) << " ms" << std::endl;

		/******************************************
		 * This part allows to plot output spectrum
		 * for the processed chunk. It is used for debugging
		 * (comment the output above and uncomment the section 
		 * below to use it)
		 ******************************************
		 */
		double re, im;
		for (size_t i = n_DM/freq_num; i < buf_max/freq_num/2; ++i)
		{
		#pragma omp simd
			for (size_t k = 0; k < freq_num; ++k)
			{
				t_small[k][0] = t_space[i*freq_num + k][0];
				t_small[k][1] = t_space[i*freq_num + k][1];
			}

			fftw_execute(p);

		#pragma omp simd
			for (size_t k = 0; k < freq_num; ++k)
			{
				re = f_small[k][0];
				im = f_small[k][1];

				spec[k] = re*re + im*im;
			}

			output.write(reinterpret_cast<const char*>(spec),
					(freq_num) * sizeof(double));
		}
	}


	if (save_raw)
	{
		raw_output.close();

		PSRFITS_Writer writer(output_dir + "raw_" + reader->filename);
		writer.createPrimaryHDU("SEARCH", hdr);
		writer.append_subint_stream(
				output_dir + "raw_" + id, nullptr, 
				1, 1, 
				DM, fmin, fmax, tau/2.0);
	}

	if (save_dyn)
	{
		dyn_output.close();

		PSRFITS_Writer writer(output_dir + "dyn_" + reader->filename);
		writer.createPrimaryHDU("SEARCH", hdr);
		writer.append_subint_stream(
				output_dir + "dyn_" + id, nullptr, 
				2, 1, 
				DM, fmin, fmax, tau, true);
	}

	if (save_sum)
	{
		sum_output.close();

		PSRFITS_Writer writer(output_dir + "sum_" + reader->filename);
		writer.createPrimaryHDU("SEARCH", hdr);
		writer.append_subint_stream(
				output_dir + "sum_" + id, nullptr, 
				1, 1, 
				DM, fmin, fmax, tau);
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

	fftw_complex* dphase;
	fftw_complex *f_space, *t_space;
	fftw_complex *conv_f_space;
	fftw_complex *ker_f = nullptr;
	double *sum_dm1, *conv_t_space;
	fftw_plan fft, ifft;
	fftw_plan conv_fft, conv_ifft;

	fmin = hdr->fmin;
	fmax = hdr->fmax;
	fcomp = hdr->fcomp;
	tau = hdr->tau;

	double dtmax = 4.15e6 * DM * (std::pow(std::min(fmax, fmin), -2) - std::pow(std::max(fmax, fmin), -2));
	n_DM = static_cast<size_t>(dtmax/tau);
	n_DM += n_DM % 2;
	
	if (nchann <= n_DM)
        throw std::runtime_error("The number of channels is too small for coherent dedispersion. Set at least 2^" + std::to_string(size_t(std::log2(n_DM)) + 1));
	obs_window = 2*nchann;


	dphase  = (fftw_complex*)(fftw_malloc(sizeof(fftw_complex) * (nchann)));
	calc_shift_phase(dphase, DM, fcomp, fmin, fmax, nchann, redshift);


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
	csv << "# ===== Search Results =====" << std::endl;
	csv << "# left; right; power;" << std::endl;
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


	// Pad the beginning of the buffer with zeros
	// to preserve the beginning of the file
	std::fill(buff, buff + 2*n_DM, 0.0);
	buf_max = 2*n_DM;

	while(true && !eof)
	{

		if (buf_pos + obs_window >= buf_max)
			fill_1d(buff, buf_pos, buf_max, obs_window);

		if (buf_max < obs_window)
		{
			eof = true; // EOF is reached
			std::fill(buff + buf_max, buff + obs_window, 0.0);
		}


		N = buf_max/2 - n_DM;
		shift_window_coherent(fft, ifft, f_space+1, dphase, nchann);
		detect(t_space, sum_dm1, nchann);

		if (ker_f)
		{
			shift_window_coherent(conv_fft, conv_ifft, conv_f_space+1, ker_f, nchann);
		}
		else if (conv_type == "box")
		{
			std::cout << "box conv: " << fwhm*1e3/tau << std::endl;
			math::box_conv(sum_dm1, conv_t_space, size_t(fwhm * 1e3/tau + .5),  nchann);
		}
		else
		{
			conv_t_space = sum_dm1;
		}

		//math::subtract_baseline(conv_t_space+n_DM, N, BL_window);
		matched_filter(conv_t_space+n_DM, N, threshold, pulses_dm1, power_dm1);


		if (n_found == 0)
		{
			std::ofstream test1(output_dir + "test1.bin");
			test1.write((char*) (conv_t_space + n_DM), N*sizeof(double));
			test1.close();
		}

		// Save info of the search
		//for(size_t i = 0; i < power_dm1.size(); ++i)
		//	csv << csv_result(pulses_dm1[2*i], pulses_dm1[2*i+1], power_dm1[i]) << '\n';

		// Save profiles of the search
		is_pulse = pulses_dm1.size() > 0;
		is_pulse = true;

		if (is_pulse)
			n_found ++;

		if (save_raw && is_pulse) 
		{
			PSRFITS_Writer writer1(output_dir + "raw_" + reader->filename + "_" + std::to_string(n_found));
			writer1.createPrimaryHDU("SEARCH", hdr);
			writer1.append_subint_search(
					buff + 2*n_DM, nullptr,
					2*N, 1, 1, 
					DM, fmin, fmax, tau);
		}

		if (save_dyn && is_pulse) 
		{
			PSRFITS_Writer writer1(output_dir + "dyn_" + reader->filename + "_" + std::to_string(n_found));
			writer1.createPrimaryHDU("SEARCH", hdr);
			writer1.append_subint_search(
					(double*) (t_space + n_DM), nullptr,
					N, 2, 1, 
					DM, fmin, fmax, tau, true);
		}

		if (save_sum && is_pulse) 
		{

			PSRFITS_Writer writer0(output_dir + "conv1_" + reader->filename + "_" + std::to_string(n_found));
			writer0.createPrimaryHDU("SEARCH", hdr);
			writer0.append_subint_search(
					conv_t_space + n_DM, nullptr,
					N, 1, 1, 
					DM, fmin, fmax, tau);

			PSRFITS_Writer writer1(output_dir + "sum1_" + reader->filename + "_" + std::to_string(n_found));
			writer1.createPrimaryHDU("SEARCH", hdr);
			writer1.append_subint_search(
					sum_dm1 + n_DM, nullptr,
					N, 1, 1, 
					DM, fmin, fmax, tau);
		}

		buf_pos = buf_max - 2*n_DM;
		sumidx += buf_pos/2;
		std::cout << "t = " << reader->point2time(sumidx) << " s" << std::endl;

	}

	std::cout << "Found " << n_found << " windows" << std::endl;

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

	hdr->total_pulses = rev;

	#pragma omp simd
	for(size_t i = 0; i < obs_window*nchann; ++i)
		raw[i] = raw[i] / double(rev);

	if (save_raw)
	{
		PSRFITS_Writer writer(output_dir + "raw_" + reader->filename);
		writer.createPrimaryHDU("PSR", hdr);
		writer.append_subint_fold(
				raw, mask, 
				obs_window, nchann, 1, 
				hdr->period, 0.0, hdr->fmin, hdr->fmax, tau);
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

	hdr->total_pulses = rev;

	#pragma omp simd
	for(size_t i = 0; i < obs_window*nchann; ++i)
		raw[i] = raw[i] / double(rev);

	if (save_raw)
	{
		PSRFITS_Writer writer(output_dir + "raw_" + reader->filename);
		writer.createPrimaryHDU("PSR", hdr);
		writer.append_subint_fold(
				raw, mask, 
				obs_window, nchann, 1, 
				hdr->period, 0.0, hdr->fmin, hdr->fmax, tau);
	}

	delete[] buff;
	buff = nullptr;
}

double Profile::get_redshift (std::string par_path, std::string site)
{
	// reading pulsar parameters
    // Initialize pulsar and observation
    pulsar psr;
    initialiseOne(&psr, 1, 0); // minimal init, with warnings enabled
    psr.nobs = 1;
    allocateMemory(&psr, 0);
    observation* obs = &psr.obsn[0];	

	char t2_path[500];
	strncpy(t2_path, par_path.c_str(), par_path.length());
	t2_path[par_path.length()] = '\0';
	readParfile(&psr, &t2_path, nullptr, 1); /* Read .par file to define the pulsar's initial parameters */  

    // Set site arrival time and observatory
	const char* obs_code = site.c_str();
    obs->sat = hdr->t0;
    strcpy(obs->telID, obs_code);

	psr.t2cMethod = T2C_TEMPO;
    obs->clockCorr = 1;	
    obs->delayCorr = 1;	



    readEphemeris(&psr, 1, 0);	// fill Earth SSB posvel
	get_obsCoord(&psr, 1);		// fill siteVel
	vectorPulsar(&psr, 1);	// fill pulsar position
							


	double v_total[3];
	for (int i = 0; i < 3; ++i)
		v_total[i] = obs->earth_ssb[i+3] + obs->siteVel[i];

	// Project onto pulsar direction
	redshift = 0.0;
	for (int i = 0; i < 3; ++i)
		redshift += v_total[i] * psr.posPulsar[i];


	// This is z ≈ v_radial / c (special relativistic + kinematic Doppler)


	return redshift;
}

void Profile::create_mask(size_t nchann_in, double sig_threshold, double tail_threshold, size_t max_len, size_t downsample)
{

	if (hdr->nchann != nchann_in && hdr->nchann != 1)
        throw std::runtime_error("The signal was obtained with different number of freq channels");

	std::cout << "Creating mask" << std::endl;

	size_t nchann = 0;
	if (downsample > 0)
		nchann = downsample;
	else
		nchann = nchann_in;

	if (nchann > nchann_in)
		throw std::runtime_error("Downsampling may only be performed with smaller number of channels");


	// define the mask 
	fr = new double[nchann];
	mask = new double[nchann];

	// use 256 MiB buffer for 2d filling
	size_t obs_window = (256ul << 20)/nchann/sizeof(double);
	double *buff = nullptr; 
	buff = (double*) fftw_malloc(sizeof(double) * nchann * obs_window);


	// go to the beginning of the file,
	// remembering current position
	std::streampos current = reader->file.tellg();

	reader->reset();

	size_t filled = 0;
	size_t buf_pos = 0; 
	size_t buf_max = 0;
	size_t counter = 0;

	// ===== Kurtosis calculation ===== 
	// G. M. Nita and D. E. Gary
	// The generalized spectral kurtosis estimator
	// Mon. Not. R. Astron. Soc. 406, L60–L64 (2010) 
	// doi:10.1111/j.1745-3933.2010.00882.x
	
	// State per channel
	fr = new double[nchann];
	mask = new double[nchann];
	double* M2 = new double[nchann];
	double* M4 = new double[nchann];
	double* slice;
	double n;

	std::fill(M2, M2 + nchann, 0.0);
	std::fill(M4, M4 + nchann, 0.0);

	while(true)
	{
		filled = fill_2d(buff, nchann, buf_pos, buf_max, obs_window);

		if (filled == 0) break; // EOF is reached

		for (size_t i = 0; i < filled; ++i)
		{
			slice = buff + i*nchann;

			math::vec_add(M2, slice, nchann);
			math::vec_prod(slice, slice, nchann);
			math::vec_add(M4, slice, nchann);


			counter += 1;
			if (counter > max_len && max_len > 0) break;

			std::cout << "\r\033[K"; // move to the beginning of the line and clear the line
			std::cout << "steps: " << counter << std::flush;
		}

		buf_pos += filled;
		if (counter > max_len && max_len > 0) break;
	}
	std::cout << std::endl;

	n = double(counter);
	math::vec_copy(fr, M2, nchann);
	
	math::vec_prod(M2, M2, nchann);
	math::vec_div(M4, M2, nchann);
	math::vec_scale(M4, n, nchann);
	math::vec_sub(M4, 1.0, nchann);
	math::vec_scale(M4, (n+1.0) / (n-1.0), nchann);
	math::vec_sub(M4, 1.0, nchann); // to shift mean towards zero

	/*
	* std::ofstream test(output_dir + "test.bin");
	* test.write((char*) M4, sizeof(double)*nchann);
	* test.close();
	*/
	// ===== Kurtosis calculation ===== 



	

	// ===== Filtration section ===== 
	 
	// Regect faint tails of the bandpass
	// and kurtosis deviant points 
	// (expected mean of M4 is 0, std is sqrt(4/n))
	//
	double kurt_reg = sig_threshold * std::sqrt(4.0/n);
	double sum = 0.0;
	counter = 0;
	for (size_t i = 0; i < nchann; ++i)
	{
		if (std::abs(M4[i]) < kurt_reg)
		{
			sum += fr[i];
			counter += 1;
		}
	}

	double mean_sens = sum / double(counter);
	double tail_reg = tail_threshold * mean_sens;
	for (size_t i = 0; i < nchann; ++i)
	{
		if (fr[i] > tail_reg && std::abs(M4[i]) < kurt_reg)
			mask[i] = 1.0 / fr[i];
		else
			mask[i] = 0.0;
	}

	// ===== Filtration section ===== 
	
	// ===== Downsampling section ===== 
	//
	// Create a bigger mask according to the smaller one
	// The bigger is filled by piecewise linear interpolation
	// In case leftmost or rightmost channel of smaller mask 
	// is zero, all according bins in the big mask are zero
	//
	double* mask_small = new double[nchann];
	math::vec_copy(mask_small, mask, nchann);

	delete[] mask;
	mask = new double[nchann_in];

	size_t s_n = nchann;
	size_t b_n = nchann_in;
	size_t start_idx = 0, end_idx = 0, bin_size = 0;
	for (size_t i = 0; i < s_n - 1; i++) 
	{
		double left_val  = mask_small[i];
		double right_val = mask_small[i + 1];

		start_idx = static_cast<size_t>(i * (b_n - 1.0) / (s_n - 1.0) + .5);
		end_idx = static_cast<size_t>((i + 1) * (b_n - 1.0) / (s_n - 1.0) + .5);
		bin_size = end_idx - start_idx;


		if (left_val == 0.0 || right_val == 0.0) 
		{
			// Set entire bin to zero
			std::fill(mask + start_idx, mask + end_idx+1, 0.0);
		} 
		else 
		{
			// Piecewise linear interpolation
			for (size_t j = 0; j <= bin_size; j++) 
			{
				double t = static_cast<double>(j) / bin_size;
				mask[start_idx + j] = left_val * (1 - t) + right_val * t;
			}
		}
	}
	std::fill(mask + end_idx, mask + nchann_in, 0.0);
	delete[] mask_small;
	mask_small = nullptr;
	// ===== Downsampling section ===== 



	// ===== Final section ===== 
	// Normilize mask according to the PSRFITS standard:
	// mask \in [0, 1]
	double max = *std::max_element(mask, mask + nchann_in);
	double min = *std::min_element(mask, mask + nchann_in);

	for (size_t i = 0; i < nchann_in; ++i)
		mask[i] = (mask[i] - min) / (max - min);


	// turn back to the initial position in the file
	reader->reset();
	reader->file.seekg(current, std::ios::beg);

    delete[] M2;
    delete[] M4;
	// ===== Filtration section ===== 

	std::cout << "Mask created" << std::endl;
}


BaseHeader* Profile::getHeader()
{
    return reader ? hdr : nullptr;
}
