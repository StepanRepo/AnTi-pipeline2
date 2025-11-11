#include "Profile.h"
#include "PRAO_adc.h"   // full definition of PRAO_adc
#include "IAA_vdif.h"   // full definition of IAA_vdif
#include <stdexcept>
#include <memory>
#include <algorithm>

#include "tempo2.h"
#include"tempo2pred.h"  // API for TEMPO2 prediction files


# define M_PI           3.14159265358979323846
# define C           299792.458

Profile::Profile(
		const std::string& filename, 
		const std::string& format, 
		size_t buffer_size) 
{
    if (format == "PRAO_adc") 
	{
        reader = std::make_unique<PRAO_adc>(filename, buffer_size);
    } 
	else if (format == "IAA_vdif") 
	{
        reader = std::make_unique<IAA_vdif>(filename, buffer_size);
    }
   	else 
	{
        throw std::invalid_argument("Unsupported format: " + format);
    }

    if (!reader || !reader->is_open) 
	{
        throw std::runtime_error("Reader not initialized or file not open");
	}

	dyn = nullptr;
	dync = nullptr;
	sum = nullptr;
	fr = nullptr;
	mask = nullptr;
	int_prf = nullptr;
}

size_t Profile::fill_2d(double* dyn_spec, size_t time_steps, size_t freq_num) 
{
	if (!reader || !reader->is_open) 
        throw std::runtime_error("Reader not initialized or file not open");

    return reader->fill_2d(dyn_spec, time_steps, freq_num);
}

size_t Profile::fill_1d(fftw_complex *vec, size_t n) 
{
    if (!reader || !reader->is_open) 
        throw std::runtime_error("Reader not initialized or file not open");

    return reader->fill_1d(vec, n);
}

void Profile::dedisperse_incoherent(double DM, size_t nchann)
{
    if (!reader || !reader->is_open) 
	{
        throw std::runtime_error("Reader not initialized or file not open");
	}

	size_t obs_window;
	double tau;

	double fcomp, fmin, fmax;

	double *freqs, *dt;
	double* temp = nullptr;

	if (dyn == nullptr)
		throw std::runtime_error("There is no dynamic spectrum to de-disperce");
	if (sum != nullptr)
		throw std::runtime_error("The file already contains frequency averaged data");

	fmin = reader->header_ptr->fmin;
	fmax = reader->header_ptr->fmax;
	fcomp = reader->header_ptr->fcomp;
	tau = reader->header_ptr->tau;
	nchann = reader->header_ptr->nchann;
	obs_window = reader->header_ptr->obs_window;


	if (fmin == 0.0 || fmax == 0.0)
        throw std::runtime_error("Frequency information was not provided");
	if (nchann != reader->header_ptr->nchann && reader->header_ptr->nchann != 1)
        throw std::runtime_error("File was written with diffderent number of channels");
	if (obs_window == 0)
        throw std::runtime_error("Unknown observational window");

	if (fcomp == 0.0)
	{
		fcomp = std::max(fmin, fmax);
		reader->header_ptr->fcomp = fcomp;
	}


	freqs = new double[nchann];
	dt = new double[nchann];
	double df = (fmax - fmin)  / nchann;

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		freqs[i] = fmin + df * (static_cast<double>(i) + .5);

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		dt[i] = 4.148808e6 * DM * (1.0/freqs[i]/freqs[i] - 1.0/fcomp/fcomp);


	try
	{
		sum = new double[obs_window];
		temp = new double[obs_window];
	} 
	catch (const std::bad_alloc& e) 
	{
		std::cerr << "Allocation failed for dynamic profile. Requested size: " << ((obs_window * nchann *sizeof(double)) / 1024/1024/1024) << " GiB." << std::endl;
		std::cerr << e.what() << std::endl;
			
		delete[] sum;
		delete[] temp;
		sum = nullptr;
		temp = nullptr;

        throw; // Re-throw to signal failure
    }

	int shift;
	for (size_t i = 0; i < nchann; ++i) 
	{
		shift = static_cast<int> (dt[i] / tau + 0.5);


		// Copy time series for this freq into temp
#pragma omp simd
		for (size_t t = 0; t < obs_window; ++t) 
			temp[t] = dyn[((t+shift)%obs_window)  * nchann + i];

		// Write back
		for (size_t t = 0; t < obs_window; ++t) 
			dyn[t * nchann + i] = temp[t];
	}
}

void Profile::dedisperse_incoherent_stream(double DM, size_t nchann)
{
    if (!reader || !reader->is_open) 
	{
        throw std::runtime_error("Reader not initialized or file not open");
	}

	size_t obs_window;
	double tau;

	double fcomp, fmin, fmax;

	double *freqs, *dt;

	int *shift;
	size_t n_DM;

	double *pre = nullptr, *post = nullptr;
	size_t buf_pos, buf_max;

	std::ofstream output;
	std::string id = "stream";

	//std::srand(time(NULL));
	id += std::to_string(std::rand());
	id += ".bin";
	output = std::ofstream(id);

	if (sum != nullptr)
		throw std::runtime_error("The file already contains frequency averaged data");

	fmin = reader->header_ptr->fmin;
	fmax = reader->header_ptr->fmax;
	fcomp = reader->header_ptr->fcomp;
	tau = reader->header_ptr->tau;


	if (fmin == 0.0 || fmax == 0.0)
        throw std::runtime_error("Frequency information was not provided");
	if (fcomp == 0.0)
	{
		fcomp = std::max(fmin, fmax);
		reader->header_ptr->fcomp = fcomp;
	}

	if (reader->header_ptr-> nchann == 1)
		reader->header_ptr-> tau = 
			2.0e-3 * nchann / reader->header_ptr->sampling;

	tau = reader->header_ptr->tau;

	freqs = new double[nchann];
	dt = new double[nchann];
	shift = new int[nchann];
	double df = (fmax - fmin)  / nchann;

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		freqs[i] = fmin + df * (static_cast<double>(i) + .5);

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		dt[i] = 4.148808e6 * DM * (1.0/freqs[i]/freqs[i] - 1.0/fcomp/fcomp);

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		shift[i] = static_cast<int> (dt[i] / tau + 0.5);



	double dtmax = *std::max_element(dt, dt + nchann,
			[](const double& a, const double& b)
			{
			return std::abs(a) < std::abs(b);
			});
	n_DM = static_cast<size_t>(dtmax / tau) + 1;
	n_DM += n_DM % 2;
	
	// set 256 Mib buffer as standard size
	obs_window = std::max(n_DM, (256ul << 20)/nchann/sizeof(double)); 
	obs_window += n_DM;

	try
	{
		pre = new double[obs_window*nchann];
		post = new double[obs_window*nchann];
	} 
	catch (const std::bad_alloc& e) 
	{
		std::cerr << "Allocation failed for dynamic profile. Requested size: " << ((obs_window * nchann *sizeof(double)) / 1024/1024/1024) << " GiB." << std::endl;
		std::cerr << e.what() << std::endl;
			
		delete[] pre;
		pre = nullptr;

		delete[] post;
		post = nullptr;

        throw; // Re-throw to signal failure
    }

	buf_pos = 0;
	buf_max = 0;

	while(true)
	{
		if (buf_pos + obs_window >= buf_max)
		{

			size_t valid_samples = buf_max - buf_pos;
			size_t bytes_to_copy = valid_samples * nchann * sizeof(double);
			std::memmove(pre, pre + buf_pos * nchann, bytes_to_copy);

			buf_pos = 0;
			buf_max = valid_samples;

			size_t filled = fill_2d(
					pre + buf_max*nchann, 
					obs_window - buf_max, nchann);

			if (filled > 0)
				buf_max += filled;
			else
				break; // EOF is reached

			if (buf_max < obs_window)
				break; // EOF is reached
					   //
		}

		std::cout << "step" << std::endl;

		for (size_t t = 0; t < obs_window; ++t) 
		{
			#pragma omp simd
			for (size_t i = 0; i < nchann; ++i) 
				post[t * nchann + i] = pre[((t+shift[i])%obs_window)  * nchann + i];
		}

		// save processed buffer
		// regecting first and last n_DM/2 inputs
		output.write(reinterpret_cast<const char*>(post),
				nchann * (buf_max - n_DM) * sizeof(double));
		buf_pos = buf_max - n_DM;
	}

	// zero padding to save the last part of the file
	std::fill(pre + buf_max*nchann, pre + obs_window*nchann, 0.0);

	// process last read buffer
	for (size_t t = 0; t < obs_window - n_DM; ++t) 
	{
		// Write back
		#pragma omp simd
		for (size_t i = 0; i < nchann; ++i) 
			post[t * nchann + i] = pre[(t+shift[i])  * nchann + i];
	}


	// save processed buffer
	output.write(reinterpret_cast<const char*>(post),
			nchann * buf_max * sizeof(double));

	std::cout << "last step" << std:: endl;

	output.close();
	return;
}
void Profile::dedisperse_coherent(double DM, size_t nchann)
{
return;
}

void Profile::dedisperse_coherent_stream(double DM, size_t nchann)
{
	if (!reader || !reader->is_open) 
	{
		throw std::runtime_error("Reader not initialized or file not open");
	}

	std::ofstream output;

	size_t n_DM, obs_window;
	size_t buf_pos, buf_max;

	double fcomp, fmin, fmax;
	double tau;

	double *freqs;
	double *dt;
	fftw_complex* dphase;
	double re, im;


	fmin = reader->header_ptr->fmin;
	fmax = reader->header_ptr->fmax;
	fcomp = reader->header_ptr->fcomp;

	if (fmin == 0.0 || fmax == 0.0)
        throw std::runtime_error("Frequency information was not provided");
	if (reader->header_ptr->nchann != 1)
        throw std::runtime_error("Coherent dedispersion is unavailable for this file");
	if (sum != nullptr)
		throw std::runtime_error("The file already contains frequency averaged data");

	if (fcomp == 0.0)
	{
		fcomp = std::max(fmin, fmax);
		reader->header_ptr->fcomp = fcomp;
	}


	if (reader->header_ptr-> nchann == 1)
		reader->header_ptr-> tau = 
			1.0e-3 / reader->header_ptr->sampling;

	tau = reader->header_ptr->tau;

	freqs = new double[2*nchann];
	dt = new double[2*nchann];
	dphase = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * 2*nchann));

	double df = (fmax - fmin) / nchann;

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		freqs[i] = fmin + df * (static_cast<double>(i));
	#pragma omp simd
	for (size_t i = nchann; i < 2*nchann; ++i)
		freqs[i] = fmin + df * (static_cast<double>(i));


	#pragma omp simd
	for (size_t i = 0; i < 2*nchann; ++i)
		dt[i] = 4.148808e6 * DM * (1.0/freqs[i]/freqs[i] - 1.0/fcomp/fcomp);

	#pragma omp simd
	for (size_t i = 0; i < 2*nchann; ++i)
	{
		dphase[i][0] = std::cos(-2.0e3 * M_PI * freqs[i] * dt[i]);
		dphase[i][1] = std::sin(-2.0e3 * M_PI * freqs[i] * dt[i]);
	}

	double dtmax = *std::max_element(dt, dt + nchann,
			[](const double& a, const double& b)
			{
			return std::abs(a) < std::abs(b);
			});
	n_DM = static_cast<size_t>(dtmax/tau);
	n_DM += n_DM % 2;

	if (2*nchann <= n_DM)
        throw std::runtime_error("The number of channels is too small for coherent dedispersion. Set at least 2^" + std::to_string(size_t(std::log2(n_DM/2)) + 1));

	obs_window = 2*nchann;

	fftw_complex *in, *out;
	fftw_plan fft, ifft;

	in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * obs_window));
	out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * obs_window));
	sum = new double[obs_window];

	fft = fftw_plan_dft_1d (obs_window, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	ifft = fftw_plan_dft_1d(obs_window, out, out, FFTW_BACKWARD, FFTW_ESTIMATE);


	//std::srand(time(NULL));
	std::string id = "stream";
	id += std::to_string(std::rand());
	id += ".bin";
	output = std::ofstream(id);


	#pragma omp simd
	for (size_t i = 0; i < n_DM; ++i)
	{
		in[i][0] = 0.0;
		in[i][1] = 0.0;
	}


	buf_max = n_DM; 
	buf_pos = 0;

	while(true)
	{
		if (buf_pos + obs_window >= buf_max)
		{

			size_t valid_samples = buf_max - buf_pos;
			size_t bytes_to_copy = valid_samples * sizeof(fftw_complex);
			std::memmove(in, in + buf_pos, bytes_to_copy);

			buf_pos = 0;
			buf_max = valid_samples;

			size_t filled = fill_1d(
					in + buf_max, 
					obs_window - buf_max);

			if (filled > 0)
				buf_max += filled;
			else
				break; // EOF is reached

			if (buf_max < obs_window)
				break; // EOF is reached
		}

		std::cout << "step" << std::endl;

		fftw_execute(fft);

#pragma omp simd
		for (size_t i = 0; i < obs_window; ++i)
		{
			re = out[i][0];
			im = out[i][1];

			out[i][0] = re*dphase[i][0] - im*dphase[i][1];
			out[i][1] = re*dphase[i][1] + im*dphase[i][0];
		}

		fftw_execute(ifft);

		#pragma omp simd
		for (size_t i = 0; i < obs_window; ++i)
		{
			re = out[i][0]/obs_window;	// FFT normalization
			im = out[i][1]/obs_window;	// FFT normalization

			sum[i] = re*re + im*im;
		}

		output.write(reinterpret_cast<const char*>(sum + n_DM),
				(obs_window - n_DM) * sizeof(double));

		buf_pos = obs_window - n_DM;
	}

	// pad the end with zeros
	#pragma omp simd
	for (size_t i = buf_max; i < obs_window; ++i)
	{
		in[i][0] = 0.0;
		in[i][1] = 0.0;
	}
	std::cout << "last step" << std::endl;

	fftw_execute(fft);

	#pragma omp simd
	for (size_t i = 0; i < obs_window; ++i)
	{
		re = out[i][0];
		im = out[i][1];

		out[i][0] = re*dphase[i][0] - im*dphase[i][1];
		out[i][1] = re*dphase[i][1] + im*dphase[i][0];
	}

	fftw_execute(ifft);


	#pragma omp simd
	for (size_t i = 0; i < obs_window; ++i)
	{
		re = out[i][0]/obs_window;	// FFT normalization
		im = out[i][1]/obs_window;	// FFT normalization

		sum[i] = re*re + im*im;
	}

	output.write(reinterpret_cast<const char*>(sum + n_DM),
			(buf_max - n_DM) * sizeof(double));

	output.close();
	fftw_destroy_plan(fft);
	fftw_destroy_plan(ifft);
	fftw_free(in);
	fftw_free(out);
}

void Profile::fold_dyn(double P, size_t nchann)
{
	if (!reader || !reader->is_open) 
		throw std::runtime_error("Reader not initialized or file not open");


	size_t obs_window;
	double tau;

	double *buff = nullptr;
	size_t buf_pos, buf_max;

	// vars for time correction
	size_t rev = 0;
	size_t sumidx = 0;
	double diff;


	if (reader->header_ptr->nchann == 1)
	{
		reader->header_ptr-> tau = 
			2.0e-3 * nchann / reader->header_ptr->sampling;
		reader->header_ptr-> nchann = nchann; 
	}
	else
	{
		throw std::runtime_error("Profile was recorded with different number of frequency channels");
	}

	tau = reader->header_ptr->tau;
	obs_window = size_t(P*1e3 / tau);

	reader->header_ptr->obs_window = obs_window;

	if (obs_window * reader->header_ptr->tau > P*1e3)
		throw std::runtime_error("Observational window must be less than period!");


	try
	{
		dyn = new double [obs_window*nchann];
		buff = new double [2 * obs_window * nchann];
	} 
	catch (const std::bad_alloc& e) 
	{
		std::cerr << "Allocation failed for dynamic profile. Requested size: " << ((obs_window * nchann *sizeof(double)) / 1024/1024/1024) << " GiB." << std::endl;
		std::cerr << e.what() << std::endl;
			
		delete[] dyn;
		delete[] buff;

		dyn = nullptr;
		buff = nullptr;

        throw; // Re-throw to signal failure
    }

	std::fill(dyn, dyn + obs_window*nchann, 0.0);

	buf_pos = 0;
	buf_max = 0;
	rev = 0;

	while(true)
	{
		if (buf_pos + obs_window >= buf_max)
		{

			size_t valid_samples = buf_max - buf_pos;
			size_t bytes_to_copy = valid_samples * nchann * sizeof(double);
			std::memmove(buff, buff + buf_pos * nchann, bytes_to_copy);

			buf_pos = 0;
			buf_max = valid_samples;

			size_t filled = fill_2d(
					buff + buf_max*nchann, 
					2*obs_window - buf_max, nchann);

			if (filled > 0)
				buf_max += filled;
			else
				break; // EOF is reached


			if (buf_pos + obs_window >= buf_max)
				break; // EOF is reached
			
		}


		for (size_t i = 0; i < obs_window; ++i)
		{
			#pragma omp simd
			for (size_t f = 0; f < nchann; ++f)
			{
				dyn[i*nchann + f] += buff[buf_pos*nchann + f];
			}
			buf_pos += 1;
			sumidx += 1;	
		}

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

	if (reader->header_ptr->t0 < T2Predictor_GetStartMJD(pred) ||
			reader->header_ptr->t0 > T2Predictor_GetEndMJD(pred))
        throw std::runtime_error("Date of observation is out of range of predictor dates");


	double P;
	size_t obs_window;
	double tau;
	long double phase, t0, phase0;

	double *buff = nullptr, *buff_curr = nullptr;
	size_t buf_pos, buf_max;

	// vars for time correction
	size_t rev = 0;
	size_t sumidx = 0;
	double diff;

	// vars for frequency correction
	double fmin, fmax, fcomp;

	fmin = reader->header_ptr->fmin;
	fmax = reader->header_ptr->fmax;
	fcomp = reader->header_ptr->fcomp;

	if (fcomp == 0.0)
	{
		fcomp = std::max(fmin, fmax);
		reader->header_ptr->fcomp = fcomp;
	}

	if (reader->header_ptr->nchann == 1)
	{
		reader->header_ptr-> tau = 
			2.0e-3 * nchann / reader->header_ptr->sampling;
		reader->header_ptr-> nchann = nchann; 
	}
	else
	{
		throw std::runtime_error("Profile was recorded with different number of frequency channels");
	}



	// Make obsservational window corresponding to the initial period value
	t0 = reader->header_ptr->t0;
	phase0 = fmodl(T2Predictor_GetPhase(pred, t0, fcomp), 1.0L);
	P = 1.0 / T2Predictor_GetFrequency(pred, t0, fcomp);
	tau = reader->header_ptr->tau;
	obs_window = size_t(P*1e3 / tau);
	reader->header_ptr->obs_window = obs_window;



	if (obs_window * tau > P*1e3)
		throw std::runtime_error("Observational window must be less than period!");


	try
	{
		dyn = new double [obs_window*nchann];
		buff = new double [2 * obs_window * nchann];
	} 
	catch (const std::bad_alloc& e) 
	{
		std::cerr << "Allocation failed for dynamic profile. Requested size: " << ((obs_window * nchann *sizeof(double)) / 1024/1024/1024) << " GiB." << std::endl;
		std::cerr << e.what() << std::endl;
			
		delete[] dyn;
		delete[] buff;

		dyn = nullptr;
		buff = nullptr;

        throw; // Re-throw to signal failure
    }

	std::fill(dyn, dyn + obs_window*nchann, 0.0);

	buf_pos = 0;
	buf_max = 0;
	rev = 0;

	while(true)
	{
		if (buf_pos + obs_window >= buf_max)
		{

			size_t valid_samples = buf_max - buf_pos;
			size_t bytes_to_copy = valid_samples * nchann * sizeof(double);
			std::memmove(buff, buff + buf_pos * nchann, bytes_to_copy);

			buf_pos = 0;
			buf_max = valid_samples;

			size_t filled = fill_2d(
					buff + buf_max*nchann, 
					2*obs_window - buf_max, nchann);

			if (filled > 0)
				buf_max += filled;
			else
				break; // EOF is reached


			if (buf_pos + obs_window >= buf_max)
				break; // EOF is reached
			
		}


		buff_curr = buff + buf_pos * nchann;
		for (size_t i = 0; i < obs_window; ++i)
		{
			#pragma omp simd
			for (size_t f = 0; f < nchann; ++f)
			{
				dyn[i*nchann + f] += buff_curr[i*nchann + f];
			}
		}

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

	reader->header_ptr->total_pulses = rev;

	delete[] buff;
	buff = nullptr;
}

double Profile::get_redshift (std::string par_path)
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
	const char* obs_code = "PO";
    obs->sat = reader->header_ptr->t0;
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


BaseHeader* Profile::getHeader()
{
    return reader ? reader->header_ptr : nullptr;
}
