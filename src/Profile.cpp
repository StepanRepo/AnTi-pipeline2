#include "Profile.h"
#include "PRAO_adc.h"   // full definition of PRAO_adc
#include "IAA_vdif.h"   // full definition of IAA_vdif
#include"tempo2pred.h"  // API for TEMPO2 prediction files
#include"jpleph.h"
#include <stdexcept>
#include <memory>
#include <algorithm>

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

void Profile::fill_1d(fftw_complex *vec, size_t n) 
{
    if (!reader || !reader->is_open) 
        throw std::runtime_error("Reader not initialized or file not open");

    reader->fill_1d(vec, n);
}

void Profile::dedisperse_incoherent(double DM)
{
    if (!reader || !reader->is_open) 
	{
        throw std::runtime_error("Reader not initialized or file not open");
	}

	size_t obs_window;
	double tau;

	double fcomp, fmin, fmax;
	size_t nchann;

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

	if (fmin == 0.0 || fmax == 0.0 || nchann == 1)
        throw std::runtime_error("Frequency information was not provided");
	if (obs_window == 0)
        throw std::runtime_error("Unknown observational window");

	if (fcomp == 0.0)
	{
		fcomp = std::max(fmin, fmax);
		reader->header_ptr->fcomp = fcomp;
	}


	freqs = new double[nchann];
	dt = new double[nchann];
	double df = reader->header_ptr->sampling / 2.0  / nchann;

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		freqs[i] = std::min(fmin, fmax) + df * (static_cast<double>(i) + .5);

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
			temp[t] = dyn[t * nchann + i];

		// Roll in temp
		std::rotate(temp, temp+shift, temp+obs_window);

		// Write back
		for (size_t t = 0; t < obs_window; ++t) 
			dyn[t * nchann + i] = temp[t];
	}
}

void Profile::dedisperse_coherent(double DM)
{
	throw std::runtime_error("The function is broken");


    if (!reader || !reader->is_open) 
	{
        throw std::runtime_error("Reader not initialized or file not open");
	}

	size_t sum_len;
	size_t nchann;

	double fcomp, fmin, fmax;
	double tau;

	double *freqs;
	double *dt;

	if (sum != nullptr)
		throw std::runtime_error("The file already contains frequency averaged data");

	fmin = reader->header_ptr->fmin;
	fmax = reader->header_ptr->fmax;
	fcomp = reader->header_ptr->fcomp;
	nchann = reader->header_ptr->nchann;

	if (fmin == 0.0 || fmax == 0.0)
        throw std::runtime_error("Frequency information was not provided");

	if (fcomp == 0.0)
	{
		fcomp = std::max(fmin, fmax);
		reader->header_ptr->fcomp = fcomp;
	}




	if (! (reader->header_ptr-> tau > 0.0))
		reader->header_ptr-> tau = 
			2.0e-3 / reader->header_ptr->sampling;

	tau = reader->header_ptr->tau;



	sum_len = reader->header_ptr->OBS_SIZE / 2;
	sum = new double [sum_len];

	freqs = new double[nchann];
	dt = new double[nchann];

	double df = reader->header_ptr->sampling / nchann;
	double f0 = std::max(fmin, fmax);

	#pragma omp simd
	for (size_t i = 0; i < nchann/2; ++i)
		freqs[i] = f0 - df * (static_cast<double>(i));

	for (size_t i = nchann/2; i < nchann; ++i)
		freqs[i] = f0 - df * (static_cast<double>(nchann - i));

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		dt[i] = 4.148808e6 * DM * (1.0/freqs[i]/freqs[i] - 1.0/fcomp/fcomp);


	double dtmax = *std::max_element(dt, dt + nchann,
			[](const double& a, const double& b)
			{
			return std::abs(a) < std::abs(b);
			});


	fftw_complex* dphase;
	dphase = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * nchann));

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
	{
		dphase[i][0] = std::cos(-2.0e3 * M_PI * freqs[i] * dt[i]);
		dphase[i][1] = std::sin(-2.0e3 * M_PI * freqs[i] * dt[i]);
	}

	fftw_complex *in, *out;
	fftw_plan fft, ifft;

	in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * nchann ));
	out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * nchann));

	fft = fftw_plan_dft_1d(nchann, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	ifft = fftw_plan_dft_1d(nchann, out, out, FFTW_BACKWARD, FFTW_ESTIMATE);

	for (int k = 0; k < 2; ++k)
		fill_1d(in, nchann);

	double re, im;

	fftw_execute(fft);
	
	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
	{
		re = out[i][0];
		im = out[i][1];

		out[i][0] = re*dphase[i][0] - im*dphase[i][1];
		out[i][1] = re*dphase[i][1] + im*dphase[i][0];
	}

	fftw_execute(ifft);

	sum = new double[nchann/2];
	#pragma omp simd
	for (size_t i = 0; i < nchann/2; ++i)
	{
		re = out[i][0]/nchann;
		im = out[i][1]/nchann;

		sum[i] = re*re + im*im;
	}
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
/*
	// find Earth's posvel in AU and AU/day respectively
	void *jpleph;
	double posvel[6];
	jpleph =  jpl_init_ephemeris("/home/wall-e/miniconda3/envs/ent15y/share/tempo2/ephemeris/DE405.1950.2050", NULL, NULL);

	jpl_pleph(jpleph, 2400000.5 + reader->header_ptr->t0, 3, 11, posvel, 1);

	double au2km = jpl_get_double(jpleph, JPL_EPHEM_AU_IN_KM);

	double ra = (3.0*15.0 + 32.0*15.0/60.0 * 59.4096*15.0/3600.0) * M_PI/180.0;
	double dec = (54.0 + 34.0/60.0 * 43.329/3600.0) * M_PI/180.0;

	double psr_coord[3];
	psr_coord[0] = cos(ra) * cos(dec);
	psr_coord[1] = sin(ra) * cos(dec);
	psr_coord[2] = sin(dec);

	for (int i = 0; i < 3; ++i)
		std::cout << posvel[i]*au2km << " km" << std::endl;
	for (int i = 3; i < 6; ++i)
		std::cout << posvel[i]*au2km/86400.0 << " km/s" << std::endl;

	double proj = 0.0;
	for (int i = 0; i < 3; ++i)
		proj += psr_coord[i] * posvel[i+3];

	std::cout << "redshift: " << proj/C  << std::endl;
	std::cout << "dP: " << proj/C * reader->header_ptr->period  << std::endl;


	return;
*/

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
		dyn = new double [obs_window*nchann]();
		buff = new double [2 * obs_window * nchann]();
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

			size_t to_fill = 2*obs_window - buf_max;

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
		#pragma omp simd
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


BaseHeader* Profile::getHeader()
{
    return reader ? reader->header_ptr : nullptr;
}
