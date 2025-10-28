#include "Profile.h"
//#include "PRAO_adc.h"   // full definition of PRAO_adc
#include "IAA_vdif.h"   // full definition of IAA_vdif
#include <stdexcept>
#include <memory>
#include <algorithm>

# define M_PI           3.14159265358979323846

Profile::Profile(
		const std::string& filename, 
		const std::string& format, 
		size_t buffer_size) 
{
    //if (format == "PRAO_adc") 
	//{
    //    reader = std::make_unique<PRAO_adc>(filename, buffer_size);
    //} 
	//else 
	if (format == "IAA_vdif") 
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

	dyn = std::vector<double>();
	dync = std::vector<std::complex<double>>();
	sum = std::vector<double>();
	fr = std::vector<double>();
	mask = std::vector<double>();
	int_prf = std::vector<double>();
}

void Profile::fill_2d(std::vector<double>& dyn_spec, size_t freq_num) 
{
	if (!reader || !reader->is_open) 
        throw std::runtime_error("Reader not initialized or file not open");

    reader->fill_2d(dyn_spec, freq_num);
}

void Profile::fill_1d(std::vector<std::complex<double>>& dyn_spec, size_t freq_num) 
{
    if (!reader || !reader->is_open) 
        throw std::runtime_error("Reader not initialized or file not open");

    reader->fill_1d(dyn_spec, freq_num);
}

void Profile::dedisperse_incoherent(double DM)
{
    if (!reader || !reader->is_open) 
	{
        throw std::runtime_error("Reader not initialized or file not open");
	}

	size_t obs_window;
	size_t sum_len;
	size_t nchann;

	double fcomp, fmin, fmax;
	double tau;

	if (sum.size() > 0)
		throw std::runtime_error("The file already contains frequency averaged data");

	fmin = reader->header_ptr->fmin;
	fmax = reader->header_ptr->fmax;
	fcomp = reader->header_ptr->fcomp;

	if (fmin == 0.0 || fmax == 0.0)
        throw std::runtime_error("Frequency information was not provided");

	if (fcomp == 0.0)
	{
		fcomp = std::max(fmin, fmax);
		reader->header_ptr->fcomp = fcomp;
	}


	if (! (reader->header_ptr-> tau > 0.0))
		reader->header_ptr-> tau = 
			2.0e-3  * reader->header_ptr->nchann / 
			reader->header_ptr->sampling;

	tau = reader->header_ptr->tau;

	nchann = reader->header_ptr->nchann;


	sum_len = reader->header_ptr->OBS_SIZE / nchann / 2;
	sum = std::vector<double>(sum_len);

	std::vector<double> freqs (nchann);
	double df = reader->header_ptr->sampling / 2.0  / nchann;

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		freqs[i] = std::min(fmin, fmax) + df * (static_cast<double>(i) + .5);


	std::vector<double> dt (nchann);

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		dt[i] = 4.148808e6 * DM * (1.0/freqs[i]/freqs[i] - 1.0/fcomp/fcomp);

	double dtmax = *std::max_element(dt.begin(), dt.end(),
			[](const double& a, const double& b)
			{
			return std::abs(a) < std::abs(b);
			});

	obs_window = 4*size_t (std::abs(dtmax)) / tau;
	std::vector<double> pre (obs_window * nchann);
	std::vector<double> post (obs_window * nchann);

	fill_2d(pre, nchann);


	std::vector<double> temp(obs_window);
	int shift;
	for (size_t i = 0; i < nchann; ++i) 
	{
		shift = static_cast<int> (dt[i] / tau + 0.5);

		// Copy time series for this freq into temp
		#pragma omp simd
		for (size_t t = 0; t < obs_window; ++t) 
			temp[t] = pre[t * nchann + i];

		// Roll in temp
		std::rotate(temp.begin(), temp.begin()+shift, temp.end());

		// Write back
		for (size_t t = 0; t < obs_window; ++t) 
			post[t * nchann + i] = temp[t];
	}

	dyn = post;
}

void Profile::dedisperse_coherent(double DM)
{
    if (!reader || !reader->is_open) 
	{
        throw std::runtime_error("Reader not initialized or file not open");
	}

	size_t obs_window;
	size_t sum_len;
	size_t nchann;

	double fcomp, fmin, fmax;
	double tau;

	if (sum.size() > 0)
		throw std::runtime_error("The file already contains frequency averaged data");

	fmin = reader->header_ptr->fmin;
	fmax = reader->header_ptr->fmax;
	fcomp = reader->header_ptr->fcomp;

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

	nchann = reader->header_ptr->nchann;


	sum_len = reader->header_ptr->OBS_SIZE / nchann / 2;
	sum = std::vector<double>(sum_len);

	std::vector<double> freqs (nchann);
	double df = reader->header_ptr->sampling / 2.0  / nchann;

	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		freqs[i] = std::min(fmin, fmax) + df * (static_cast<double>(i) + .5);

	std::vector<double> dt (nchann);
	#pragma omp simd
	for (size_t i = 0; i < nchann; ++i)
		dt[i] = 4.148808e6 * DM * (1.0/freqs[i]/freqs[i] - 1.0/fcomp/fcomp);

	double dtmax = *std::max_element(dt.begin(), dt.end(),
			[](const double& a, const double& b)
			{
			return std::abs(a) < std::abs(b);
			});

	obs_window = 4*size_t (std::abs(dtmax)) / tau / nchann;
	std::vector<std::complex<double>> pre(nchann * obs_window);

	fill_1d(pre, nchann);

	fftw_complex* dphase;
	dphase = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * nchann));


	std::complex<double> I (0.0, 1.0);
	for (size_t i = 0; i < nchann; ++i)
	{
		dphase[i][0] = std::exp(-2.0e9 * M_PI * I * freqs[i] * dt[i]).real();
		dphase[i][1] = std::exp(-2.0e9 * M_PI * I * freqs[i] * dt[i]).imag();
	}

	fftw_complex* fft_arr;
	fftw_plan p;

	fft_arr = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * nchann));
	p = fftw_plan_dft_1d(nchann, fft_arr, fft_arr, FFTW_BACKWARD, FFTW_PATIENT);

	for (size_t i = 0; i < obs_window; ++i)
	{
		std::memcpy(
				reinterpret_cast<void*> (fft_arr),          // Destination: the array for IFFT
				reinterpret_cast<const void*>(&pre[i * nchann]), // Source: a part of complex dynamic spectrum
				nchann * sizeof(fftw_complex)                      // Number of bytes to copy
				);

		#pragma omp simd
		for (size_t k = 0; k < nchann; ++k)
		{
			fft_arr[k][0] = fft_arr[k][0] * dphase[k][0] - fft_arr[k][1] * dphase[k][1];
			fft_arr[k][1] = fft_arr[k][0] * dphase[k][1] + fft_arr[k][0] * dphase[k][1];
		}

		fftw_execute(p);

		double re, im;
		for (size_t k = 0; k < nchann; ++k)
		{
			re = fft_arr[k][0];
			im = fft_arr[k][1];
			
			sum[i*nchann + k] = re*re + im*im;
		}
	}

	std::ofstream output("data.bin", std::ios::binary);
	output.write(reinterpret_cast<const char*>(sum.data()),
			sum.size() * sizeof(double));
	output.close();


}

void Profile::fold(double P)
{
	if (!reader || !reader->is_open) 
        throw std::runtime_error("Reader not initialized or file not open");

	std::cout << P << std::endl;
}	

BaseHeader* Profile::getHeader()
{
    return reader ? reader->header_ptr : nullptr;
}
