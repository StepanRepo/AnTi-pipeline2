#include "aux_math.h"
#include "fftw3.h"
#include <Eigen/Core>
#include <cstddef>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath> // for sqrt, abs
#include <cstdint>

#include <iostream>
#include <fstream>

#define EIGEN_NO_DEBUG

namespace math 
{

	// Helper typedef to map a raw pointer to an Eigen Array
	using MapType = Eigen::Map<Eigen::ArrayXd>;
	using ConstMapType = Eigen::Map<const Eigen::ArrayXd>;

	// ------------------------------------------------------------
	// 1. Basic Vector Math
	// Eigen automatically uses SIMD (AVX/SSE) here.
	// ------------------------------------------------------------

	void vec_add(double* a, double* b, size_t n) 
	{
		MapType(a, n) += ConstMapType(b, n);
	}

	void vec_sub(double* a, double* b, size_t n) 
	{
		MapType(a, n) -= ConstMapType(b, n);
	}

	void vec_prod(double* a, double* b, size_t n) 
	{
		MapType(a, n) *= ConstMapType(b, n);
	}

	void vec_prod(fftw_complex* __restrict__ a, fftw_complex* __restrict__ b, size_t n) 
	{
		double re, im;
		#pragma omp simd
		for (size_t i = 0; i < n; ++i)
		{
			re = a[i][0];
			im = a[i][1];

			a[i][0] = re*b[i][0] - im*b[i][1];
			a[i][1] = re*b[i][1] + im*b[i][0];
		}
	}

	void vec_prod(fftw_complex* __restrict__ a, double* __restrict__ b, size_t n) 
	{
		#pragma omp simd
		for (size_t i = 0; i < n; ++i)
		{
			a[i][0] = a[i][0] * b[i];
			a[i][1] = a[i][1] * b[i];
		}
	}

	void vec_prod(fftw_complex* __restrict__ a, double b, size_t n) 
	{
		#pragma omp simd
		for (size_t i = 0; i < n; ++i)
		{
			a[i][0] = a[i][0] * b;
			a[i][1] = a[i][1] * b;
		}
	}

	void vec_div(double* a, double* b, size_t n) 
	{
		MapType(a, n) /= ConstMapType(b, n);
	}

	void vec_add(double* a, double b, size_t n) 
	{
		MapType(a, n) += b;
	}

	void vec_sub(double* a, double b, size_t n) 
	{
		MapType(a, n) -= b;
	}

	void vec_prod(double* a, double b, size_t n) 
	{
		MapType(a, n) *= b;
	}

	void vec_div(double* a, double b, size_t n) 
	{
		MapType(a, n) /= b;
	}

	void vec_scale(double* a, double b, size_t n) 
	{
		MapType(a, n) *= b;
	}

	void vec_add(double* c, double* a, double* b, size_t n) 
	{
		MapType(c, n) = ConstMapType(a, n) + ConstMapType(b, n);
	}

	void vec_sub(double* c, double* a, double* b, size_t n) 
	{
		MapType(c, n) = ConstMapType(a, n) - ConstMapType(b, n);
	}

	void vec_prod(double* c, double* a, double* b, size_t n) 
	{
		MapType(c, n) = ConstMapType(a, n) * ConstMapType(b, n);
	}

	void vec_div(double* c, double* a, double* b, size_t n) 
	{
		MapType(c, n) = ConstMapType(a, n) / ConstMapType(b, n);
	}

    void vec_copy  (double* dst, double* src, size_t n)
	{
		std::memcpy(dst, src, sizeof(double)*n);
	}

	// ------------------------------------------------------------
	// 2. Statistics
	// ------------------------------------------------------------

	double mean(double* a, size_t n) 
	{
		if (n == 0) return 0.0;
		return ConstMapType(a, n).mean();
	}

	double median(double* a, size_t n) 
	{
		std::sort(a, a+n);

		if (n % 2 != 0) 
			return a[n / 2];
		else
			return (a[(n - 1) / 2] + a[n / 2]) / 2.0;
	}

	double var(double *a, size_t n, double ddof) 
	{
		if (n <= ddof) return 0.0;

		ConstMapType vec(a, n);
		double avg = vec.mean();

		// (x - mean)^2 . sum()
		double sq_sum = (vec - avg).square().sum();
		return sq_sum / (n - ddof);
	}

	void sigmaclip(double* a, bool* mask_in, size_t n, double threshold, double* mu, double* sigma)
	{
		if (n == 0) return;
		bool* mask = nullptr;

		// Initial pass: mark all as valid
		if (!mask_in)
			mask = new bool[n];
		else
			mask = mask_in;

		std::fill(mask, mask + n, true);
		size_t valid_count = n;

		double m, stddev, sum;
		size_t count;
		bool clipped;

		while (true)
		{
			// Compute mean of valid points
			sum = 0.0;
			count = 0;

			for (size_t i = 0; i < n; ++i)
			{
				if (mask[i])
				{
					sum += a[i];
					++count;
				}
			}
			if (count == 0) break;
			m = sum / double(count);

			// Compute STD of valid points
			sum = 0.0;
			count = 0;

			for (size_t i = 0; i < n; ++i)
			{
				if (mask[i])
				{
					sum += (a[i] - m) * (a[i] - m);
					++count;
				}
			}
			if (count < 2) break;
			stddev = std::sqrt(sum / (double(count) - 1.0));

			// Clip points beyond threshold
			// Track if any point were clipped
			clipped = false;
			for (size_t i = 0; i < n; ++i)
			{
				if (mask[i] && std::abs(a[i] - m) > threshold * stddev)
				{
					mask[i] = false;
					clipped = true;
					--valid_count;
				}
			}

			if (sigma)
				sigma[0] = stddev; 
			if (mu)
				mu[0] = m; 

			// Stop if no points were clipped or only 1 point remains
			if (!clipped || valid_count <= 1) break;
		}

		if (!mask_in)
			delete[] mask;
	}

    void kurtosis_2d(double *data, double *kurt, size_t nt, size_t nf)
	{
		double* sum1 = new double[nf];
		double* sum2 = new double[nf];
		double* sum4 = new double[nf];
		double* slice = new double[nf];

		std::fill(sum1, sum1 + nf, 0.0);
		std::fill(sum2, sum2 + nf, 0.0);
		std::fill(sum4, sum4 + nf, 0.0);


		for (size_t i = 0; i < nt; ++i)
		{
			std::memcpy(slice, data + i*nf, sizeof(double)*nf);

			vec_add(sum1, slice, nf);

			vec_prod(slice, slice, nf);
			vec_add(sum2, slice, nf);

			vec_prod(slice, slice, nf);
			vec_add(sum4, slice, nf);
		}

		for (size_t i = 0; i < nf; ++i)
			sum1[i] = sum1[i] / double(nt);

		for (size_t i = 0; i < nf; ++i)
			sum2[i] = sum2[i] / double(nt);

		vec_prod(sum1, sum1, nf);
		vec_sub(sum2, sum1, nf);

		for (size_t i = 0; i < nf; ++i)
			sum4[i] = sum4[i] / double(nt);

		for (size_t i = 0; i < nf; ++i)
		{
			if (sum2[i] > 0.0)
				kurt[i] = sum4[i] / (sum2[i] * sum2[i]) - 3.0;
			else
				kurt[i] = 0.0;
		}

		if (nt > 3)
		{
			for (size_t i = 0; i < nf; ++i)
				kurt[i] = (nt - 1) / ((nt - 2)*(nt - 3)) * ((nt + 1)*kurt[i] + 6);
		}


		delete[] sum1;
		delete[] sum2;
		delete[] sum4;
		delete[] slice;
	}


	// P² Algorithm to find approximate value of a given quantile
	// This algorithm was implemented by Qwen3-Coder AI
	double quantile(double *data, double p, size_t n) {
		if (n == 0 || p < 0.0 || p > 1.0) return NAN;

		if (n <= 5) {
			// For small arrays, sort and return exact quantile
			double *temp = new double[n];

			for (size_t i = 0; i < n; ++i) temp[i] = data[i];
			qsort(temp, n, sizeof(double),
					[](const void *a, const void *b) {
					double x = *(const double*)a, y = *(const double*)b;
					return (x > y) - (x < y);
					});
			double result = temp[(int)round((n - 1) * p)];
			delete [] temp;
			return result;
		}

		// Initialize markers from first 5 elements
		double q[5], ns[5], dns[5];
		int nm[5];

		for (int i = 0; i < 5; ++i) q[i] = data[i];
		qsort(q, 5, sizeof(double),
				[](const void *a, const void *b) {
				double x = *(const double*)a, y = *(const double*)b;
				return (x > y) - (x < y);
				});

		for (int i = 0; i < 5; ++i) nm[i] = i;
		ns[0] = 0; ns[1] = 2*p; ns[2] = 4*p; ns[3] = 2+2*p; ns[4] = 4;
		dns[0] = 0; dns[1] = p/2; dns[2] = p; dns[3] = (1+p)/2; dns[4] = 1;

		// Process remaining elements
		for (size_t idx = 5; idx < n; ++idx) {
			double x = data[idx];

			int k;
			if (x < q[0]) { q[0] = x; k = 0; }
			else if (x < q[1]) k = 0;
			else if (x < q[2]) k = 1;
			else if (x < q[3]) k = 2;
			else if (x < q[4]) k = 3;
			else { q[4] = x; k = 3; }

			for (int i = k + 1; i < 5; ++i) nm[i]++;
			for (int i = 0; i < 5; ++i) ns[i] += dns[i];

			for (int i = 1; i <= 3; ++i) {
				double d = ns[i] - nm[i];
				if ((d >= 1 && nm[i + 1] - nm[i] > 1) ||
						(d <= -1 && nm[i - 1] - nm[i] < -1)) {
					int dInt = (d > 0) ? 1 : -1;

					double d1 = nm[i] - nm[i - 1];
					double d2 = nm[i + 1] - nm[i];
					double term1 = (d1 + d) * (q[i + 1] - q[i]) / d2;
					double term2 = (d2 - d) * (q[i] - q[i - 1]) / d1;
					double qs = q[i] + d / (nm[i + 1] - nm[i - 1]) * (term1 + term2);

					if (q[i - 1] < qs && qs < q[i + 1])
						q[i] = qs;
					else
						q[i] = q[i] + dInt * (q[i + dInt] - q[i]) / (nm[i + dInt] - nm[i]);

					nm[i] += dInt;
				}
			}
		}

		return q[2];
	}


	// ------------------------------------------------------------
	// 3. Time-domain profile processing
	// ------------------------------------------------------------

	// This function was written by Qwen3-Max AI
	// with human tweaks
	void subtract_baseline(double* data, size_t n, size_t k) {
		if (k == 0 || n == 0) return;
		if (k % 2 == 0) k++; // Ensure odd
		const size_t half = k / 2;

		// Step 1: Find global min/max for 8-bit quantization
		double min_val = quantile(data, .05, n);
		double max_val = quantile(data, .95, n);

		double mean = (max_val + min_val) / 2.0;
		double dev = max_val - mean; // about 2 sigmas
		min_val = mean - 5.0*dev; // make it 10 sigmas
		max_val = mean + 5.0*dev; // to account for trends



		//std::cout << "q_0.01: " << min_val << std::endl;
		//std::cout << "q_0.99: " << max_val << std::endl;


		const double eps = 1e-12;
		if (max_val - min_val < eps) return; // constant data

		// Use 8-bit quantization (256 bins)
		const uint32_t NBINS = 256;
		const double scale = NBINS / (max_val - min_val);
		const double inv_scale = (max_val - min_val) / NBINS;

		// Histogram for current window
		std::vector<uint32_t> hist(NBINS, 0);
		uint32_t count = 0;

		// Helper: quantize value to bin (clamped to [0,255])
		auto quantize = [&](double x) -> uint8_t {
			int bin = static_cast<int>((x - min_val) * scale);
			if (bin < 0) bin = 0;
			else if (uint32_t(bin) >= NBINS) bin = NBINS - 1;
			return static_cast<uint8_t>(bin);
		};

		// Helper: get median from histogram (linear interpolation)
		auto get_median = [&]() -> double {
			uint32_t target = (count + 1) / 2;
			uint32_t cum = 0;
			for (uint32_t b = 0; b < NBINS; ++b) {
				cum += hist[b];
				if (cum >= target) {
					double frac = (target - (cum - hist[b])) / static_cast<double>(hist[b]);
					return min_val + (b + frac) * inv_scale;
				}
			}
			return max_val;
		};

		// Reflection index helper
		auto reflect_index = [&](int64_t idx) -> size_t {
			if (idx >= 0 && idx < (int64_t)n) return idx;
			if (idx < 0) return static_cast<size_t>(-idx - 1);
			return static_cast<size_t>(2 * (n - 1) - idx);
		};

		// Initialize window for i=0
		count = 0;
		std::memset(hist.data(), 0, NBINS * sizeof(uint32_t));
		for (size_t j = 0; j < k; ++j) {
			size_t idx = reflect_index(static_cast<int64_t>(j) - static_cast<int64_t>(half));
			double val = data[idx];
			uint8_t bin = quantize(val);
			hist[bin]++;
			count++;
		}

		// Store medians temporarily (we can't overwrite data while sliding)
		static thread_local std::vector<double> medians;
		if (medians.size() != n) medians.resize(n);
		medians[0] = get_median();

		// Sliding window: add right, remove left
		for (size_t i = 1; i < n; ++i) {
			// Remove leftmost element of previous window
			size_t left_idx = reflect_index(static_cast<int64_t>(i - 1) - static_cast<int64_t>(half));
			double left_val = data[left_idx];
			uint8_t left_bin = quantize(left_val);
			hist[left_bin]--;

			// Add new rightmost element
			size_t right_idx = reflect_index(static_cast<int64_t>(i) + static_cast<int64_t>(half));
			double right_val = data[right_idx];
			uint8_t right_bin = quantize(right_val);
			hist[right_bin]++;

			medians[i] = get_median();
		}

		// Subtract baseline IN-PLACE
		for (size_t i = 0; i < n; ++i) {
			data[i] -= medians[i];
		}

		/*
			std::ofstream test0("data/med.bin");
			test0.write((char*) medians.data(), n*sizeof(double));
			test0.close();
			*/
	}



	// This function was written by Qwen3-Max AI
	// with human tweaks
	void normalize_std(double* data, size_t n, size_t k) {
		if (k == 0 || n == 0) return;
		if (k % 2 == 0) k++; // Ensure odd
		const size_t half = k / 2;

		// Reflect index (same as subtract_baseline)
		auto reflect_index = [&](int64_t idx) -> size_t {
			if (idx >= 0 && idx < (int64_t)n) return idx;
			if (idx < 0) return static_cast<size_t>(-idx - 1);
			return static_cast<size_t>(2 * (n - 1) - idx);
		};

		// Quantize absolute values |data[i]| over full range
		//double max_abs = 0.0;
		//for (size_t i = 0; i < n; ++i) {
		//	double a = std::abs(data[i]);
		//	if (a > max_abs) max_abs = a;
		//}

		// Step 1: Find global min/max for 8-bit quantization
		double min_val = quantile(data, .05, n);
		double max_val = quantile(data, .95, n);

		double mean = (max_val + min_val) / 2.0;
		double dev = max_val - mean; // about 2 sigmas
		min_val = mean - 2.5*dev; // make it 5 sigmas
		max_val = mean + 2.5*dev; // assuming there is no trend
		double max_abs = max_val - min_val;


		const double eps = 1e-12;
		if (max_abs < eps) return;

		const uint32_t NBINS = 256;
		const double scale = NBINS / max_abs;
		const double inv_scale = max_abs / NBINS;

		auto quantize_abs = [&](double x) -> uint8_t {
			double a = std::abs(x);
			int bin = static_cast<int>(a * scale);
			if (bin < 0) bin = 0;
			else if (uint32_t(bin) >= NBINS) bin = NBINS - 1;
			return static_cast<uint8_t>(bin);
		};

		// Sliding histogram for |x|
		std::vector<uint32_t> hist(NBINS, 0);
		static thread_local std::vector<double> mad_arr;
		if (mad_arr.size() != n) mad_arr.resize(n);

		// Initialize first window
		for (size_t j = 0; j < k; ++j) {
			size_t idx = reflect_index(static_cast<int64_t>(j) - static_cast<int64_t>(half));
			uint8_t bin = quantize_abs(data[idx]);
			hist[bin]++;
		}

		// Helper: get MAD from histogram
		auto get_mad = [&](uint32_t count) -> double {
			uint32_t target = (count + 1) / 2;
			uint32_t cum = 0;
			for (uint32_t b = 0; b < NBINS; ++b) {
				cum += hist[b];
				if (cum >= target) {
					double frac = (target - (cum - hist[b])) / static_cast<double>(hist[b]);
					return (b + frac) * inv_scale;
				}
			}
			return max_abs;
		};

		mad_arr[0] = get_mad(k);

		// Slide window
		for (size_t i = 1; i < n; ++i) {
			// Remove left
			size_t left_idx = reflect_index(static_cast<int64_t>(i - 1) - static_cast<int64_t>(half));
			uint8_t left_bin = quantize_abs(data[left_idx]);
			hist[left_bin]--;

			// Add right
			size_t right_idx = reflect_index(static_cast<int64_t>(i) + static_cast<int64_t>(half));
			uint8_t right_bin = quantize_abs(data[right_idx]);
			hist[right_bin]++;

			mad_arr[i] = get_mad(k);
		}

		// Apply normalization: divide by scaled MAD
		const double scale_mad = 1.4826;
		for (size_t i = 0; i < n; ++i) {
			double sigma = scale_mad * mad_arr[i];
			if (sigma < eps) sigma = 1.0;
			data[i] /= sigma;
		}
	}

	void gaussian_kernel(double* x, size_t n, double fwhm)
	{
		double sigma = fwhm / 2.355;
		double t0 = double(n/2);
		double sum_sq = 0;

		for (size_t i = 0; i < n; ++i)
			x[i] = std::exp(-.5 * (double(i) - t0) * (double(i) - t0) / sigma / sigma); 

		for (size_t i = 0; i < n; ++i)
			sum_sq += x[i]*x[i];

		vec_scale(x, 1.0/sum_sq, n);
	}

	void box_conv(double* x, double* out, size_t win, size_t n)
	{
		if (n == 0 || win == 0 || win > n) return;

		double* window = new double[win];
		size_t head = 0;
		double sum = 0.0;

		// Initialize first window with circular indexing
		for (int i = -int(win/2); i < int(win/2); ++i)
		{
			int idx = ((i % int(n)) + int(n)) % int(n); // Safe modulo
			window[i + win/2] = x[idx];
		}

		// Precompute initial sum
		sum = 0.0;
		for (size_t i = 0; i < win; ++i)
			sum += window[i];

		// Process each point
		for (size_t i = 0; i < n; ++i)
		{
			// Output mean
			out[i] = sum / win;

			// Get new value (circular index)
			int new_idx = ((int(i) - int(win/2)) + int(n)) % int(n);
			double new_val = x[new_idx];

			// Update window and sum
			double old_val = window[head];
			sum = sum - old_val + new_val;
			window[head] = new_val;
			head = (head + 1) % win;
		}

		delete[] window;
	}

	// define vars that will be used multiple times
	thread_local static fftw_complex* X = nullptr;
	thread_local static fftw_complex* Y = nullptr;
	thread_local static fftw_plan fft, ifft = nullptr;
	thread_local static size_t size = 0;
	void ccf(double* x, double *y, size_t n1, size_t n2, double *res)
	{
		// define temporal vars
		double *X_re = (double*) X;
		double *Y_re = (double*) Y;
		size_t curr_size = 1L << size_t(log2(n1+n2-1) + 1);


		// allocate internal arrays for performing FFTs
		if (size < curr_size)
		{
			size = curr_size;

			if (X) fftw_free(X);
			if (Y) fftw_free(Y);
			if (fft) fftw_destroy_plan(fft);
			if (ifft) fftw_destroy_plan(ifft);

			X = (fftw_complex*) fftw_malloc((size/2 + 1) * sizeof(fftw_complex));
			Y = (fftw_complex*) fftw_malloc((size/2 + 1) * sizeof(fftw_complex));

			X_re = (double*) X;
			Y_re = (double*) Y;

			fft = fftw_plan_dft_r2c_1d(size, X_re, X, FFTW_ESTIMATE);
			ifft = fftw_plan_dft_c2r_1d(size, X, X_re, FFTW_ESTIMATE);
		}


		// copy input arrays and pad them with zeros
		vec_copy(X_re, x, n1);
		vec_copy(Y_re, y, n2);

		std::fill(X_re + n1, X_re + size, 0.0);
		std::fill(Y_re + n2, Y_re + size, 0.0);

		fftw_execute_dft_r2c(fft, X_re, X);
		fftw_execute_dft_r2c(fft, Y_re, Y);

		vec_prod(X, Y, size/2 + 1);
		fftw_execute(ifft);
		vec_prod(X_re, 1.0/double(size), size);

		vec_copy(res, X_re, n1+n2-1);
		//vec_copy(res, X_re + (size - n2), n2);
		//vec_copy(res+n2, X_re , n1-1);
	}


	// ------------------------------------------------------------
	// 3. FITS Layout Conversion 
	// ------------------------------------------------------------


	// Helper for the general tensor shuffle (Out of place logic used internally)
	// Template helper to handle specific dimensions (2D, 3D, 4D)
	// Eigen Tensors need rank at compile time for max speed.
	template <typename T, int Rank>
		void tensor_shuffle_buffer(const T* src, T* dest, const std::vector<size_t>& shape) {
			// 1. Setup Dimensions
			Eigen::array<Eigen::Index, Rank> src_dims;
			Eigen::array<Eigen::Index, Rank> dest_dims;    // Reversed shape
			Eigen::array<Eigen::Index, Rank> shuffle_idxs; // Indices to reverse axes

			for(int i=0; i<Rank; ++i) {
				src_dims[i] = shape[i];
				// Reverse dimensions for the destination view
				dest_dims[i] = shape[Rank - 1 - i];
				// Define shuffle (transpose) order: {2, 1, 0} for 3D
				shuffle_idxs[i] = Rank - 1 - i;
			}

			// 2. Map BOTH as RowMajor (C-Style) to satisfy the static assert
			// We treat the 'dest' pointer as if it holds the Transposed array in C-layout.
			Eigen::TensorMap<Eigen::Tensor<const T, Rank, Eigen::RowMajor>> src_t(src, src_dims);
			Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor>> dest_t(dest, dest_dims);

			// 3. Perform the physical shuffle
			// This reads src, transposes the indices, and writes linearly to dest.
			// The result in 'dest' is the binary equivalent of F-Layout.
			dest_t = src_t.shuffle(shuffle_idxs);
		}

	template <typename T>
		void layout_c_to_f(T* data, const std::vector<size_t>& shape) {
			size_t rank = shape.size();
			size_t total_size = 1;
			for(auto s : shape) total_size *= s;

			// OPTIMIZATION: 2D Square Matrix
			if (rank == 2 && shape[0] == shape[1]) {
				size_t dim = shape[0];
				for (size_t i = 0; i < dim; ++i) {
					for (size_t j = i + 1; j < dim; ++j) {
						std::swap(data[i * dim + j], data[j * dim + i]);
					}
				}
				return;
			}

			// GENERAL CASE: Swap Buffer
			std::vector<T> temp(total_size);

			switch(rank) {
				case 2: tensor_shuffle_buffer<T, 2>(data, temp.data(), shape); break;
				case 3: tensor_shuffle_buffer<T, 3>(data, temp.data(), shape); break;
				case 4: tensor_shuffle_buffer<T, 4>(data, temp.data(), shape); break;
				default:
						std::copy(data, data + total_size, temp.begin());
						break;
			}

			// Copy back to original pointer
			std::copy(temp.begin(), temp.end(), data);
		}


	template void layout_c_to_f<double>(double*, const std::vector<size_t>&);
	template void layout_c_to_f<float>(float*, const std::vector<size_t>&);
	template void layout_c_to_f<int>(int*, const std::vector<size_t>&);
	template void layout_c_to_f<int16_t>(int16_t*, const std::vector<size_t>&);
	template void layout_c_to_f<char>(char*, const std::vector<size_t>&);
	template void layout_c_to_f<unsigned char>(unsigned char*, const std::vector<size_t>&);


	// Clenup function for static variables
	void cleanup()
	{
		if(X) 
		{
			fftw_free(X);
			X = nullptr;
		}

		if(Y) 
		{
			fftw_free(Y);
			Y = nullptr;
		}

		if(fft)
		{
			fftw_destroy_plan(fft);
			fft = nullptr;
		}

		if(ifft)
		{
			fftw_destroy_plan(ifft);
			ifft = nullptr;
		}
	}

} // namespace math
