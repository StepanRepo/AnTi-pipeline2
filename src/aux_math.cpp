#include "aux_math.h"
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <iostream>

namespace math 
{

	void vec_add(double* __restrict__ a, double* __restrict__ b, size_t n) 
	{
		#pragma GCC ivdep
		//#pragma omp simd
		for (size_t i = 0; i < n; ++i) 
		{
			a[i] += b[i];
		}
	}

	void vec_sub(double* __restrict__ a, double* __restrict__ b, size_t n) 
	{
		#pragma GCC ivdep
		//#pragma omp simd
		for (size_t i = 0; i < n; ++i) 
		{
			a[i] -= b[i];
		}
	}

	void vec_prod(double* __restrict__ a, double* __restrict__ b, size_t n) 
	{
		#pragma GCC ivdep
		//#pragma omp simd
		for (size_t i = 0; i < n; ++i) 
		{
			a[i] *= b[i];
		}
	}

	void vec_div(double* __restrict__ a, double* __restrict__ b, size_t n) 
	{
		#pragma GCC ivdep
		//#pragma omp simd
		for (size_t i = 0; i < n; ++i) 
		{
			a[i] /= b[i];
		}
	}

	double mean(double* a, size_t n)
	{
		double sum = std::accumulate(a, a + n, 0.0);
		return sum / double(n);
	}

	double var(double* a, size_t n, double ddof)
	{
		double m = mean(a, n);

		double sq_sum = std::inner_product(
				a, a + n,	// first beg, end
				a,		 	// second beg
				0.0,		// initial value
				std::plus<>(), // plus operation (standard)
				[m](double v1, double v2) { return (v1 - m) * (v2 - m); }); // prod operation)

		return sq_sum / (double(n) - ddof);
	}


	void sigmaclip(double* a, bool* mask, size_t n, double threshold) 
	{
		if (n == 0) return;

		// Initial pass: mark all as valid
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

			// Stop if no points were clipped or only 1 point remains
			if (!clipped || valid_count <= 1) break;
		}
	}

} 
