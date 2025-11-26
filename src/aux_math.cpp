#include "aux_math.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath> // for sqrt, abs
#include <iostream>

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

	void vec_div(double* a, double* b, size_t n) 
	{
		MapType(a, n) /= ConstMapType(b, n);
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

	// ------------------------------------------------------------
	// 3. Time-domain profile processing
	// ------------------------------------------------------------
	void subtract_baseline(double *data, size_t n, size_t window_size) 
	{
		if (n == 0 || window_size == 0 || window_size > n) return;
		bool *mask = nullptr;
		double *window = nullptr;

		mask = new bool[window_size];
		window = new double[window_size];

		double mu, std, med;


		for (size_t i = 0; i < n-window_size; ++i) 
		{
			std::memcpy(window, data+i, window_size*sizeof(double));

			med = median(window, window_size);
			sigmaclip(window, mask, window_size, 3.0, &mu, &std);

			if (std == 0.0) std = 1.0; // Avoid division by zero

			// Apply normalization: (x - median) / std
			data[i] = (data[i] - med) / std;
		}

		for (size_t i = n-window_size; i < n; ++i) 
		{
			std::memcpy(window, data+i, (n-i)*sizeof(double));
			med = median(window, n-i);
			sigmaclip(window, mask, n-i, 3.0, &mu, &std);

			if (std == 0.0) std = 1.0; // Avoid division by zero

			// Apply normalization: (x - median) / std
			data[i] = (data[i] - med) / std;
		}


		delete[] mask;
		delete[] window;
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

} // namespace math
