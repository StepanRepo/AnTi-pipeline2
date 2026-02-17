#include "Profile.h"
#include "PSRFITS_Writer.h"
#include "aux_math.h"
#include "fftw3.h"

#include <cstddef>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <new>
#include <numeric>
#include <stdexcept>


void Profile::load_mask(size_t nchann)
{
	mask = new double[nchann];

	try 
	{
		reader->fill_wts(mask, nchann);
	} 
	catch(std::runtime_error &error)
	{
		delete[] mask;
		throw error;
	}
}

void Profile::create_mask(size_t nchann_in, double sig_threshold, double tail_threshold, size_t max_len, size_t downsample)
{

	if (hdr->nchann != nchann_in && hdr->nchann != 1)
        throw std::runtime_error("The signal was obtained with different number of freq channels");

	if (verbose > 0)
		std::cout << "Creating mask" << std::endl;

	size_t nchann = 0;
	if (downsample > 0)
		nchann = downsample;
	else
		nchann = nchann_in;

	if (nchann > nchann_in) nchann = nchann_in;


	// use 256 MiB buffer for 2d filling
	size_t obs_window = (256ul << 20)/nchann/hdr->npol/sizeof(double);
	obs_window = std::min(obs_window, hdr->OBS_SIZE);

	double *buff = nullptr; 
	buff = (double*) fftw_malloc(sizeof(double) * nchann * obs_window);


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
	double* slice = nullptr;
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

			if (verbose > 0)
			{
				std::cout << "\r\033[K"; // move to the beginning of the line and clear the line
				std::cout << "steps: " << counter << std::flush;
			}
		}

		buf_pos += filled;
		if (counter > max_len && max_len > 0) break;
	}
	std::cout << std::endl;

	fftw_free(buff);
	buff = nullptr;
	slice = nullptr;

	n = double(counter);
	math::vec_copy(fr, M2, nchann);
	
	math::vec_prod(M2, M2, nchann);
	math::vec_div(M4, M2, nchann);
	math::vec_scale(M4, n, nchann);
	math::vec_sub(M4, 1.0, nchann);
	math::vec_scale(M4, (n+1.0) / (n-1.0), nchann);
	math::vec_sub(M4, 1.0, nchann); // to shift mean towards zero

	
	// ===== Kurtosis calculation ===== 

	if (counter == 0)
		throw std::runtime_error("The file is empty");

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
		if ((fr[i] > tail_reg) && (std::abs(M4[i]) < kurt_reg))
			mask[i] = 1.0/fr[i];
		else
			mask[i] = 0.0;
	}

	// ===== Filtration section ===== 


	// ===== Checking section ===== 
	// If SK method failed, drop to 
	// a simpler method: sigmaclip
	if (std::accumulate(mask, mask+nchann, 0.0) == 0.0)
	{

		bool *res = new bool[nchann];
		double mu, sigma;

		math::sigmaclip(M4, res, nchann, sig_threshold, &mu, &sigma);

		// Find characteristic level of frequency 
		// response to regect faint tails
		//mu = math::median(fr, nchann);
		//tail_reg = tail_threshold * mu;

		mu = 0.0;
		counter = 0;
		for (size_t i = 0; i < nchann; ++i)
		{
			if (res[i])
			{
				mu += fr[i];
				counter += 1;
			}
		}
		mu = mu / double(counter);
		tail_reg = tail_threshold * mu;

		for (size_t i = 0; i < nchann; ++i)
		{
			if ((fr[i] > tail_reg) && res[i])
				mask[i] = 1.0/fr[i];
			else
				mask[i] = 0.0;
		}

		delete[] res;
	}
	// ===== Checking section ===== 

	
	std::ofstream test(output_dir + "mask.bin");
	test.write((char*) mask, sizeof(double)*nchann);
	test.close();
	


	// ===== Downsampling section ===== 
	//
	// Create a bigger mask according to the smaller one
	// The bigger is filled by piecewise linear interpolation
	// In case leftmost or rightmost channel of smaller mask 
	// is zero, all according bins in the big mask are zero
	//

	if (nchann < nchann_in)
	{
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
	}
	// ===== Downsampling section ===== 




	// ===== Final section ===== 
	// Normilize mask according to the PSRFITS standard:
	// mask \in [0, 1]
	//
	// Find max and min of non-masked channels
	double max_val = *std::max_element(mask, mask+nchann);
	double min_val = *std::min_element(mask, mask+nchann);

	for (size_t i = 0; i < nchann_in; ++i)
		mask[i] = (mask[i] - min_val) / (max_val - min_val);

	// turn back to the initial position in the file
	reader->reset();

    delete[] M2;
    delete[] M4;
	// ===== Filtration section ===== 

	PSRFITS_Writer writer(output_dir + "wts_" + reader->filename);
	writer.createPrimaryHDU("SEARCH", hdr);
	writer.append_subint_search(
			nullptr, 
			hdr->freqs, mask,
			1, nchann, 1, 
			0.0, 0.0, 0.0, "");

	if (verbose > 0)
		std::cout << "Mask created" << std::endl;

}
