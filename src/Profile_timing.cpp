#include "Profile.h"
#include "aux_math.h"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <iostream>
#include <iomanip>


namespace 
{
    // Resamples src buffer to match target temporal resolution (tau).
    // Returns the new number of bins. Preserves buffer capacity.
    size_t resample_to_grid(
        double* src, double* dst,
        size_t nbin_src, 
		double tau_src, double tau_dst
    ) 
	{
        if (std::abs(tau_src - tau_dst) < 1.0e-12) 
		{
            math::vec_copy(dst, src, nbin_src);
            return nbin_src;
        }

        // Calculate new length based on total duration
        double duration = nbin_src * tau_src;
        size_t nbin_new = size_t(std::round(duration / tau_dst));

        // Prepare grids
        thread_local static std::vector<double> x_grid, t_grid;
        if (x_grid.capacity() < nbin_new) x_grid.reserve(nbin_new);
        if (t_grid.capacity() < nbin_src) t_grid.reserve(nbin_src);
        x_grid.resize(nbin_new);
        t_grid.resize(nbin_src);

        for (size_t i = 0; i < nbin_new; ++i) x_grid[i] = i * tau_dst;
        for (size_t i = 0; i < nbin_src; ++i) t_grid[i] = i * tau_src;

        math::interp(x_grid.data(), dst, nbin_new, t_grid.data(), src, nbin_src);
        return nbin_new;
    }

    // Calculates phase shift between reference and target sum profiles using CCF.
    double calculate_ccf_shift(double* ref, double* tgt, size_t n_ref, size_t n_tgt) 
	{
        thread_local static std::vector<double> convolution;
        size_t conv_size = n_ref + n_tgt - 1;
        if (convolution.capacity() < conv_size) convolution.reserve(conv_size);
        convolution.resize(conv_size);

        math::ccf(ref, tgt, n_ref, n_tgt, convolution.data(), true);
        double peak_pos = math::max_continuous(convolution.data(), conv_size);
        return peak_pos - (n_tgt - 1);
    }

    double calculate_timing_shift(T2Predictor* pred, long double t0, long double t1, double fcomp0, double fcomp1)
	{
		double P1 = 1.0 / T2Predictor_GetFrequency(pred, t1, fcomp1);

		double phase0 = fmodl(T2Predictor_GetPhase(pred, t0, fcomp0), 1.0L);
		double phase1 = fmodl(T2Predictor_GetPhase(pred, t1, fcomp1), 1.0L);

		double diff = (phase0 - phase1) * P1;

		return diff;
	}	

    // Applies phase shift to src and accumulates into dst using inverse-variance weights.
    // Handles border truncation if lengths differ.
    void shift_and_accumulate_1d(
        double* src, double* dst,
        size_t n_src, size_t n_dst,
        double shift
    ) 
	{
        thread_local static std::vector<double> shifted;
        if (shifted.capacity() < n_src) shifted.reserve(n_src);
        shifted.resize(n_src);

        math::shift(src, shifted.data(), n_src, -shift);

        size_t n_avg = std::min(n_src, n_dst);
        math::vec_add(dst, shifted.data(), n_avg);
    }
} // anonymous namespace



void Profile::accumulate_prf(Profile& other, std::string t2_pred_file)
{
    if (!sum || !other.sum) throw std::runtime_error("Sum array required for accumulation");
    if (hdr->nchann != other.hdr->nchann || hdr->npol != other.hdr->npol)
        throw std::runtime_error("Channel/Polarization mismatch");

    size_t nbin = hdr->obs_window;
    size_t nchann = hdr->nchann;
    size_t npol = hdr->npol;
    double tau = hdr->tau;

	std::vector<size_t> shape(2), shape_other(2);

    // Thread-local buffers for intermediate data
    thread_local static std::vector<double> buf_sum, buf_dyn, buf_raw;

    // 1. Interpolate 'other' to match 'this' temporal resolution
    size_t nbin_other = other.hdr->obs_window;
    double tau_other = other.hdr->tau;

    // Ensure capacity based on worst-case expansion
    size_t max_expected = size_t(std::round((nbin_other * tau_other) / tau)) + 10;
    if (buf_sum.capacity() < max_expected) buf_sum.reserve(max_expected);
    if (dyn && other.dyn)
	{
        if (buf_dyn.capacity() < max_expected * nchann * npol)
            buf_dyn.reserve(max_expected * nchann * npol);
    }
    if (raw && other.raw)
	{
        if (buf_raw.capacity() < max_expected * nchann * npol)
            buf_raw.reserve(max_expected * nchann * npol);
    }

    buf_sum.resize(max_expected);
    if (dyn && other.dyn) buf_dyn.resize(max_expected * nchann * npol);
    if (raw && other.raw) buf_raw.resize(max_expected * nchann * npol);

    // Resample Sum
    nbin_other = resample_to_grid(
        other.sum, buf_sum.data(),
        nbin_other, 
		tau_other, tau
    );
    buf_sum.resize(nbin_other);

    // Resample Dyn (channel/polarization independent)
    if (dyn && other.dyn) 
	{
		shape = {nbin,nchann};
		math::layout_c_to_f(dyn, shape);

        size_t orig_nbin_other = other.hdr->obs_window;
		shape_other = {orig_nbin_other, nchann};
		math::layout_c_to_f(other.dyn, shape);

        for (size_t ch = 0; ch < nchann; ++ch) 
		{
			double* src = other.dyn + ch * orig_nbin_other;
			double* dst = buf_dyn.data() + ch * nbin_other;

			resample_to_grid(src, dst, orig_nbin_other, tau_other, tau);
        }

		shape_other = {nchann, orig_nbin_other};
		math::layout_c_to_f(other.dyn, shape);

        buf_dyn.resize(nbin_other * nchann * npol);
    }

    // Resample Raw (channel/polarization independent)
    if (raw && other.raw) 
	{
		shape = {nbin,nchann};
		math::layout_c_to_f(raw, shape);

        size_t orig_nbin_other = other.hdr->obs_window;
		shape_other = {orig_nbin_other, nchann};
		math::layout_c_to_f(other.raw, shape);

        for (size_t ch = 0; ch < nchann; ++ch) 
		{
			double* src = other.raw + ch * orig_nbin_other;
			double* dst = buf_raw.data() + ch * nbin_other;

			resample_to_grid(src, dst, orig_nbin_other, tau_other, tau);
        }

		shape_other = {nchann, orig_nbin_other};
		math::layout_c_to_f(other.raw, shape);

        buf_raw.resize(nbin_other * nchann * npol);
    }

    // 2. Calculate Phase Shift using Sum
    double shift = 0.0;
	
	if (t2_pred_file == "")
		shift = calculate_ccf_shift(sum, buf_sum.data(), nbin, nbin_other);
	else
	{
		if (!pred) load_predictor(t2_pred_file);

		shift = calculate_timing_shift(pred, hdr->t0, other.hdr->t0, hdr->fcomp, other.hdr->fcomp);
		shift = shift / (hdr->tau*1e-3);
	}

    // 3. Shift and Accumulate Sum
    shift_and_accumulate_1d(
        buf_sum.data(), sum,
        nbin_other, nbin,
        shift);


    // 4. Shift and Accumulate Dyn
    if (dyn && other.dyn) 
	{
		for (size_t f = 0; f < nchann; ++f)
			shift_and_accumulate_1d(
					//buf_dyn.data() + f*nbin_other, dyn + f*nbin,
					buf_dyn.data() + f*nbin_other, dyn + f*nbin,
					nbin_other, nbin,
					shift);

		shape = {nchann, nbin};
		math::layout_c_to_f(dyn, shape);
	}

    if (raw && other.raw) 
	{
		for (size_t f = 0; f < nchann; ++f)
			shift_and_accumulate_1d(
					//buf_raw.data() + f*nbin_other, raw + f*nbin,
					buf_raw.data() + f*nbin_other, raw + f*nbin,
					nbin_other, nbin,
					shift);

		shape = {nchann, nbin};
		math::layout_c_to_f(raw, shape);
	}


	if(! mask)
	{
		mask = new double[nchann];
		std::fill(mask, mask+nchann, 1.0);
	}

	if (other.mask)
		math::vec_add(mask, other.mask, nchann);
	else
		math::vec_add(mask, 1.0, nchann);
}

void Profile::finish_accumulation()
{
	double m = 0.0;
	if (mask)
	{
		m = *std::max_element(mask, mask+hdr->nchann);
		
		for (size_t f = 0; f < hdr->nchann; ++f)
			if (mask[f] == 0.0) 
				mask[f] = std::numeric_limits<double>::infinity();
	}


	if (raw)
	{
		for (size_t t = 0; t < hdr->obs_window; ++t)
			math::vec_div(raw + t*hdr->nchann, mask, hdr->nchann);
	}

	if (dyn)
	{
		for (size_t t = 0; t < hdr->obs_window; ++t)
			math::vec_div(dyn + t*hdr->nchann, mask, hdr->nchann);
	}

	if (sum)
	{
		math::vec_scale(sum, 1.0/m, hdr->obs_window);
	}

	if (mask)
	{
		for (size_t t = 0; t < hdr->nchann; ++t)
			mask[t] = 1.0;
	}


}



void Profile::get_toa (const Profile& tpl, long double *toa, double *toa_err)
{
	if (! (tpl.sum && sum))
		throw std::runtime_error ("There is no folded profile in the input file for timing");
	if (hdr->nsubint > 1)
		throw std::runtime_error ("There is no folded profile in the input file for timing");

}
