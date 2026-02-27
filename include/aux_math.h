// aux_math.h

#ifndef AUX_MATH_H
#define AUX_MATH_H

#include <cstddef>
#include <vector>

// C mathematical functions
#include <numeric>
#include <cmath>
#include <algorithm>

#include <fftw3.h>      // For FFTW library types (fftw_complex, fftw_plan)


namespace math
{
    // --- Basic Math ---
    void vec_add (double* a, double* b, size_t n);
    void vec_sub (double* a, double* b, size_t n);
    void vec_prod(double* a, double* b, size_t n);
    void vec_div (double* a, double* b, size_t n);

    void vec_prod(fftw_complex* a, fftw_complex* b, size_t n);
    void vec_prod(fftw_complex* a, double* b, size_t n);
    void vec_prod(fftw_complex* a, double b, size_t n);

    void vec_add (double* a, double b, size_t n);
    void vec_sub (double* a, double b, size_t n);
    void vec_prod(double* a, double b, size_t n);
    void vec_div (double* a, double b, size_t n);
    void vec_scale (double* a, double b, size_t n);

    void vec_add  (double* c, double* a, double* b, size_t n);
    void vec_sub  (double* c, double* a, double* b, size_t n);
    void vec_prod (double* c, double* a, double* b, size_t n);
    void vec_div  (double* c, double* a, double* b, size_t n);

    void vec_copy  (double* dst, double* src, size_t n);

	double horner (const std::vector<double> &p, double x);
	double find_root (const std::vector<double> &p, double left, double right);


    // --- Statistics ---
    double mean(double* a, size_t n);
    double median(double* a, size_t n);
    double var(double *a, size_t n, double ddof = 0.0);
    void sigmaclip(double *a, bool *mask, size_t n, double threshold, double* mu = nullptr, double* sigma = nullptr);
    void kurtosis_2d(double *data, double *result, size_t n, size_t m);
	double quantile(const double* data, double p, size_t n);


    // --- Time-domain profile processing ---
	void subtract_baseline(double *data, size_t n, size_t window_size);
	void normalize_std(double *data, size_t n, size_t window_size);
	void gaussian_kernel(double* x, size_t n, double fwhm);
	void box_conv(double* x, double* out, size_t win, size_t n);


	/**
	 * Linear interpolation for sorted query points.
	 *
	 * This function assumes:
	 *   - xp is strictly increasing (xp[i] < xp[i+1] for all i)
	 *   - x  is non‑decreasing (x[i] <= x[i+1] for all i)
	 *
	 * @param x      Array of x‑coordinates where interpolation is evaluated (sorted).
	 * @param y      Output array of interpolated values (size nx).
	 * @param nx     Number of elements in x and y.
	 * @param xp     Array of x‑coordinates of the data points (strictly increasing).
	 * @param fp     Array of y‑coordinates of the data points, same length as xp.
	 * @param n      Number of data points (length of xp and fp).
	 * @param left   Value returned for x < xp[0].
	 * @param right  Value returned for x > xp[n-1].
	 */
	void interp(
			const double *x, double *y, size_t nx, 
			const double *xp, const double *fp, size_t n,
			double left = 0.0, double right = 0.0);

	void shift(const double *in, double *out, size_t n, double shift);

	void ccf(double* x, double *y, size_t n1, size_t n2, double *res, bool conj = false);
	double max_continuous(double* x, size_t n, double *err = nullptr);

    // --- Freq-domain profile processing ---

    // --- FITS Layout Conversion ---
    // Converts nD data from C-Style (Row-Major) to Fortran-Style (Col-Major)
    // for writing to FITS files.
	template<typename T>
    void layout_c_to_f(T* src, const std::vector<size_t>& shape);

	// Cleanup function to avoid memory leaks
	void cleanup();


} // namespace math

#endif // AUX_MATH_H

