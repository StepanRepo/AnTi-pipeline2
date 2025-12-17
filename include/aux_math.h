// aux_math.h

#ifndef AUX_MATH_H
#define AUX_MATH_H

#include <cstddef>
#include <vector>

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

    // --- Statistics ---
    double mean(double* a, size_t n);
    double median(double* a, size_t n);
    double var(double *a, size_t n, double ddof = 0.0);
    void sigmaclip(double *a, bool *mask, size_t n, double threshold, double* mu = nullptr, double* sigma = nullptr);
    void kurtosis_2d(double *data, double *result, size_t n, size_t m);

    // --- Time-domain profile processing ---
	void subtract_baseline(double *data, size_t n, size_t window_size);
	void normalize_std(double *data, size_t n);
	void gaussian_kernel(double* x, size_t n, double fwhm);
	void box_conv(double* x, double* out, size_t win, size_t n);

    // --- Freq-domain profile processing ---

    // --- FITS Layout Conversion ---
    // Converts nD data from C-Style (Row-Major) to Fortran-Style (Col-Major)
    // for writing to FITS files.
	template<typename T>
    void layout_c_to_f(T* src, const std::vector<size_t>& shape);


} // namespace math

#endif // AUX_MATH_H
