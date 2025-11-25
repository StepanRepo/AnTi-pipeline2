// aux_math.h

#ifndef AUX_MATH_H
#define AUX_MATH_H

#include <cstddef>
#include <vector>

namespace math
{
    // --- Basic Math ---
    void vec_add (double* a, double* b, size_t n);
    void vec_sub (double* a, double* b, size_t n);
    void vec_prod(double* a, double* b, size_t n);
    void vec_div (double* a, double* b, size_t n);

    // --- Statistics ---
    double mean(double* a, size_t n);
    double var(double *a, size_t n, double ddof = 0.0);
    void sigmaclip(double *a, bool *mask, size_t n, double threshold);

    // --- FITS Layout Conversion ---
    // Converts nD data from C-Style (Row-Major) to Fortran-Style (Col-Major)
    // for writing to FITS files.
	template<typename T>
    void layout_c_to_f(T* src, const std::vector<size_t>& shape);


} // namespace math

#endif // AUX_MATH_H
