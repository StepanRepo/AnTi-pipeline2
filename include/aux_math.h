// aux_math.h

#ifndef AUX_MATH_H
#define AUX_MATH_H

#include <cstddef>

namespace math 
{

/**
 * @brief In-place element-wise addition: a[i] += b[i] for i in [0, n)
 *
 * @param a Input/Output array (modified in-place)
 * @param b Input array (read-only)
 * @param n Size of the arrays
 *
 * @note Arrays must be non-overlapping except where a == destination.
 */
void vec_add(double* a, double* b, size_t n);
void vec_sub(double* a, double* b, size_t n);
void vec_prod(double* a, double* b, size_t n);
void vec_div(double* a, double* b, size_t n);

double mean(double* a, size_t n);
double var(double *a, size_t n, double ddof = 0.0);

void sigmaclip(double *a, bool *mask, size_t n, double threshold);
} // namespace math

#endif // AUX_MATH_H
