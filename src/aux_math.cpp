#include "aux_math.h"

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

} 
