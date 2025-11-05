// BaseHeader.h
#ifndef BASE_HEADER_H
#define BASE_HEADER_H

#include <string>
#include <stdint.h> // For declaration of maximum size_t value: MAX_SIZE


class BaseHeader 
{
	public:
		// File information
		size_t OBS_SIZE = 0;		// Number of points in the file
		size_t CUT_SIZE = SIZE_MAX;		// Number of points before cutoff
		long double t0 = 0.0L;      // Start time in MJD
		std::string name = "";
		size_t numpar = 0;

		
		
		// System information
		double sampling = 0.0;      // Sampling rate in MHz
		double tau = 0.0;			// Temporal resolution in ms (1/sampling)
		double fmin = 0.0, fmax = 0.0, fcomp = 0.0;

		// file description
		std::string type = "";
		bool folded = false;
		bool dedispersed = false;
		bool summed = false;
		bool complex = false;

		// Dedispersion description
		double dm = 0.0;
		size_t nchann = 0;

		// Folding description
		double period = 0.0;
		std::string cheb_file = "";
		size_t obs_window = 0;
		size_t total_pulses = 0;

		// Virtual destructor for safe polymorphism
		virtual ~BaseHeader() = default;

		// Common interface methods 
		virtual void print() const = 0;
		void update_header(std::string key, std::string value);
};

#endif // BASE_HEADER_H
