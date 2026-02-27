// BaseHeader.h
#ifndef BASE_HEADER_H
#define BASE_HEADER_H

#include <string>
#include <stdint.h> // For declaration of maximum size_t value: MAX_SIZE
#include <vector>


class BaseHeader 
{
	public:
		// File information
		size_t OBS_SIZE = 0;		// Number of TIME STEPS in the file
		size_t CUT_SIZE = SIZE_MAX;	// Number of TIME STEPS before cutoff
		long double t0 = 0.0L;      // Start time in MJD
		std::string name = "";
		std::string pol_type = "";
		std::string MODE = "";

		size_t nsubint = 0;
		std::vector<double> t_subint; // time of the centers of subintegrations
		
		
		// System information
		double sampling = 0.0;      // Sampling rate in MHz
		double tau = 0.0;			// Temporal resolution in ms (1/sampling)
		double fmin = 0.0, fmax = 0.0, fcomp = 0.0;
		std::vector<double> freqs;

		// file description
		bool cmplx = false;

		// Dedispersion description
		double dm = 0.0;
		size_t nchann = 0;
		size_t npol = 0;
		std::string dds_mthd = "";

		// Folding description
		double period = 0.0;
		std::string cheb_file = "";
		size_t obs_window = 0;


		// ===== Additional information =====
		// Antenna
		std::string observer = "";
		std::string proj_id = "";
		std::string telescop = "";

		double ant_x = 0.0, ant_y = 0.0, ant_z = 0.0;
		std::string fd_poln = "";
		std::string backend = "";

		// Source
		float equinox = 2000.0;
		std::string RA = "", DEC = "";


		// ===== Functions =====
		//
		//Default constructor
		BaseHeader() = default;

		// Virtual destructor for safe polymorphism
		virtual ~BaseHeader();

		// Compiler generated copy operations
		BaseHeader(const BaseHeader&) = default;
		BaseHeader& operator=(const BaseHeader&) = default;

		// Common interface methods 
		virtual void print() const;
		void update_header(std::string key, std::string value);
};

#endif // BASE_HEADER_H

