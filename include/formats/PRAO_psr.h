#ifndef PRAO_PSR_H
#define PRAO_PSR_H

#include "BaseHeader.h"
#include "BaseReader.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>


long double PRAOTime2MJD(std::string const time_in);

class PRAOHeader : public BaseHeader 
{
	private:
	public:
		size_t numpar = 0;
		std::string start_date_s = "";
		std::string start_utc_s = "";



		// Constructors
		PRAOHeader();
		PRAOHeader(const PRAOHeader&) = delete;
		PRAOHeader& operator=(const PRAOHeader&) = delete;

		// Destructor - overrides BaseHeader destructor
		virtual ~PRAOHeader() override;

		// PRAO specific methods
		void read_header(std::ifstream &file);

		// Override BaseHeader virtual methods if needed
		//virtual void print() const override;

		// Accessors for PRAO specific data
		std::string get_start_date() const { return start_date_s; }
		std::string get_start_utc() const { return start_utc_s; }
		int get_numpar() const { return numpar; }
};


class PRAO_psr : public BaseReader 
{
	private:
		// Data buffers
		int32_t* raw_data = nullptr;
		long double t0_orig = 0.0L;

		// Local header object (NOT dynamically allocated)
		PRAOHeader header;  // Stack-allocated, not pointer

	public:
		// Constructor
		PRAO_psr(const std::string& filename_in, size_t buffer_size);

		// Delete copy/move operations
		PRAO_psr(const PRAO_psr&) = delete;
		PRAO_psr& operator=(const PRAO_psr&) = delete;
		PRAO_psr(PRAO_psr&&) = delete;
		PRAO_psr& operator=(PRAO_psr&&) = delete;

		// Destructor
		virtual ~PRAO_psr();

		// Methods
		void set_limit(double t) override;
		bool fill_buffer() override;
		void skip(double sec) override;
		double point2time(size_t point) override;
		bool allow_1d() override;
		bool allow_2d() override;
};

#endif // PRAO_LPA3_H

