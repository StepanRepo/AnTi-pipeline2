#ifndef PRAO_LPA3_H
#define PRAO_LPA3_H

#include "BaseHeader.h"
#include "BaseReader.h"

#include <cstddef>
#include <fstream>
#include <string>



class LPA3Header : public BaseHeader 
{
	private:
		// LPA3 specific members
		size_t numpar = 0;
		std::string start_date_s;
		std::string start_utc_s;

	public:
		// Constructors
		LPA3Header();
		LPA3Header(const LPA3Header&) = delete;
		LPA3Header& operator=(const LPA3Header&) = delete;

		// Destructor - overrides BaseHeader destructor
		virtual ~LPA3Header() override;

		// LPA3 specific methods
		void read_header(std::ifstream &file);

		// Override BaseHeader virtual methods if needed
		//virtual void print() const override;

		// Accessors for LPA3 specific data
		std::string get_start_date() const { return start_date_s; }
		std::string get_start_utc() const { return start_utc_s; }
		int get_numpar() const { return numpar; }

	private:
		// Helper methods
		void parse_frequency_info();
};


class PRAO_lpa3 : public BaseReader 
{
	private:
		// Data buffers
		float* raw_data = nullptr;

		// Local header object (NOT dynamically allocated)
		LPA3Header header;  // Stack-allocated, not pointer

	public:
		// Constructor
		PRAO_lpa3(const std::string& filename_in, size_t buffer_size);

		// Delete copy/move operations
		PRAO_lpa3(const PRAO_lpa3&) = delete;
		PRAO_lpa3& operator=(const PRAO_lpa3&) = delete;
		PRAO_lpa3(PRAO_lpa3&&) = delete;
		PRAO_lpa3& operator=(PRAO_lpa3&&) = delete;

		// Destructor
		virtual ~PRAO_lpa3();

		// Methods
		void set_limit(double t) override;
		bool fill_buffer() override;
		void skip(double sec) override;
		double point2time(size_t point) override;
		bool allow_1d() override;
		bool allow_2d() override;
};

#endif // PRAO_LPA3_H
