#include "Profile.h"
#include "formats/PRAO_adc.h"   // full definition of PRAO_adc
#include "formats/PRAO_lpa3.h"   // full definition of PRAO_lpa3
#include "formats/PRAO_psr.h"   // full definition of PRAO_adc
#include "formats/IAA_vdif.h"   // full definition of IAA_vdif
#include "formats/PSRFITS.h"   // full definition of IAA_vdif

#include <cstddef>
#include <stdexcept>
#include <ctime>

# define C           299792.458

Profile::Profile(
		const std::string& filename, 
		const std::string& format, 
		size_t buffer_size,
		std::string output_dir_in,
		int verbose_in
		)
{


	if (format == "PRAO_adc") 
	{
		reader = new PRAO_adc(filename, buffer_size);
	} 
	else if (format == "PRAO_lpa3") 
	{
		reader = new PRAO_lpa3(filename, buffer_size);
	} 
	else if (format == "PRAO_psr") 
	{
		reader = new PRAO_psr(filename, buffer_size);
	} 
	else if (format == "IAA_vdif") 
	{
		reader = new IAA_vdif(filename, buffer_size);
	}
	else if (format == "PSRFITS") 
	{
		reader = new PSRFITS(filename, buffer_size);
	}
	else 
	{
		throw std::invalid_argument("Unsupported format: " + format);
	}

	if (!reader || !reader->is_open) 
	{
		throw std::runtime_error("Reader not initialized or file not open");
	}

	raw = nullptr;
	dyn = nullptr;
	sum = nullptr;
	fr = nullptr;
	mask = nullptr;

	pred = nullptr;
	psr  = nullptr;

	output_dir = output_dir_in;

	hdr = reader->header_ptr;

	redshift = 0.0;
	sumidx = 0;

	verbose = verbose_in;

	// disable streaming and load the whole file in RAM
	if (hdr->MODE == "PSR")
		fill_PSR();

}

Profile::~Profile()
{
	if(reader)
	{
		delete reader;
		reader = nullptr;
	}

	if(pred)
	{
		T2Predictor_Destroy(pred);
		delete pred;
		pred = nullptr;
	}

	if(psr)
	{
		destroyOne(psr);
		delete psr;
	}

	if(raw)
	{
		delete[] raw;
		raw = nullptr;
	}

	if(dyn)
	{
		delete[] dyn;
		dyn = nullptr;
	}

	if(sum)
	{
		delete[] sum;
		sum = nullptr;
	}

	if(fr)
	{
		delete[] fr;
		fr = nullptr;
	}

	if(mask)
	{
		delete[] mask;
		mask = nullptr;
	}

}


void Profile::skip(const double t_in)
{
	long double t = 0.0;

	// Consider time to be in MJD if
	// it exceeds 10hrs 
	if (t_in > 36000.0L)
		t = (t_in - hdr->t0) * 86400.0L;
	else
		t = t_in;

	t = t < 0.0 ? 0.0 : t;
	reader->skip(double(t));
}

void Profile::set_limit(const double t_in)
{
	long double t = 0.0;

	// Consider time to be in MJD if
	// it exceeds 10hrs 
	if (t_in > 36000.0L)
		t = (t_in - hdr->t0) * 86400.0L;
	else
		t = t_in;

	t = t < 0.0 ? 0.0 : t;

	double DM = hdr->dm;
	double fmin = hdr->fmin;
	double fmax = hdr->fmax;

	double t_DM = 4.15e3 * DM * std::abs(1/fmin/fmin - 1/fmax/fmax);

	reader->set_limit(double(t) + t_DM);
}


BaseHeader* Profile::getHeader()
{
	return reader ? hdr : nullptr;
}
