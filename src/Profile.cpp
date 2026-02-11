#include "Profile.h"
#include "PRAO_adc.h"   // full definition of PRAO_adc
#include "PRAO_lpa3.h"   // full definition of PRAO_lpa3
#include "IAA_vdif.h"   // full definition of IAA_vdif
#include "PSRFITS.h"   // full definition of IAA_vdif

#include "aux_math.h"
#include "tempo2.h"


#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <ctime>

# define M_PI           3.14159265358979323846
# define C           299792.458

Profile::Profile(
		const std::string& filename, 
		const std::string& format, 
		size_t buffer_size,
		bool save_raw_in, bool save_dyn_in, bool save_sum_in,
		std::string output_dir_in)
{


    if (format == "PRAO_adc") 
	{
        reader = new PRAO_adc(filename, buffer_size);
    } 
	else if (format == "PRAO_lpa3") 
	{
        reader = new PRAO_lpa3(filename, buffer_size);
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

	save_raw = save_raw_in;
	save_dyn = save_dyn_in;
	save_sum = save_sum_in;
	output_dir = output_dir_in;

	hdr = reader->header_ptr;

	redshift = 0.0;
	sumidx = 0;

}

Profile::~Profile()
{
	if(reader)
	{
		delete reader;
		reader = nullptr;
	}


	if(raw)
	{
		delete raw;
		raw = nullptr;
	}

	if(dyn)
	{
		delete dyn;
		dyn = nullptr;
	}

	if(sum)
	{
		delete sum;
		sum = nullptr;
	}

	if(fr)
	{
		delete fr;
		fr = nullptr;
	}

	if(mask)
	{
		delete mask;
		mask = nullptr;
	}

	math::cleanup();

}

double Profile::get_redshift (std::string par_path, std::string site)
{
	// reading pulsar parameters
	// Initialize pulsar and observation
	pulsar psr;
	initialiseOne(&psr, 1, 0); // minimal init, with warnings enabled
	psr.nobs = 1;
    allocateMemory(&psr, 0);
    observation* obs = &psr.obsn[0];	

	char t2_path[500];
	strncpy(t2_path, par_path.c_str(), par_path.length());
	t2_path[par_path.length()] = '\0';
	readParfile(&psr, &t2_path, nullptr, 1); /* Read .par file to define the pulsar's initial parameters */  

    // Set site arrival time and observatory
	const char* obs_code = site.c_str();
    obs->sat = hdr->t0;
    strcpy(obs->telID, obs_code);

	psr.t2cMethod = T2C_TEMPO;
    obs->clockCorr = 1;	
    obs->delayCorr = 1;	



    readEphemeris(&psr, 1, 0);	// fill Earth SSB posvel
	get_obsCoord(&psr, 1);		// fill siteVel
	vectorPulsar(&psr, 1);	// fill pulsar position
							


	double v_total[3];
	for (int i = 0; i < 3; ++i)
		v_total[i] = obs->earth_ssb[i+3] + obs->siteVel[i];

	// Project onto pulsar direction
	redshift = 0.0;
	for (int i = 0; i < 3; ++i)
		redshift += v_total[i] * psr.posPulsar[i];


	// This is z ≈ v_radial / c (special relativistic + kinematic Doppler)


	return redshift;
}



BaseHeader* Profile::getHeader()
{
    return reader ? hdr : nullptr;
}
