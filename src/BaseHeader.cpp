#include "BaseHeader.h"



#include <cstddef>
#include <string>
#include <iostream>
#include <iomanip>

BaseHeader::~BaseHeader()
{

	if (t_subint) 
	{
		delete[] t_subint;
		t_subint = nullptr;
	}
	if (freqs) 
	{
		delete[] freqs;
		freqs = nullptr;
	}
}


void BaseHeader::print() const
{
    std::cout << "\nOBSERVATION HEADER\n";
    std::cout << "==================\n";
    
    std::cout << std::left;
    
    // File information
    std::cout << std::setw(25) << "Name:" << (name.empty() ? "(unset)" : name) << "\n";
    std::cout << std::setw(25) << "Mode:" << (MODE.empty() ? "(unset)" : MODE) << "\n";
    std::cout << std::setw(25) << "Data type:" << (cmplx ? "Complex" : "Real") << "\n";
    std::cout << std::setw(25) << "Start time (MJD):" << std::fixed << std::setprecision(12) << t0 << "\n";
    
    // Data dimensions
    std::cout << std::setw(25) << "Total points:" << OBS_SIZE << "\n";
    if (CUT_SIZE != SIZE_MAX) {
        std::cout << std::setw(25) << "Cutoff points:" << CUT_SIZE << "\n";
    }
    std::cout << std::setw(25) << "Channels:" << nchann << "\n";
    std::cout << std::setw(25) << "Polarizations:" << npol << "\n";
    std::cout << std::setw(25) << "Subintegrations:" << nsubint << "\n";
    
    // Time domain
    if (sampling > 0) {
        std::cout << std::setw(25) << "Sampling rate:" << std::fixed << std::setprecision(3) 
                  << sampling << " MHz\n";
    } else {
        std::cout << std::setw(25) << "Sampling rate:" << "(unset)\n";
    }
    
    if (tau > 0) {
        std::cout << std::setw(25) << "Temporal resolution:" << std::fixed << std::setprecision(3) 
                  << tau << " ms\n";
    }
    
    // Frequency domain
    if (fmin > 0) {
        std::cout << std::setw(25) << "Frequency range:" << std::fixed << std::setprecision(2) 
                  << fmin << " - " << fmax << " MHz\n";
        std::cout << std::setw(25) << "Bandwidth:" << std::fixed << std::setprecision(2) 
                  << (fmax - fmin) << " MHz\n";
        if (nchann > 0) {
            std::cout << std::setw(25) << "Channel bandwidth:" << std::fixed << std::setprecision(2) 
                      << ((fmax - fmin) / nchann) << " MHz\n";
        }
    } else {
        std::cout << std::setw(25) << "Frequency range:" << "(unset)\n";
    }
    
    // Dedispersion
    std::cout << std::setw(25) << "DM:" << std::fixed << std::setprecision(3) 
              << dm << " pc cm^-3\n";
    
    // Folding
    if (period > 0) {
        std::cout << std::setw(25) << "Period:" << std::scientific << std::setprecision(6) 
                  << period << " s\n";
        std::cout << std::setw(25) << "Frequency:" << std::fixed << std::setprecision(6) 
                  << (1.0/period) << " Hz\n";
        std::cout << std::setw(25) << "Observation window:" << obs_window << " bins\n";
        if (!cheb_file.empty()) {
            std::cout << std::setw(25) << "Chebyshev file:" << cheb_file << "\n";
        }
    } else {
        std::cout << std::setw(25) << "Folding:" << "Not folded (search mode)\n";
    }
    
    // Subintegrations info
    if (nsubint > 0 && t_subint) {
        if (nsubint <= 5) {
            for (size_t i = 0; i < nsubint; ++i) {
                std::cout << std::setw(25) << ("Subint " + std::to_string(i) + ":") 
                          << std::fixed << std::setprecision(1) << t_subint[i] << " ms\n";
            }
        } else {
            double min_val = t_subint[0];
            double max_val = t_subint[0];
            double sum = 0;
            for (size_t i = 0; i < nsubint; ++i) {
                sum += t_subint[i];
                if (t_subint[i] < min_val) min_val = t_subint[i];
                if (t_subint[i] > max_val) max_val = t_subint[i];
            }
            std::cout << std::setw(25) << "Subint min duration:" << std::fixed << std::setprecision(1) 
                      << min_val << " ms\n";
            std::cout << std::setw(25) << "Subint avg duration:" << std::fixed << std::setprecision(1) 
                      << (sum/nsubint) << " ms\n";
            std::cout << std::setw(25) << "Subint max duration:" << std::fixed << std::setprecision(1) 
                      << max_val << " ms\n";
        }
    }
    
    std::cout << std::right; // Reset alignment
    std::cout << "\n";
}

void BaseHeader:: update_header(std::string key, std::string value)
{

	if (key == "name") 
	{
		name = value; 
	} else if (key == "t0") 
	{
		t0 = std::stod(value);
	}
	else if (key == "period") 
	{
		period = std::stold(value);
	} 
	else if (key == "tay" || key == "tau") 
	{
		// time sampling is stored in ms
		tau = std::stod(value);
		// sampling rate stored in MHz
		sampling = 1.0e-3 / tau;
	}
	else if (key == "sampling") 
	{
		// sampling rate stored in MHz
		sampling = std::stod(value);
		// time sampling is stored in ms
		tau = 1.0e-3/sampling;
	}
	else if (key == "numpointwin" || key == "obs_window") 
	{
		obs_window = std::stoi(value);
	} 
	else if (key == "dm") 
	{
		dm = std::stod(value);
	}
	else if (key == "freq0" || key == "F0" || key == "Fmin" || key == "fmin") 
	{
		fmin = std::stod(value);

		double df = (fmax - fmin) / double(nchann);
		if (!freqs) freqs = new double[nchann];

		for (size_t i = 0; i < nchann; ++i)
			freqs[i] = fmin + df*(double(i) + .5);
	}
	else if (key == "freq511" || key == "F511" || key == "Fmax" || key == "fmax") 
	{
		fmax = std::stod(value);

		double df = (fmax - fmin) / double(nchann);
		if (!freqs) freqs = new double[nchann];

		for (size_t i = 0; i < nchann; ++i)
			freqs[i] = fmin + df*(double(i) + .5);
	}
	else
	{
		std::cout << "Unknown parameter key to update observational information: " << key << std::endl;
	}
}
