#include <string>
#include <iostream>
#include "BaseHeader.h"

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
	}
	else if (key == "freq511" || key == "F511" || key == "Fmax" || key == "fmax") 
	{
		fmax = std::stod(value);
	}
	else
	{
		std::cout << "Unknown parameter key to update observational information: " << key << std::endl;
	}
}
