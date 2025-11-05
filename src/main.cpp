// main.cpp
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include "Profile.h" 

YAML::Node config;

int main() 
{
	try 
	{
		// Load YAML file
		config = YAML::LoadFile("default.yaml");
	}
	catch (const YAML::Exception& e) 
	{
		std::cerr << "YAML Error: " << e.what() << "\n";
		return 1;
	}


	
	std::string mode;
	std::string input_dir;
	std::string output_dir;
	std::string filename;
	int verbose;

	mode = config["general"]["mode"].as<std::string>();
	input_dir = config["general"]["input_dir"].as<std::string>() + "/";
	output_dir = config["general"]["output_dir"].as<std::string>() + "/";
	verbose = config["general"]["verbose"].as<int>(); 


	std::string format = "IAA_vdif";

	for (const auto& filename_yaml : config["files"]) 
	{

		filename = filename_yaml.as<std::string>();

		Profile profile(input_dir + filename, format); 
		BaseHeader* hdr = profile.getHeader();
		if (!hdr) 
			throw std::runtime_error("Header not available");
		else
			hdr->print();

		if (config["advanced"])
		{
			for (const auto& kv : config["advanced"]) 
			{
				const std::string key = kv.first.as<std::string>();
				const std::string value = kv.second.as<std::string>();

				hdr-> update_header(key, value);
			}
		}


		if (config["general"]["t0"] && !config["general"]["t0"].IsNull())
			profile.reader->skip(config["general"]["t0"].as<double>());

		if (config["general"]["t1"] && !config["general"]["t1"].IsNull())
			profile.reader->set_limit(config["general"]["t1"].as<double>());

		if (mode == "dedisperse")
		{


			if (config["options"]["fold"].as<bool>())
			{
				std::string t2_pred_file = "";

				if (config["options"] && config["options"]["t2pred"]) 
					if (! config["options"]["t2pred"].IsNull()) 
						t2_pred_file = config["options"]["t2pred"].as<std::string>();
				

				if (t2_pred_file != "")
				{
					profile.fold_dyn(
							input_dir + config["options"]["t2pred"].as<std::string>(), 
							config["options"]["nchann"].as<size_t>());
				}
				else 
				{
					profile.fold_dyn(
							hdr->period,
							config["options"]["nchann"].as<size_t>());
				}
			}
			else
			{
			throw("in progress");
			}

			profile.dedisperse_incoherent(hdr->dm);


			// After computing dynamic_spectrum
			if (profile.dyn != nullptr)
			{
				std::ofstream output("dyn.bin", std::ios::binary);
				output.write(reinterpret_cast<const char*>(profile.dyn),
						hdr->nchann * hdr->obs_window * sizeof(double));
				output.close();
			}

			if (profile.sum != nullptr)
			{
				std::ofstream output("sum.bin", std::ios::binary);
				output.write(reinterpret_cast<const char*>(profile.sum),
						hdr->OBS_SIZE/2 * sizeof(double));
				output.close();
			}


		}


		//profile.fold_dyn("t2pred.dat", freq_num);
		//profile.dedisperse_incoherent(hdr->dm);
	}

	// Access common header info via polymorphic interface


	//profile.dedisperse_coherent(56.7365);
	//profile.fold_dyn("t2pred.dat", freq_num);
	//profile.dedisperse_incoherent(hdr->dm);


	return 0;
}
