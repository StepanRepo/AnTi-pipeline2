// main.cpp
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include "Profile.h" 

std::string resolve_path(const std::string& input) 
{
	std::string til_result, result;
	std::string::size_type pos = 0;

	if (input.substr(0, 1) == "~")
		til_result = "$HOME/" + input.substr(1, input.length());
	else
		til_result = input;


	while (pos < til_result.length()) 
	{
		std::string::size_type dollar = til_result.find('$', pos);

		// There is no dollar sign
		if (dollar == std::string::npos) 
		{
			result += til_result.substr(pos);
			break;
		}

		result += til_result.substr(pos, dollar - pos);
		++dollar; // skip '$'

		// Handle ${VAR} or $VAR
		bool braced = (dollar < til_result.length() && til_result[dollar] == '{');
		std::string::size_type start = dollar + (braced ? 1 : 0);
		std::string::size_type end = start;

		while (end < til_result.length() &&
				((til_result[end] >= 'A' && til_result[end] <= 'Z') ||
				 (til_result[end] >= '0' && til_result[end] <= '9') ||
				 til_result[end] == '_')) 
		{
			++end;
		}

		std::string varName = til_result.substr(start, end - start);
		const char* envValue = std::getenv(varName.c_str());

		if (envValue) 
		{
			result += envValue;
		} 
		else 
		{
			// Optionally: leave unresolved var as-is or throw
			result += '$' + (braced ? '{' + varName + '}' : varName);
		}

		pos = end + (braced && end < til_result.length() && til_result[end] == '}' ? 1 : 0);
	}

	try 
	{
		return std::filesystem::canonical(result).string() + "/";
	} catch (const std::filesystem::filesystem_error& e) 
	{
		throw std::runtime_error("Invalid path after expansion: " + result);
	}
}



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
	std::string site;
	std::string parfile, ehemeris;
	int verbose;
	double buf_size;

	mode = config["general"]["mode"].as<std::string>();
	input_dir = config["general"]["input_dir"].as<std::string>() + "/";
	output_dir = config["general"]["output_dir"].as<std::string>() + "/";
	site = config["general"]["site"].as<std::string>(); 
	verbose = config["general"]["verbose"].as<int>(); 

	if (config["general"]["buf_size"] && !config["general"]["buf_size"].IsNull())
		buf_size = config["general"]["buf_size"].as<double>(); 
	else
		buf_size = 2.0;

	input_dir = resolve_path(input_dir);
	output_dir = resolve_path(output_dir);


	std::string format = "IAA_vdif";


	for (const auto& filename_yaml : config["files"]) 
	{

		filename = filename_yaml.as<std::string>();

		Profile profile(input_dir + filename, format, 
				size_t(buf_size * 1024 * 1024 * 1024)); 

		BaseHeader* hdr = profile.getHeader();
		if (!hdr) 
			throw std::runtime_error("Header not available");

		if (config["advanced"])
		{
			for (const auto& kv : config["advanced"]) 
			{
				const std::string key = kv.first.as<std::string>();
				const std::string value = kv.second.as<std::string>();

				hdr-> update_header(key, value);
			}
		}


		hdr->print();


		if (config["general"]["t0"] && !config["general"]["t0"].IsNull())
			profile.reader->skip(config["general"]["t0"].as<double>());

		if (config["general"]["t1"] && !config["general"]["t1"].IsNull())
			profile.reader->set_limit(config["general"]["t1"].as<double>());



		// Find redshift correction if the parameter file is available
		if (config["general"]["parfile"] && !config["general"]["parfile"].IsNull())
		{
			parfile = input_dir + config["general"]["parfile"].as<std::string>();
			profile.get_redshift(parfile, site);
		}


		if (mode == "dedisperse")
		{
			size_t nchann = config["options"]["nchann"].as<size_t>();
			std::string mask_file = "";

			if (config["options"] && config["options"]["mask"]) 
				mask_file = config["options"]["mask"].as<std::string>();


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
							nchann);
				}
				else 
				{
					profile.fold_dyn(hdr->period, nchann);
				}

				std::ofstream raw_out (output_dir + filename + "_dyn.bin");
				raw_out.write(reinterpret_cast<const char*>(profile.dyn),
						hdr->nchann * hdr->obs_window * sizeof(double));
				raw_out.close();
				
				if (config["options"]["ddtype"].as<std::string>() == "incoherent")
				{
					profile.dedisperse_incoherent(hdr->dm, nchann);

					std::ofstream dd_out (output_dir + filename + "_dd.bin");
					dd_out.write(reinterpret_cast<const char*>(profile.dyn),
							hdr->nchann * hdr->obs_window * sizeof(double));
					dd_out.close();
				}
				else if (config["options"]["ddtype"].as<std::string>() == "coherent")
				{
					profile.dedisperse_coherent(hdr->dm, nchann);
				}
				else
					throw("Unknown type of de-dispersion: " + config["options"]["ddtype"].as<std::string>());
			}
			else
			{
				if (config["options"]["ddtype"].as<std::string>() == "incoherent")
				{
					profile.dedisperse_incoherent_stream (hdr->dm, nchann);
				}
				else if (config["options"]["ddtype"].as<std::string>() == "coherent")
				{
					profile.dedisperse_coherent_stream (hdr->dm, nchann);
				}
				else
					throw("Unknown type of de-dispersion: " + config["options"]["ddtype"].as<std::string>());
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
