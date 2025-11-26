// main.cpp
#include <iostream>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <cctype> // For the C-style isspace


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

std::string get_format(const std::string &filename)
{
	if (filename.length() >= 3 && 
			filename.substr(filename.length() - 3) == "adc") 
	{
		return "PRAO_adc";
	}
	if (filename.length() >= 5 && 
			filename.substr(filename.length() - 5) == ".vdif") {
		return "IAA_vdif";
	}

	return "Unknown";
}

template<typename T>
void read_key(const std::string& key, T* value, const YAML::Node& config, T* def = nullptr)
{
	if (config && config[key] && !config[key].IsNull())
		*value = config[key].as<T>();
	else if (def != nullptr)
		*value = *def;
	else
		throw std::invalid_argument("There is no key: " + key + " in the provided configuration");
}

void load_mask(Profile &profile, const YAML::Node &config)
{
	std::string mask_file = "";
	double std_threshold, tail_threshold;
	size_t nchann = 0;
	size_t max_len = 0;
	bool filter = false;

	read_key<size_t>("nchann", &nchann, config["options"]);
	read_key<size_t>("max_len", &max_len, config["options"], &max_len);
	read_key<std::string>("mask", &mask_file, config["options"], &mask_file);
	read_key<bool>("filter", &filter, config["options"], &filter);


	if (mask_file == "" && filter)
	{
		read_key<double>("tail_threshold", &tail_threshold, config["options"]);
		read_key<double>("std_threshold",   &std_threshold, config["options"]);

		profile.create_mask(nchann, std_threshold, tail_threshold, max_len);
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


	
	// Setting general working configuration
	// for the run
	std::string mode;
	std::string input_dir;
	std::string output_dir;
	std::string filename;
	std::string site;
	std::string parfile;
	int verbose;
	double buf_size;
	double t0, t1;
	bool save_raw, save_dyn, save_sum;

	if (!config["general"])
		throw std::invalid_argument("The \"general\" section must present in the configuration file");
	if (!config["options"])
		throw std::invalid_argument("The \"options\" section must present in the configuration file");
	if (!config["files"])
		throw std::invalid_argument("The \"files\" section must present in the configuration file");

	read_key<std::string>("mode", &mode, config["general"]);

	std::string def_path = ".";
	read_key<std::string>("input_dir", &input_dir, config["general"], &def_path);
	read_key<std::string>("output_dir", &output_dir, config["general"], &def_path);
	input_dir = input_dir + "/";
	output_dir = output_dir + "/";
	input_dir = resolve_path(input_dir);
	output_dir = resolve_path(output_dir);


	read_key<std::string>("site", &site, config["general"]);
	read_key<int>("verbose", &verbose, config["general"], new int(1));

	t0 = -1.0;
	t1 = -1.0;
	read_key("t0", &t0, config["general"], &t0);
	read_key("t1", &t1, config["general"], &t1);

	parfile = "";
	read_key<std::string>("parfile", &parfile, config["general"], &parfile);



	// Pre-processing mode variable
	// convert the string to lowercase
	std::transform(mode.begin(), mode.end(), mode.begin(),
			[](unsigned char c){ return std::tolower(c); });
	// strip whites
	mode.erase(std::remove_if(mode.begin(), mode.end(), ::isspace), mode.end());


	read_key<double>("buf_size", &buf_size, config["general"], new double(2.0));



	std::string format = "Unknown";
	save_raw = false;
	save_dyn = false;
	save_sum = false;

	read_key<bool>("save_raw", &save_raw, config["general"], &save_raw);
	read_key<bool>("save_dyn", &save_dyn, config["general"], &save_dyn);
	read_key<bool>("save_sum", &save_sum, config["general"], &save_sum);


	// Going through all provided files
	for (const auto& filename_yaml : config["files"]) 
	{

		filename = filename_yaml.as<std::string>();
		format = get_format(filename);


		Profile profile(input_dir + filename, format, 
				size_t(buf_size * 1024 * 1024 * 1024),
				save_raw, save_dyn, save_sum, output_dir); 
		BaseHeader* hdr = profile.getHeader();



		// Apply specific corrections to the file header
		if (config["advanced"])
		{
			for (const auto& kv : config["advanced"]) 
			{
				const std::string key = kv.first.as<std::string>();
				const std::string value = kv.second.as<std::string>();

				hdr -> update_header(key, value);
			}
		}


		if (verbose > 0)
			hdr->print();

		// Cutting a piece from the file if needed
		if (t0 > 0.0)
			profile.reader->skip(t0);

		if (t1 > 0.0)
			profile.reader->set_limit(t1);



		// Find redshift correction if the parameter file is available
		// if not, the redshift is defaulted to zero
		if (parfile != "")
		{
			parfile = input_dir + config["general"]["parfile"].as<std::string>();
			profile.get_redshift(parfile, site);
		}


		if (mode == "fold")
		{
			// Parse mode-specific options
			size_t nchann = 0;
			std::string t2_pred_file = "";
			std::string ddtype = "";


			read_key<size_t>("nchann", &nchann, config["options"], &nchann);
			read_key<std::string>("t2pred", &t2_pred_file, config["options"], &t2_pred_file);
			read_key<std::string>("ddtype", &ddtype, config["options"], &ddtype);

			load_mask(profile, config);

			if (ddtype == "incoherent")
			{

				if (t2_pred_file != "")
				{
					profile.fold_dyn(input_dir + t2_pred_file, nchann);
				}
				else 
				{
					profile.fold_dyn(hdr->period, nchann);
				}

				profile.dedisperse_incoherent(hdr->dm, nchann);
			}
			else if (ddtype == "coherent")
			{
				profile.dedisperse_coherent(hdr->dm, nchann);
			}
			else
			{
				throw("Unknown type of de-dispersion: " + config["options"]["ddtype"].as<std::string>());
			}

		}
		else if (mode == "stream")
		{
			size_t nchann = 0;
			std::string ddtype = "";


			read_key<size_t>("nchann", &nchann, config["options"], &nchann);
			read_key<std::string>("ddtype", &ddtype, config["options"], &ddtype);

			load_mask(profile, config);

			if (ddtype == "incoherent")
			{
				profile.dedisperse_incoherent_stream(hdr->dm, nchann);
			}
			else if (ddtype == "coherent")
			{
				profile.dedisperse_coherent_stream(hdr->dm, nchann);
			}
			else
			{
				throw("Unknown type of de-dispersion: " + config["options"]["ddtype"].as<std::string>());
			}

		}
		else if (mode == "stream")
		{
			size_t nchann = 0;
			std::string ddtype = "";


			read_key<size_t>("nchann", &nchann, config["options"], &nchann);
			read_key<std::string>("ddtype", &ddtype, config["options"], &ddtype);

			load_mask(profile, config);
		}
		else if (mode == "search")
		{
			// Parse mode-specific options
			size_t nchann = 0;
			std::string ddtype = "";


			read_key<size_t>("nchann", &nchann, config["options"], &nchann);
			read_key<std::string>("ddtype", &ddtype, config["options"], &ddtype);

			load_mask(profile, config);

			if (ddtype == "incoherent")
			{
				profile.dedisperse_incoherent_search(hdr->dm, nchann);
			}
			else if (ddtype == "coherent")
			{
			}
			else
			{
				throw("Unknown type of de-dispersion: " + config["options"]["ddtype"].as<std::string>());
			}
		}
		else
		{
			throw("Unknown processing mode: " + mode);
		}



	}


	return 0;
}
