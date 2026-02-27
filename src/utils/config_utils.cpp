#include "config_utils.h"

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

//=============================================================================
// PATH RESOLUTION UTILITIES
//=============================================================================

/**
 * Expands environment variables and '~' in paths, returns canonical path
 * (keeps the original algorithm, but uses fs::path for the final step)
 */
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
			result += '$' + (braced ? '{' + varName + '}' : varName);
		}

		pos = end + (braced && end < til_result.length() && til_result[end] == '}' ? 1 : 0);
	}

	try
	{
		// Use fs::path to handle canonical and ensure a trailing separator
		fs::path p = fs::canonical(result);
		return (p / "").string();  // appends preferred separator
	}
	catch (const std::filesystem::filesystem_error& e)
	{
		throw std::runtime_error("Invalid path after expansion: " + result);
	}
}




//=============================================================================
// FILE FORMAT DETECTION
//=============================================================================

/**
 * Determines file format based on extension (or special patterns)
 */
std::string get_format(const std::string& filename)
{
	std::regex prao_psr_pattern(R"(^[0-9]{6}_.*_[0-9]{2}$)");

	fs::path p(filename);
	std::string name = p.filename().string();      // pure filename, no directories
	std::string ext  = p.extension().string();     // includes the dot, e.g. ".fits"

	// Check for special suffix "adc" (without dot)
	if (name.size() >= 3 && name.substr(name.size() - 3) == "adc")
	{
		return "PRAO_adc";
	}
	if (std::regex_match(name, prao_psr_pattern))
	{
		return "PRAO_psr";
	}
	if (ext == ".lpa3")
	{
		return "PRAO_lpa3";
	}
	if (ext == ".vdif")
	{
		return "IAA_vdif";
	}
	if (ext == ".fits")
	{
		return "PSRFITS";
	}
	return "Unknown";
}

//=============================================================================
// YAML CONFIGURATION READING
//=============================================================================

/**
 * Reads a key from YAML config with optional default value
 */
	template<typename T>
void read_key(const std::string& key, T* value, const YAML::Node& config, T* def)
{
	if (config && config[key] && !config[key].IsNull())
		*value = config[key].as<T>();
	else if (def != nullptr)
		*value = *def;
	else
		throw std::invalid_argument("Missing required key: " + key);
}
// Basic scalar types
template void read_key<double>(const std::string&, double*, const YAML::Node&, double*);
template void read_key<float>(const std::string&, float*, const YAML::Node&, float*);
template void read_key<bool>(const std::string&, bool*, const YAML::Node&, bool*);
template void read_key<std::string>(const std::string&, std::string*, const YAML::Node&, std::string*);

// Integer types
template void read_key<int16_t>(const std::string&, int16_t*, const YAML::Node&, int16_t*);
template void read_key<uint16_t>(const std::string&, uint16_t*, const YAML::Node&, uint16_t*);
template void read_key<int32_t>(const std::string&, int32_t*, const YAML::Node&, int32_t*);
template void read_key<uint32_t>(const std::string&, uint32_t*, const YAML::Node&, uint32_t*);
template void read_key<int64_t>(const std::string&, int64_t*, const YAML::Node&, int64_t*);
template void read_key<uint64_t>(const std::string&, uint64_t*, const YAML::Node&, uint64_t*);

/**
 * Loads or generates a channel mask for RFI mitigation
 */
void load_mask(Profile& profile, const YAML::Node& config)
{
	std::string mask_file = "";
	double std_threshold, tail_threshold;
	size_t nchann = profile.hdr->nchann;
	size_t downsample = 0;
	size_t max_len = 0;
	bool filter = false;

	read_key<std::string>("mask_file", &mask_file, config["options"], &mask_file);
	read_key<size_t>("nchann", &nchann, config["options"], &nchann);
	read_key<size_t>("max_len", &max_len, config["options"], &max_len);
	read_key<size_t>("downsample", &downsample, config["options"], &downsample);
	read_key<bool>("filter", &filter, config["options"], &filter);

	if (!filter)
		return;

	if (nchann == 0)
		throw std::runtime_error("The number of channels is unknown. Can not perform filtration");

	if (mask_file == "" && filter)
	{
		read_key<double>("tail_threshold", &tail_threshold, config["options"]);
		read_key<double>("std_threshold", &std_threshold, config["options"]);

		if (downsample > 0)
			profile.create_mask(nchann, std_threshold, tail_threshold, max_len, downsample);
		else
			profile.create_mask(nchann, std_threshold, tail_threshold, max_len);
	}
	else if (mask_file != "")
	{
		Profile wts_prf(mask_file, "PSRFITS", 0);
		wts_prf.load_mask(nchann);

		profile.mask = new double[nchann];
		profile.fr = new double[nchann];
		math::vec_copy(profile.mask, wts_prf.mask, nchann);
		math::vec_copy(profile.fr, wts_prf.fr, nchann);
	}
}

/**
 * Applies header corrections from advanced config section
 */
void apply_header_corrections(BaseHeader* hdr, const YAML::Node& config)
{
	if (config["advanced"])
	{
		for (const auto& kv : config["advanced"])
		{
			const std::string key = kv.first.as<std::string>();
			const std::string value = kv.second.as<std::string>();
			hdr->update_header(key, value);
		}
	}
}


/**
 * Sets up time limits for data processing
 */
void setup_time_limits(Profile& profile, double t0, double t1)
{
	if (t0 > 0.0)
		profile.skip(t0);
	if (t1 > 0.0)
		profile.set_limit(t1);
}
