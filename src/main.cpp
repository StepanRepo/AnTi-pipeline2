// main.cpp
#include <cstddef>
#include <cstring>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <cctype>
#include <ctime>

#include "Profile.h"
#include "aux_math.h"

namespace fs = std::filesystem;

//=============================================================================
// PATH RESOLUTION UTILITIES
//=============================================================================

/**
 * Expands environment variables and '~' in paths, returns canonical path
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
		return std::filesystem::canonical(result).string() + "/";
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
 * Determines file format based on extension
 */
std::string get_format(const std::string& filename)
{
	if (filename.length() >= 3 && 
			filename.substr(filename.length() - 3) == "adc") 
	{
		return "PRAO_adc";
	}
	if (filename.length() >= 5 && 
			filename.substr(filename.length() - 5) == ".lpa3") 
	{
		return "PRAO_lpa3";
	}
	if (filename.length() >= 5 && 
			filename.substr(filename.length() - 5) == ".vdif") 
	{
		return "IAA_vdif";
	}
	if (filename.length() >= 5 && 
			filename.substr(filename.length() - 5) == ".fits") 
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
void read_key(const std::string& key, T* value, const YAML::Node& config, T* def = nullptr)
{
	if (config && config[key] && !config[key].IsNull())
		*value = config[key].as<T>();
	else if (def != nullptr)
		*value = *def;
	else
		throw std::invalid_argument("Missing required key: " + key);
}

/**
 * Loads or generates a channel mask for RFI mitigation
 */
void load_mask(Profile& profile, const YAML::Node& config)
{
	std::string mask_file = "";
	double std_threshold, tail_threshold;
	size_t nchann = 0;
	size_t downsample = 0;
	size_t max_len = 0;
	bool filter = false;

	read_key<std::string>("mask_file", &mask_file, config["options"], &mask_file);
	read_key<size_t>("nchann", &nchann, config["options"]);
	read_key<size_t>("max_len", &max_len, config["options"], &max_len);
	read_key<size_t>("downsample", &downsample, config["options"], &downsample);
	read_key<bool>("filter", &filter, config["options"], &filter);

	if (!filter)
		return;

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
		math::vec_copy(profile.mask, wts_prf.mask, nchann);
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

//=============================================================================
// PROCESSING MODES
//=============================================================================

/**
 * MODE: fold
 * - Creates folded profiles
 * - Applies de-dispersion (coherent or incoherent)
 * - Uses T2 prediction if available
 */
void run_fold_mode(
		const YAML::Node& config,
		const std::string& input_dir,
		const std::string& output_dir,
		size_t buf_size,
		bool save_raw,
		bool save_dyn,
		bool save_sum,
		const std::string& site,
		const std::string& parfile,
		double t0,
		double t1,
		int verbose)
{
	std::cout << "\n=== FOLD MODE ===\n";

	// Parse mode-specific options
	size_t nchann = 0;
	std::string t2_pred_file = "";
	std::string ddtype = "";

	read_key<size_t>("nchann", &nchann, config["options"], &nchann);
	read_key<std::string>("t2pred", &t2_pred_file, config["options"], &t2_pred_file);
	read_key<std::string>("ddtype", &ddtype, config["options"], &ddtype);

	// Process each file
	for (const auto& filename_yaml : config["files"]) 
	{
		std::string filename = filename_yaml.as<std::string>();
		std::string format = get_format(filename);

		std::cout << "\nProcessing: " << filename << std::endl;

		// Initialize profile
		Profile profile(input_dir + filename, format, 
				buf_size, save_raw, save_dyn, save_sum, output_dir, verbose);
		BaseHeader* hdr = profile.getHeader();

		// Apply configurations
		apply_header_corrections(hdr, config);
		setup_time_limits(profile, t0, t1);

		if (verbose > 0)
			hdr->print();

		// Get redshift correction if available
		if (parfile != "")
		{
			std::string full_parfile = input_dir + parfile;
			profile.get_redshift(full_parfile, site);
		}

		// Load or generate RFI mask
		load_mask(profile, config);

		// Execute folding based on de-dispersion type
		if (ddtype == "incoherent")
		{
			if (t2_pred_file != "")
				profile.fold_dyn(input_dir + t2_pred_file, nchann);
			else 
				profile.fold_dyn(hdr->period, nchann);

			profile.dedisperse_incoherent(hdr->dm, nchann);
		}
		else if (ddtype == "coherent")
		{
			profile.dedisperse_coherent(hdr->dm, nchann);
		}
		else
		{
			throw std::runtime_error("Unknown de-dispersion type: " + ddtype);
		}
	}
}

/**
 * MODE: stream
 * - Real-time streaming de-dispersion
 */
void run_stream_mode(
		const YAML::Node& config,
		const std::string& input_dir,
		const std::string& output_dir,
		size_t buf_size,
		bool save_raw,
		bool save_dyn,
		bool save_sum,
		const std::string& site,
		const std::string& parfile,
		double t0,
		double t1,
		int verbose)
{
	if (! (save_raw || save_dyn || save_sum))
		return;
	std::cout << "\n=== STREAM MODE ===\n";

	// Parse mode-specific options
	size_t nchann = 0;
	std::string ddtype = "";

	read_key<size_t>("nchann", &nchann, config["options"], &nchann);
	read_key<std::string>("ddtype", &ddtype, config["options"], &ddtype);

	// Process each file
	for (const auto& filename_yaml : config["files"]) 
	{
		std::string filename = filename_yaml.as<std::string>();
		std::string format = get_format(filename);

		std::cout << "\nProcessing: " << filename << std::endl;

		// Initialize profile
		Profile profile(input_dir + filename, format, 
				buf_size, save_raw, save_dyn, save_sum, output_dir, verbose);
		BaseHeader* hdr = profile.getHeader();

		// Apply configurations
		apply_header_corrections(hdr, config);
		setup_time_limits(profile, t0, t1);

		if (verbose > 0)
			hdr->print();

		// Get redshift correction if available
		if (parfile != "")
		{
			std::string full_parfile = input_dir + parfile;
			profile.get_redshift(full_parfile, site);
		}

		// Load or generate RFI mask
		load_mask(profile, config);

		// Execute streaming de-dispersion
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
			throw std::runtime_error("Unknown de-dispersion type: " + ddtype);
		}
	}
}


/**
 * MODE: search
 * - Searches for pulses/candidates
 * - Saves results to CSV
 */

//==============================
// PULSE TRIMMING FUNCTIONALITY
//==============================

/**
 * Trims input files around detected pulses and centers them in the output window
 * */
void trim_pulses(
    const std::string& pulse_csv,
    double window_width,
    const YAML::Node& config,
    const std::string& input_dir,
    const std::string& output_dir,
    size_t buf_size,
    bool save_raw,
    bool save_dyn,
    bool save_sum,
    const std::string& site,
    const std::string& parfile,
    int verbose
	)
{
	if (! (save_raw || save_dyn || save_sum))
		return;
    //=========================================================================
    // READ PULSE CATALOG
    //=========================================================================
    std::ifstream catalog(pulse_csv);
    if (!catalog.is_open()) {
        throw std::runtime_error("Cannot open pulse catalog: " + pulse_csv);
    }
    
    // Structure to hold pulse information with ID
    struct PulseInfo {
        std::string source_file;
        double time;
        int id;
    };
    
    std::vector<PulseInfo> pulses;
    std::string line;
    
    // Skip all comment lines
    while (catalog.peek() == '#') 
    {
        std::getline(catalog, line);
    }
    
    // Read header line to understand column order
    std::getline(catalog, line);
    
    // Parse column headers to find time and ID columns
    std::stringstream header_ss(line);
    std::string header_token;
    std::vector<std::string> headers;
    while (std::getline(header_ss, header_token, ';')) {
        headers.push_back(header_token);
    }
    
    // Find indices of important columns
    int time_col = -1;
    int id_col = -1;
    int file_col = -1;  // Assume first column is filename
    
    for (size_t i = 0; i < headers.size(); i++) {
        if (headers[i] == "time_s" || headers[i] == "time" || headers[i] == "sec") 
            time_col = i;
        if (headers[i] == "pulse_id" || headers[i] == "id" || headers[i] == "ID") 
            id_col = i;
        if (headers[i] == "file" || headers[i] == "source") 
            file_col = i;
    }

    // If we couldn't identify columns, assume default positions
    if (time_col == -1) time_col = 3;  // Default to 4th column (0-indexed)
    if (id_col == -1)   id_col = 1;    // Default to 1st column
    if (file_col == -1) file_col = 0;  // Default to 0th column
    
    // Parse pulses
    while (std::getline(catalog, line)) 
    {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, ';')) 
        {
            tokens.push_back(token);
        }
        
        if (tokens.size() < 6) {
            std::cerr << "Warning: Skipping line with insufficient fields: " << line << std::endl;
            continue;
        }

        try 
        {
            PulseInfo pulse;
            pulse.source_file = tokens[file_col];  // First column is source file
            pulse.time = std::stod(tokens[time_col]);
            pulse.id = (size_t(id_col) < tokens.size()) ? std::stoi(tokens[id_col]) : pulses.size() + 1;
            
            pulses.push_back(pulse);
        } 
        catch (const std::exception& e) 
        {
            std::cerr << "Warning: Failed to parse line: " << line << std::endl;
        }
    }
    
    //=============================
    // GROUP PULSES BY SOURCE FILE
    //==============================
    std::map<std::string, std::vector<std::pair<double, int>>> file_pulses;

    for (const auto& pulse : pulses) {
        file_pulses[pulse.source_file].push_back(std::make_pair(pulse.time, pulse.id));
    }
    
    //=======================================
    // PROCESS EACH FILE AND TRIM AROUND PULSES
    //========================================
    for (const auto& entry : file_pulses) 
    {
        const std::string& filename = entry.first;
        const auto& pulse_list = entry.second;
        
        // Get base filename without extension
        fs::path file_path(input_dir+filename);
        std::string base_name = file_path.stem().string();
        std::string format = get_format(filename);
        
        for (const auto& pulse_data : pulse_list) 
        {
            double pulse_time = pulse_data.first;
            int pulse_id = pulse_data.second;

            
            // Calculate window boundaries
            double window_start = pulse_time - window_width / 2.0;
            double window_end = pulse_time + window_width / 2.0;

            // Ensure non-negative start time
            if (window_start < 0.0) 
            {
                window_start = 0.0;
                window_end = window_width;
            }
            
            
            //============================================
            // MODIFY CONFIGURATION FOR THIS PULSE
            //============================================
            
            // Create a temporary config node for this pulse
            YAML::Node pulse_config = YAML::Clone(config);
		
			// If there is no mask file in the configuration	
			if (! (config["options"]["mask_file"] &&
						!config["options"]["mask_file"].IsNull()))
			{
				pulse_config["options"]["mask_file"] = output_dir + "wts_" + filename + ".fits";
			}

            // Create a temporary config with just this file
            YAML::Node temp_files;
            temp_files.push_back(filename);
            pulse_config["files"] = temp_files;

            
            //=====================================
            // RUN STREAM MODE ON THIS PULSE WINDOW
            //======================================
            try {
                run_stream_mode(
                    pulse_config,
                    input_dir,
					output_dir,
                    buf_size,
                    save_raw,
                    save_dyn,
                    save_sum,
                    site,
                    parfile,
                    window_start,
                    window_end,
                    verbose
                );

				if (save_raw)
					fs::rename(
							output_dir+"raw_"+filename+".fits",
							output_dir+"raw_"+filename+"_id"+std::to_string(pulse_id)+".fits");

				if (save_dyn)
					fs::rename(
							output_dir+"dyn_"+filename+".fits",
							output_dir+"dyn_"+filename+"_id"+std::to_string(pulse_id)+".fits");

				if (save_sum)
					fs::rename(
							output_dir+"sum_"+filename+".fits",
							output_dir+"sum_"+filename+"_id"+std::to_string(pulse_id)+".fits");
                
            } 
			catch (const std::exception& e) {
                std::cerr << "  ERROR processing pulse ID " << pulse_id 
                          << ": " << e.what() << std::endl;
            }
        }
    }
}


std::string collect_search_results(
		const std::string& output_dir,
		const std::vector<std::string>& csv_files,
		const double dm,
		const size_t nchann,
		const std::string ddtype,
		const std::string conv_type,
		const double bl_window,
		const double fwhm,
		const double threshold
		)
{
    if (csv_files.empty()) 
	{
        std::cout << "No CSV files to collect." << std::endl;
        return "";
    }
    // Generate output filename with timestamp
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%Y.%m.%dT%H:%M:%S", std::localtime(&now_time));

    std::string collection_file = output_dir + "search_results_" + timestamp + ".csv";
    std::ofstream outfile(collection_file);

    if (!outfile.is_open()) {
        throw std::runtime_error("Cannot create collection file: " + collection_file);
    }



    // Write metadata header as comments
	outfile << "# ===== Search Results =====" << std::endl;
    outfile << "# Generated: " << timestamp << "\n";
    outfile << "# Processing parameters:\n";
    outfile << "#   DM:              " << dm << " pc cm-3\n";
    outfile << "#   Channels:        " << nchann << "\n";
    outfile << "#   De-dispersion:   " << ddtype << "\n";
    outfile << "#   Convolution:     " << conv_type << "\n";
    outfile << "#   Baseline window: " << bl_window << " s\n";
    outfile << "#   FWHM:            " << fwhm << " s\n";
    outfile << "#   Threshold:       " << threshold << " sigma\n";
    outfile << "# \n";

    // Write column headers (take from first file)
    bool headers_written = false;

    for (const auto& csv_file : csv_files) 
	{
        std::ifstream infile(csv_file);

        if (!infile.is_open()) {
            std::cerr << "Warning: Cannot open " << csv_file << " for reading" << std::endl;
            continue;
        }

        std::string line;



		size_t line_num = 0;
        // Write data rows with source filename
        while (std::getline(infile, line)) 
		{
			if (!line.empty() && line[0] == '#')  // Check first character
				continue;  // Skip comment lines
			
			line_num += 1;


			if (line_num == 1) 
			{
				if (!headers_written)
				{
					outfile << line << "\n";
					headers_written = true;
				}

				continue;
			}

			outfile << line << "\n";
        }

		fs::remove(csv_file);
    }

    outfile.close();
	return collection_file;
}


void run_search_mode(
		const YAML::Node& config,
		const std::string& input_dir,
		const std::string& output_dir,
		size_t buf_size,
		bool save_raw,
		bool save_dyn,
		bool save_sum,
		const std::string& site,
		const std::string& parfile,
		double t0,
		double t1,
		int verbose)
{
	std::cout << "\n=== SEARCH MODE ===\n";

	// Parse mode-specific options
	size_t nchann = 0;
	double bl_window;
	double threshold;
	double fwhm;
	std::string ddtype = "";
	std::string conv_type = "";

	read_key<size_t>("nchann", &nchann, config["options"], &nchann);
	read_key<std::string>("ddtype", &ddtype, config["options"], &ddtype);
	read_key<std::string>("conv_type", &conv_type, config["options"], &conv_type);
	read_key<double>("bl_window", &bl_window, config["options"]);
	read_key<double>("search_threshold", &threshold, config["options"]);
	read_key<double>("search_fwhm", &fwhm, config["options"]);


    std::vector<std::string> search_results;

	double dm = 0.0;
	// Process each file
	for (const auto& filename_yaml : config["files"]) 
	{
		std::string filename = filename_yaml.as<std::string>();
		std::string format = get_format(filename);
		std::string new_filename = filename + ".csv";

		std::cout << "\nProcessing: " << filename << std::endl;

		// Initialize profile
		Profile profile(input_dir + filename, format, 
				buf_size, save_raw, save_dyn, save_sum, output_dir, verbose);
		BaseHeader* hdr = profile.getHeader();

		// Apply configurations
		apply_header_corrections(hdr, config);
		setup_time_limits(profile, t0, t1);

		if (verbose > 0)
			hdr->print();

		// Get redshift correction if available
		if (parfile != "")
		{
			std::string full_parfile = input_dir + parfile;
			profile.get_redshift(full_parfile, site);
		}

		// Load or generate RFI mask
		load_mask(profile, config);

		// Execute search based on de-dispersion type
		std::string temp_id;
		if (ddtype == "incoherent")
		{
			temp_id = profile.dedisperse_incoherent_search(
					hdr->dm, nchann, bl_window, threshold, conv_type, fwhm);
		}
		else if (ddtype == "coherent")
		{
			temp_id = profile.dedisperse_coherent_search(
					hdr->dm, nchann, bl_window, threshold, conv_type, fwhm);
		}
		else
		{
			throw std::runtime_error("Unknown de-dispersion type: " + ddtype);
		}

		// Rename output file
		if (!temp_id.empty())
		{
			fs::rename(output_dir + temp_id, output_dir + new_filename);
			search_results.push_back(output_dir + new_filename);
		}
		 dm = profile.hdr->dm;
	}

	std::string collection_file = collect_search_results(
			output_dir, 
			search_results, 
			dm,
			nchann,
			ddtype,
			conv_type,
			bl_window,
			fwhm,
			threshold
			);


	if (collection_file == "") return;

	if (! (save_raw || save_dyn || save_sum))
		return;

	trim_pulses(
			collection_file,
			2.0,
			config, 
			input_dir, 
			output_dir, 
			buf_size,
			save_raw, 
			save_dyn, 
			save_sum, 
			site, 
			parfile, 
			verbose);
}

//=============================================================================
// MAIN PROGRAM
//=============================================================================

YAML::Node config;

int main() 
{
	try 
	{
		//=========================================================================
		// LOAD CONFIGURATION
		//=========================================================================
		config = YAML::LoadFile("default.yaml");

		// Validate required sections
		if (!config["general"])
			throw std::invalid_argument("Missing 'general' section in config");
		if (!config["options"])
			throw std::invalid_argument("Missing 'options' section in config");
		if (!config["files"])
			throw std::invalid_argument("Missing 'files' section in config");

		//=========================================================================
		// READ GENERAL CONFIGURATION
		//=========================================================================
		std::string mode;
		std::string input_dir;
		std::string output_dir;
		std::string site;
		std::string parfile = "";
		int verbose = 1;
		double buf_size = 2.0;
		double t0 = -1.0;
		double t1 = -1.0;
		bool save_raw = false;
		bool save_dyn = false;
		bool save_sum = false;

		// Read general configuration values
		read_key<std::string>("mode", &mode, config["general"]);

		std::string def_path = ".";
		read_key<std::string>("input_dir", &input_dir, config["general"], &def_path);
		read_key<std::string>("output_dir", &output_dir, config["general"], &def_path);
		read_key<std::string>("site", &site, config["general"]);
		read_key<int>("verbose", &verbose, config["general"], &verbose);
		read_key<double>("buf_size", &buf_size, config["general"], &buf_size);
		read_key<double>("t0", &t0, config["general"], &t0);
		read_key<double>("t1", &t1, config["general"], &t1);
		read_key<std::string>("parfile", &parfile, config["general"], &parfile);
		read_key<bool>("save_raw", &save_raw, config["general"], &save_raw);
		read_key<bool>("save_dyn", &save_dyn, config["general"], &save_dyn);
		read_key<bool>("save_sum", &save_sum, config["general"], &save_sum);

		// Resolve paths
		input_dir = resolve_path(input_dir + "/");
		output_dir = resolve_path(output_dir + "/");

		//=========================================================================
		// NORMALIZE MODE STRING
		//=========================================================================
		std::transform(mode.begin(), mode.end(), mode.begin(),
				[](unsigned char c) { return std::tolower(c); });
		mode.erase(std::remove_if(mode.begin(), mode.end(), ::isspace), mode.end());

		//=========================================================================
		// DISPLAY CONFIGURATION
		//=========================================================================
		std::cout << "========================================\n";
		std::cout << "ANTI-PIPELINE 2\n";
		std::cout << "========================================\n";
		std::cout << "Mode:             " << mode << "\n";
		std::cout << "Input directory:  " << input_dir << "\n";
		std::cout << "Output directory: " << output_dir << "\n";
		std::cout << "Buffer size:      " << buf_size << " GB\n";
		std::cout << "Files to process: " << config["files"].size() << "\n";
		std::cout << "========================================\n";

		//=======================================
		// ROUTE TO APPROPRIATE PROCESSING MODE
		//=======================================
		size_t buf_size_bytes = static_cast<size_t>(buf_size * (1ULL << 30));

		if (mode == "fold")
		{
			run_fold_mode(config, input_dir, output_dir, buf_size_bytes,
					save_raw, save_dyn, save_sum, site, parfile, t0, t1, verbose);
		}
		else if (mode == "stream")
		{
			run_stream_mode(config, input_dir, output_dir, buf_size_bytes,
					save_raw, save_dyn, save_sum, site, parfile, t0, t1, verbose);
		}
		else if (mode == "search")
		{
			run_search_mode(config, input_dir, output_dir, buf_size_bytes,
					save_raw, save_dyn, save_sum, site, parfile, t0, t1, verbose);
		}
		else
		{
			throw std::runtime_error("Unknown processing mode: " + mode);
		}

		//==========
		// CLEANUP
		//==========
		math::cleanup();
		std::cout << "\nProcessing completed successfully.\n";
		return 0;
	}
	catch (const YAML::Exception& e) 
	{
		std::cerr << "\nYAML ERROR: " << e.what() << "\n";
		return 1;
	}
	catch (const std::exception& e)
	{
		std::cerr << "\nERROR: " << e.what() << "\n";
		return 1;
	}
	catch (...)
	{
		std::cerr << "\nUNKNOWN ERROR\n";
		return 1;
	}
}
