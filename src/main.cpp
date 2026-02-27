// main.cpp
#include <cstddef>
#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <cctype>
#include <ctime>

#include "Profile.h"
#include "aux_math.h"
#include "config_utils.h"

namespace fs = std::filesystem;




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

	read_key("nchann", &nchann, config["options"], &nchann);
	read_key("t2pred", &t2_pred_file, config["options"], &t2_pred_file);
	read_key("ddtype", &ddtype, config["options"], &ddtype);

	// Process each file
	for (const auto& filename_yaml : config["files"])
	{
		std::string filename = filename_yaml.as<std::string>();
		std::string format = get_format(filename);

		std::cout << "\nProcessing: " << filename << std::endl;

		// Build full input path using fs::path
		fs::path in_path = fs::path(input_dir) / filename;
		Profile profile(in_path.string(), format,
				buf_size, output_dir, verbose);
		BaseHeader* hdr = profile.getHeader();

		// Apply configurations
		apply_header_corrections(hdr, config);
		setup_time_limits(profile, t0, t1);

		if (verbose > 0)
			hdr->print();

		if (nchann == 0)
			nchann = hdr->nchann;

		// Get redshift correction if available
		if (parfile != "")
		{
			fs::path par_path = fs::path(input_dir) / parfile;
			profile.get_redshift(par_path.string(), site);
		}

		// Load or generate RFI mask
		load_mask(profile, config);

		// Execute folding based on de-dispersion type
		if (ddtype == "incoherent")
		{
			if (t2_pred_file != "")
			{
				fs::path t2_path = fs::path(input_dir) / t2_pred_file;
				profile.fold_dyn(t2_path.string(), nchann);
			}
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

		if (save_raw) profile.save_raw("PSR");
		if (save_dyn) profile.save_dyn("PSR");
		if (save_sum) profile.save_sum("PSR");
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
		int verbose,
		std::string mode = "SEARCH")
{
	if (! (save_raw || save_dyn || save_sum))
		return;
	std::cout << "\n=== STREAM MODE ===\n";

	// Parse mode-specific options
	size_t nchann = 0;
	std::string ddtype = "";

	read_key("nchann", &nchann, config["options"], &nchann);
	read_key("ddtype", &ddtype, config["options"], &ddtype);

	// Process each file
	for (const auto& filename_yaml : config["files"])
	{
		std::string filename = filename_yaml.as<std::string>();
		std::string format = get_format(filename);

		std::cout << "\nProcessing: " << filename << std::endl;

		// Build full input path
		fs::path in_path = fs::path(input_dir) / filename;
		Profile profile(in_path.string(), format,
				buf_size, output_dir, verbose);
		BaseHeader* hdr = profile.getHeader();

		// Apply configurations
		apply_header_corrections(hdr, config);
		setup_time_limits(profile, t0, t1);

		if (verbose > 0)
			hdr->print();


		if (nchann == 0)
			nchann = hdr->nchann;

		// Get redshift correction if available
		if (parfile != "")
		{
			fs::path par_path = fs::path(input_dir) / parfile;
			profile.get_redshift(par_path.string(), site);
		}

		// Load or generate RFI mask
		load_mask(profile, config);

		std::string id = "";
		// Execute streaming de-dispersion
		if (ddtype == "incoherent")
		{
			id = profile.dedisperse_incoherent_stream(
					hdr->dm, nchann,
					save_raw, save_dyn, save_sum
					);
		}
		else if (ddtype == "coherent")
		{
			id = profile.dedisperse_coherent_stream(
					hdr->dm, nchann,
					save_raw, save_dyn, save_sum
					);
		}
		else
		{
			throw std::runtime_error("Unknown de-dispersion type: " + ddtype);
		}

		if (id == "") continue;

		if (save_raw) profile.save_raw(mode, output_dir+"raw_" + id);
		if (save_dyn) profile.save_dyn(mode, output_dir+"dyn_" + id);
		if (save_sum) profile.save_sum(mode, output_dir+"sum_" + id);
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
    int verbose,
	std::string mode = "SEARCH"
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

        // Get base filename without extension (for display only)
        fs::path file_path(filename);
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
				// Build mask file path with fs::path
				fs::path mask_path = fs::path(output_dir) / ("wts_" + filename + ".fits");
				pulse_config["options"]["mask_file"] = mask_path.string();
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
                    verbose,
					mode
                );

				if (save_raw)
				{
					fs::path old_raw = fs::path(output_dir) / ("raw_" + filename + ".fits");
					fs::path new_raw = fs::path(output_dir) / ("raw_" + filename + "_id" + std::to_string(pulse_id) + ".fits");
					fs::rename(old_raw, new_raw);
				}

				if (save_dyn)
				{
					fs::path old_dyn = fs::path(output_dir) / ("dyn_" + filename + ".fits");
					fs::path new_dyn = fs::path(output_dir) / ("dyn_" + filename + "_id" + std::to_string(pulse_id) + ".fits");
					fs::rename(old_dyn, new_dyn);
				}

				if (save_sum)
				{
					fs::path old_sum = fs::path(output_dir) / ("sum_" + filename + ".fits");
					fs::path new_sum = fs::path(output_dir) / ("sum_" + filename + "_id" + std::to_string(pulse_id) + ".fits");
					fs::rename(old_sum, new_sum);
				}

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

    fs::path collection_path = fs::path(output_dir) / ("search_results_" + std::string(timestamp) + ".csv");
    std::ofstream outfile(collection_path.string());

    if (!outfile.is_open()) {
        throw std::runtime_error("Cannot create collection file: " + collection_path.string());
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
	return collection_path.string();
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

	read_key("nchann", &nchann, config["options"], &nchann);
	read_key("ddtype", &ddtype, config["options"], &ddtype);
	read_key("conv_type", &conv_type, config["options"], &conv_type);
	read_key("bl_window", &bl_window, config["options"]);
	read_key("search_threshold", &threshold, config["options"]);
	read_key("search_fwhm", &fwhm, config["options"]);


    std::vector<std::string> search_results;

	double dm = 0.0;
	// Process each file
	for (const auto& filename_yaml : config["files"])
	{
		std::string filename = filename_yaml.as<std::string>();
		std::string format = get_format(filename);
		// Output CSV name: original filename + ".csv"
		std::string new_filename = filename + ".csv";

		std::cout << "\nProcessing: " << filename << std::endl;

		// Build full input path
		fs::path in_path = fs::path(input_dir) / filename;
		Profile profile(in_path.string(), format,
				buf_size, output_dir, verbose);
		BaseHeader* hdr = profile.getHeader();

		// Apply configurations
		apply_header_corrections(hdr, config);
		setup_time_limits(profile, t0, t1);

		if (verbose > 0)
			hdr->print();

		if (nchann == 0)
			nchann = hdr->nchann;

		// Get redshift correction if available
		if (parfile != "")
		{
			fs::path par_path = fs::path(input_dir) / parfile;
			profile.get_redshift(par_path.string(), site);
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
			fs::path old_path = fs::path(output_dir) / temp_id;
			fs::path new_path = fs::path(output_dir) / new_filename;
			fs::rename(old_path, new_path);
			search_results.push_back(new_path.string());
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

	double trim_window = -1.0;
	read_key("trim_window", &trim_window, config["options"], &trim_window);

	if (trim_window < 0.0)
		return;


	trim_pulses(
			collection_file,
			trim_window,
			config,
			input_dir,
			output_dir,
			buf_size,
			save_raw,
			save_dyn,
			save_sum,
			site,
			parfile,
			verbose,
			"PSR");
}



/**
 * MODE: template
 * - Produses a template pulse from a series of files
 */
void run_template_mode(
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
	std::cout << "\n=== TEMPLATE MODE ===\n";

	if (config["files"].size() < 2)
		throw std::runtime_error("Provide at least 2 files to average");

	std::string t2_pred_file = "";
	std::string ddtype = "";

	read_key("t2pred", &t2_pred_file, config["options"], &t2_pred_file);
	read_key("ddtype", &ddtype, config["options"], &ddtype);


	const auto filename_yaml = config["files"][0];
	std::string filename = filename_yaml.as<std::string>();
	std::string format = get_format(filename);

	if (verbose > 0)
		std::cout << "\nReferance file: " << filename << std::endl;

	// Build full input path
	fs::path in_path = fs::path(input_dir) / filename;
	Profile ref_prf(in_path.string(), format,
			buf_size, output_dir, verbose);
	BaseHeader* hdr = ref_prf.getHeader();

	// Apply configurations
	apply_header_corrections(hdr, config);
	setup_time_limits(ref_prf, t0, t1);


	// Get redshift correction if available
	if (parfile != "")
	{
		fs::path par_path = fs::path(input_dir) / parfile;
		ref_prf.get_redshift(par_path.string(), site);
	}

	// Load or generate RFI mask
	load_mask(ref_prf, config);

	if (verbose > 0)
		hdr->print();

	bool dd = ! (hdr->dds_mthd == "" || hdr->dds_mthd == "none");
	bool avg = (hdr->nchann == 1);

	if (hdr->MODE == "SEARCH")
	{
		ref_prf.fill_SEARCH();

		if (dd && avg)
		{
			ref_prf.sum = ref_prf.raw;
			ref_prf.raw = nullptr;
		}
		if (dd && !avg)
		{
			ref_prf.dyn = ref_prf.raw;
			ref_prf.raw = nullptr;
		}
		if (!dd && !avg)
		{
			ref_prf.raw = ref_prf.raw;
		}
	}

	if (dd && !avg)
		ref_prf.dedisperse_incoherent(0.0, hdr->nchann);
	if (!dd && !avg)
		ref_prf.dedisperse_incoherent(hdr->dm, hdr->nchann);


	// Process each file
	for (size_t i = 1; i < config["files"].size(); ++i)
	{

		// Build full input path
		const auto filename_yaml = config["files"][i];
		std::string filename = filename_yaml.as<std::string>();
		fs::path in_path = fs::path(input_dir) / filename;


		Profile profile(in_path.string(), format,
				buf_size, output_dir, verbose);
		BaseHeader* hdr = profile.getHeader();

		// Apply configurations
		apply_header_corrections(hdr, config);
		setup_time_limits(profile, t0, t1);

		if (verbose > 0)
			hdr->print();


		// Get redshift correction if available
		if (parfile != "")
		{
			fs::path par_path = fs::path(input_dir) / parfile;
			profile.get_redshift(par_path.string(), site);
		}

		// Load or generate RFI mask
		load_mask(profile, config);

		bool dd = ! (hdr->dds_mthd == "" || hdr->dds_mthd == "none");
		bool avg = (hdr->nchann == 1);

		if (hdr->MODE == "SEARCH")
		{
			profile.fill_SEARCH();

			if (dd && avg)
			{
				profile.sum = profile.raw;
				profile.raw = nullptr;
			}
			if (dd && !avg)
			{
				profile.dyn = profile.raw;
				profile.raw = nullptr;
			}
			if (!dd && !avg)
			{
				profile.raw = profile.raw;
			}
		}

		if (dd && !avg && t2_pred_file != "")
			profile.dedisperse_incoherent(0.0, hdr->nchann);
		if (!dd && !avg  && t2_pred_file != "")
			profile.dedisperse_incoherent(hdr->dm, hdr->nchann);


		// Execute ccf-based accumulation
		if (hdr->MODE == "SEARCH")
		{
			ref_prf.fill_SEARCH();

			if (ref_prf.hdr->dds_mthd == "")
				ref_prf.dedisperse_incoherent(ref_prf.hdr->dm, ref_prf.hdr->nchann);
		}

		ref_prf.accumulate_prf(profile, input_dir + t2_pred_file);
	}
	
	ref_prf.finish_accumulation();

	std::string file_orig = ref_prf.reader->filename;
	ref_prf.reader->filename = "tpl_" + file_orig;

	if (ref_prf.raw and save_raw) ref_prf.save_raw("PSR");
	if (ref_prf.dyn and save_dyn) ref_prf.save_dyn("PSR");
	if (ref_prf.sum and save_sum) ref_prf.save_sum("PSR");

	ref_prf.reader->filename = file_orig;
	
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
		read_key("mode", &mode, config["general"]);

		std::string def_path = ".";
		read_key("input_dir", &input_dir, config["general"], &def_path);
		read_key("output_dir", &output_dir, config["general"], &def_path);
		read_key("site", &site, config["general"]);
		read_key("verbose", &verbose, config["general"], &verbose);
		read_key("buf_size", &buf_size, config["general"], &buf_size);
		read_key("t0", &t0, config["general"], &t0);
		read_key("t1", &t1, config["general"], &t1);
		read_key("parfile", &parfile, config["general"], &parfile);
		read_key("save_raw", &save_raw, config["general"], &save_raw);
		read_key("save_dyn", &save_dyn, config["general"], &save_dyn);
		read_key("save_sum", &save_sum, config["general"], &save_sum);

		// Resolve paths (ensure trailing separator using fs::path)
		input_dir = resolve_path((fs::path(input_dir) / "").string());
		output_dir = resolve_path((fs::path(output_dir) / "").string());

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
		std::cout << "            ANTI-PIPELINE 2\n";
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
		else if (mode == "template")
		{
			run_template_mode(config, input_dir, output_dir, buf_size_bytes,
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
