#ifndef CONFIG_UTILS_H
#define CONFIG_UTILS_H

#include "Profile.h"
#include "aux_math.h"

#include <string>
#include <yaml-cpp/yaml.h>


// This is a false positive in GCC’s warning machinery, 
// and many users have reported it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <regex>
#pragma GCC diagnostic pop

//=============================================================================
// PATH RESOLUTION UTILITIES
//=============================================================================

/**
 * Expands environment variables and '~' in paths, returns canonical path
 * (keeps the original algorithm, but uses fs::path for the final step)
 */
std::string resolve_path(const std::string& input);
//=============================================================================
// FILE FORMAT DETECTION
//=============================================================================

/**
 * Determines file format based on extension (or special patterns)
 */
std::string get_format(const std::string& filename);

//=============================================================================
// YAML CONFIGURATION READING
//=============================================================================

/**
 * Reads a key from YAML config with optional default value
 */
template<typename T>
void read_key(const std::string& key, T* value, const YAML::Node& config, T* def = nullptr);


/**
 * Loads or generates a channel mask for RFI mitigation
 */
void load_mask(Profile& profile, const YAML::Node& config);
/**
 * Applies header corrections from advanced config section
 */
void apply_header_corrections(BaseHeader* hdr, const YAML::Node& config);

/**
 * Sets up time limits for data processing
 */
void setup_time_limits(Profile& profile, double t0, double t1);

#endif
