# Name of the final executable
TARGET = main

# Compiler to use (g++ or clang++)
CXX = g++

# Compiler flags
# -Wall: Enable most warning messages
# -Wextra: Enable extra warning messages
# -std=c++17: Use C++17 standard 
# -O3: Optimize for speed (use -O0 for debugging)
# -g: Include debugging symbols (remove for release build)
# -fopenmp: Enable to use OpenMP routines
CXXFLAGS = -std=c++17 -fopenmp -O3 -march=native -ffast-math
DEBUG = -g -Wall -Wextra #-fsanitize=address -fno-omit-frame-pointer #-fopt-info 

CXXFLAGS += $(DEBUG)


# Include directories (paths where the compiler looks for header files)
# The -I flag tells the compiler where to find #include files
INCLUDES = -I./include -I$(TEMPO2_PREFIX)/include -I/home/gpps/pulsar_software/include -I./libs/

# Library flags (paths where the linker looks for libraries and the libraries themselves)
# -L flag for library paths, -l flag for library names
# Example: -L/path/to/lib -lfftw3
LIBS = -lstdc++fs -lfftw3 -lyaml-cpp -ltempo2pred -ltempo2 -lcfitsio
LIBS += -L$(TEMPO2_PREFIX)/lib -L./libs/

RPATHS = -Wl,-rpath,$(TEMPO2_PREFIX)/lib -Wl,-rpath,./libs/


# --- Source and Build Directories ---
SRCDIR = src
INCDIR = include
BUILDDIR = build

# Find all source files (.cpp) in the src directory
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
SOURCES += $(wildcard $(SRCDIR)/*.C)

# --- Object Files ---
# Generate the list of object file names from source file names
# The $(SOURCES:.cpp=.o) substitution replaces the .cpp extension with .o
# First, get just the base names of the source files (e.g., main.cpp -> main)
# Then, prefix each base name with the build directory path
OBJ_NAMES = $(notdir $(SOURCES:.cpp=.o))
OBJECTS = $(addprefix $(BUILDDIR)/obj/,$(OBJ_NAMES))


# --- Default Target ---
# The first target in the Makefile is the default one executed when 'make' is run
all: $(TARGET)

# --- Linking Rule ---
# This rule tells make how to create the final executable $(TARGET)
# It depends on all the object files $(OBJECTS)
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(RPATHS) $(OBJECTS) $(LIBS)  -o $@ 


# --- Compilation Rule (Pattern Rule) ---
# This pattern rule tells make how to compile any .cpp file into a .o file
# $< is the first prerequisite (the .cpp file)
# $@ is the target (the .o file)
# The $(CXXFLAGS) and $(INCLUDES) are used for compilation
$(BUILDDIR)/obj/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)/obj
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILDDIR)/obj/%.o: $(SRCDIR)/%.C
	@mkdir -p $(BUILDDIR)/obj
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# --- Clean Target ---
# This target removes the executable and all object files
clean:
	rm -f $(TARGET)
	rm -rf $(BUILDDIR)

run: $(TARGET)
	tput reset
	./$(TARGET)

check: $(TARGET)
	tput reset
	valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         ./$(TARGET)
# --- Minimal self-contained PSRFITS writer library ---
LIB_TARGET = libpsrfits_writer.so
LIB_SOURCES = \
	$(SRCDIR)/PSRFITS_Writer.cpp \
	$(SRCDIR)/BaseHeader.cpp \
	$(SRCDIR)/aux_math.cpp
LIB_OBJECTS = $(LIB_SOURCES:$(SRCDIR)/%.cpp=$(BUILDDIR)/lib/%.o)

.PHONY: libpsrfits clean-lib

libpsrfits: $(LIB_TARGET)

$(LIB_TARGET): $(LIB_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) -fPIC $(CXXFLAGS) $(INCLUDES) -shared -Wl,--export-dynamic $(LIB_OBJECTS) -lcfitsio -o $@

$(BUILDDIR)/lib/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) -fPIC $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean-lib:
	rm -rf $(BUILDDIR)/lib $(LIB_TARGET)
