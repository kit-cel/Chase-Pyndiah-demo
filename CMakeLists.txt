cmake_minimum_required(VERSION 3.20)
project(CP LANGUAGES C CXX)

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Good default for pybind/shared libs
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optional: your release flags (kept from your file)
set(CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -pipe -flto -ffast-math"
)

# --- Python / pybind11 ---
#find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
#set(Python3_EXECUTABLE "/opt/anaconda3/envs/optuna-envs/bin/python")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
add_subdirectory(pybind11)

# --- Sources ---
set(HEADER_FILES
        lib/BCH.h
        lib/helpers.h
        lib/GF_N.h
        lib/GPC.h
        lib/productCode.h
        lib/staircaseCode.h
        simulation.h
)

set(SOURCE_FILES
        lib/BCH.cpp
        lib/GF_N.cpp
        lib/GPC.cpp
        simulation.cpp
        SISOsim.cpp
        NNBCHtry.cpp
        lib/SimpleNN.h
)



# --- Core library ---
add_library(CP ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(CP PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# --- OpenMP (target-based, robust on macOS) ---
find_package(OpenMP)

if (OpenMP_CXX_FOUND)
    # Best case: CMake provides the imported target
    target_link_libraries(CP PUBLIC OpenMP::OpenMP_CXX)
elseif(APPLE)
    # Fallback: Homebrew libomp (fixes 'omp.h' not found)
    execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    target_compile_options(CP PUBLIC -Xpreprocessor -fopenmp)
    target_include_directories(CP PUBLIC ${LIBOMP_PREFIX}/include)
    target_link_directories(CP PUBLIC ${LIBOMP_PREFIX}/lib)
    target_link_libraries(CP PUBLIC omp)
endif()

# --- Executables ---

add_executable(SISO SISOsim.cpp)
target_link_libraries(SISO PRIVATE CP)

add_executable(NNbchTry NNBCHtry.cpp)
target_link_libraries(NNbchTry PRIVATE CP)


# --- Python module ---
pybind11_add_module(GPC_simulator_python CPwrapper.cpp)
target_link_libraries(GPC_simulator_python PRIVATE pybind11::module CP)


#pybind11_add_module(bch_simulator bch_python.cpp)
#target_link_libraries(bch_simulator PRIVATE pybind11::module CP)

pybind11_add_module(bch_simulator bch_python.cpp)
target_link_libraries(bch_simulator PRIVATE pybind11::module CP)
target_include_directories(bch_simulator PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
)