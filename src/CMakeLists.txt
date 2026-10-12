add_library(pernix)
add_library(pernix::pernix ALIAS pernix)
set_target_properties(pernix PROPERTIES
        VERSION "${PROJECT_VERSION}"
        SOVERSION "1"
        CXX_EXTENSIONS OFF)

add_library(pernix_header_only INTERFACE)
add_library(pernix::pernix_header_only ALIAS pernix_header_only)

target_sources(pernix PRIVATE
        pernix.cpp
        dispatch/select.cpp
        fallback/fallback_compression.cpp
        fallback/fallback_decompression.cpp
)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" PERNIX_SYSTEM_PROCESSOR)

if (PERNIX_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|x64)$")
    target_sources(pernix PRIVATE dispatch/cpu_features_x86.cpp)
    if (PERNIX_ENABLE_X86_64)
        add_subdirectory(x86_64)
    endif ()
elseif (PERNIX_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    target_sources(pernix PRIVATE dispatch/cpu_features_arm.cpp)
    if (PERNIX_ENABLE_ARM64)
        add_subdirectory(arm64)
    endif ()
else ()
    # The x86 detector returns an empty feature set on other architectures.
    target_sources(pernix PRIVATE dispatch/cpu_features_x86.cpp)
    message(STATUS "Unknown processor '${CMAKE_SYSTEM_PROCESSOR}': building the fallback backend only")
endif ()

target_compile_features(pernix PUBLIC cxx_std_20)
target_compile_features(pernix_header_only INTERFACE cxx_std_20)

target_include_directories(pernix
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(pernix_header_only
        INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(pernix PRIVATE PERNIX_BUILD_LIB=1)
target_compile_definitions(pernix PUBLIC PERNIX_USE_COMPILED_LIB=1)
target_compile_options(pernix PRIVATE ${PERNIX_PRIVATE_COMPILE_OPTIONS})

if (BUILD_SHARED_LIBS)
    target_compile_definitions(pernix PUBLIC PERNIX_SHARED=1)
endif ()
