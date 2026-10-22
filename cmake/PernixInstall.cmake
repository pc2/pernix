include_guard(GLOBAL)

include(CMakePackageConfigHelpers)

set(PERNIX_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/pernix")

install(TARGETS pernix pernix_header_only
        EXPORT pernixTargets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

if (TARGET pernix_cuda_backend)
    install(TARGETS pernix_cuda_backend
            EXPORT pernixTargets
            RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
endif ()

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/pernix"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(EXPORT pernixTargets
        FILE pernixTargets.cmake
        NAMESPACE pernix::
        DESTINATION "${PERNIX_INSTALL_CMAKEDIR}")

configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/pernixConfig.cmake.in"
        "${PROJECT_BINARY_DIR}/pernixConfig.cmake"
        INSTALL_DESTINATION "${PERNIX_INSTALL_CMAKEDIR}")

write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/pernixConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion)

install(FILES
        "${PROJECT_BINARY_DIR}/pernixConfig.cmake"
        "${PROJECT_BINARY_DIR}/pernixConfigVersion.cmake"
        DESTINATION "${PERNIX_INSTALL_CMAKEDIR}")

configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/pernix.pc.in"
        "${PROJECT_BINARY_DIR}/pernix.pc"
        @ONLY)

install(FILES "${PROJECT_BINARY_DIR}/pernix.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

if (PERNIX_INSTALL_DOCS)
    install(DIRECTORY "${PROJECT_BINARY_DIR}/docs/html/"
            DESTINATION "${CMAKE_INSTALL_DOCDIR}")
endif ()
