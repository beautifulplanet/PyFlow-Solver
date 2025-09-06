cmake_minimum_required(VERSION 3.15)
project(CFD_Engine)
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(include)
include_directories(tests)

# Main engine sources
add_library(CFD_Engine_lib
    src/CFD_Engine.cpp
    src/OpenFOAMSolver.cpp
    src/ExperimentalSolver.cpp
)

# Main executable (optional, for running the engine directly)
add_executable(CFD_Engine_main src/main.cpp)
target_link_libraries(CFD_Engine_main CFD_Engine_lib)


# Test executable with Catch2 (original)
add_executable(runTests
    tests/test_cfd_engine.cpp
    tests/catch_amalgamated.cpp
    src/CFD_Engine.cpp
    src/OpenFOAMSolver.cpp
    src/ExperimentalSolver.cpp
)
target_link_libraries(runTests CFD_Engine_lib)

# Test executable for workspace tests
add_executable(runWorkspaceTests
    tests/workspace/main.cpp
    tests/workspace/test_field_updates.cpp
    tests/catch_amalgamated.cpp
    src/CFD_Engine.cpp
    src/OpenFOAMSolver.cpp
    src/ExperimentalSolver.cpp
)
target_link_libraries(runWorkspaceTests CFD_Engine_lib)

enable_testing()
add_test(NAME CFD_Engine_Tests COMMAND runTests)
add_test(NAME CFD_Engine_Workspace_Tests COMMAND runWorkspaceTests)
