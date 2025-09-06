#include "CFD_Engine.h"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string solverType = "openfoam";
    if (argc > 1) {
        solverType = argv[1];
    }
    CFD_Engine engine(solverType);
    engine.run();
    return 0;
}
