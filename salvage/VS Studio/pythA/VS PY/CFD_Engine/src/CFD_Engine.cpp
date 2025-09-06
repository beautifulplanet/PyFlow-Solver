#include "CFD_Engine.h"
#include "OpenFOAMSolver.h"
#include "ExperimentalSolver.h"
#include <iostream>

CFD_Engine::CFD_Engine(const std::string& solverType, size_t fieldSize)
    : field_(fieldSize) {
    if (solverType == "openfoam") {
        solver_ = std::make_unique<OpenFOAMSolver>(field_);
    } else if (solverType == "experimental") {
        solver_ = std::make_unique<ExperimentalSolver>(field_);
    } else {
        std::cerr << "Unknown solver type. Defaulting to OpenFOAM-like solver." << std::endl;
        solver_ = std::make_unique<OpenFOAMSolver>(field_);
    }
}

void CFD_Engine::run() {
    std::cout << "Selected solver: " << solver_->name() << std::endl;
    solver_->solve();
}
