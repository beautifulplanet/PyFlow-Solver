#include "OpenFOAMSolver.h"
#include <iostream>

void OpenFOAMSolver::solve() {
    std::cout << "Running OpenFOAM-like solver..." << std::endl;
    // Example: increment each field value by 1.0
    for (size_t i = 0; i < field_.size(); ++i) {
        field_.set(i, field_.get(i) + 1.0);
    }
}
