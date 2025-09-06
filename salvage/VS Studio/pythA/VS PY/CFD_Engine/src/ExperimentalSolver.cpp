#include "ExperimentalSolver.h"
#include <iostream>

void ExperimentalSolver::solve() {
    std::cout << "Running Experimental solver..." << std::endl;
    // Example: set each field value to its index
    for (size_t i = 0; i < field_.size(); ++i) {
        field_.set(i, static_cast<double>(i));
    }
}
