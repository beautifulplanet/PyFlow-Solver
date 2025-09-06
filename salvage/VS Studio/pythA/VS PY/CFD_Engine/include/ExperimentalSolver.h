#pragma once
#include "SolverBase.h"

class ExperimentalSolver : public SolverBase {
public:
    ExperimentalSolver(Field& field) : SolverBase(field) {}
    void solve() override;
    std::string name() const override { return "Experimental Solver"; }
};
