#pragma once
#include "SolverBase.h"

class OpenFOAMSolver : public SolverBase {
public:
    OpenFOAMSolver(Field& field) : SolverBase(field) {}
    void solve() override;
    std::string name() const override { return "OpenFOAM-like Solver"; }
};
