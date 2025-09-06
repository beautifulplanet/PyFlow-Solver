#pragma once

#include <memory>
#include <string>
#include "SolverBase.h"
#include "Field.h"

class CFD_Engine {
public:
    CFD_Engine(const std::string& solverType, size_t fieldSize = 10);
    void run();
    const Field& field() const { return field_; }
    Field& field() { return field_; }
private:
    Field field_;
    std::unique_ptr<SolverBase> solver_;
};
