#pragma once
#include <string>

class Field;
class SolverBase {
public:
    SolverBase(Field& field) : field_(field) {}
    virtual ~SolverBase() = default;
    virtual void solve() = 0;
    virtual std::string name() const = 0;
protected:
    Field& field_;
};
