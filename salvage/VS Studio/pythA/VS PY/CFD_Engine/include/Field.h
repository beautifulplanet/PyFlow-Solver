#pragma once
#include <vector>

class Field {
public:
    Field(size_t size, double initial = 0.0) : data_(size, initial) {}
    void set(size_t idx, double value) { data_[idx] = value; }
    double get(size_t idx) const { return data_[idx]; }
    size_t size() const { return data_.size(); }
    std::vector<double>& data() { return data_; }
    const std::vector<double>& data() const { return data_; }
private:
    std::vector<double> data_;
};
