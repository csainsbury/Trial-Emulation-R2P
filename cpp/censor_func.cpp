#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

/**
 * Censoring function in C++.
 * 
 * Implements the same logic as the R Rcpp censor_func.
 * This function performs artificial censoring based on treatment switches.
 */
py::dict censor_func(
    py::array_t<int> first,
    py::array_t<int> eligible,
    py::array_t<int> treatment,
    py::array_t<int> switch_flag
) {
    auto first_buf = first.request();
    auto eligible_buf = eligible.request();
    auto treatment_buf = treatment.request();
    auto switch_buf = switch_flag.request();
    
    int n = first_buf.size;
    
    if (eligible_buf.size != n || treatment_buf.size != n || switch_buf.size != n) {
        throw std::runtime_error("All input arrays must have the same size");
    }
    
    int* first_ptr = static_cast<int*>(first_buf.ptr);
    int* eligible_ptr = static_cast<int*>(eligible_buf.ptr);
    int* treatment_ptr = static_cast<int*>(treatment_buf.ptr);
    int* switch_ptr = static_cast<int*>(switch_buf.ptr);
    
    // Output arrays
    std::vector<int> started0(n, 0);
    std::vector<int> started1(n, 0);
    std::vector<int> stop0(n, 0);
    std::vector<int> stop1(n, 0);
    std::vector<int> eligible0_sw(n, 0);
    std::vector<int> eligible1_sw(n, 0);
    std::vector<bool> delete_flag(n, false);
    
    // State variables
    int started0_ = 0;
    int started1_ = 0;
    int stop0_ = 0;
    int stop1_ = 0;
    int eligible0_sw_ = 0;
    int eligible1_sw_ = 0;
    
    for (int i = 0; i < n; i++) {
        if (first_ptr[i]) {
            // Reset state at first row per ID
            started0_ = 0;
            started1_ = 0;
            stop0_ = 0;
            stop1_ = 0;
            eligible0_sw_ = 0;
            eligible1_sw_ = 0;
        }
        
        if (stop0_ == 1 || stop1_ == 1) {
            // Reset if stopped
            started0_ = 0;
            started1_ = 0;
            stop0_ = 0;
            stop1_ = 0;
            eligible0_sw_ = 0;
            eligible1_sw_ = 0;
        }
        
        if (started0_ == 0 && started1_ == 0 && eligible_ptr[i] == 1) {
            // Start tracking based on treatment
            if (treatment_ptr[i] == 0) {
                started0_ = 1;
            } else if (treatment_ptr[i] == 1) {
                started1_ = 1;
            }
        }
        
        if (started0_ == 1 && stop0_ == 0) {
            eligible0_sw_ = 1;
            eligible1_sw_ = 0;
        } else if (started1_ == 1 && stop1_ == 0) {
            eligible0_sw_ = 0;
            eligible1_sw_ = 1;
        } else {
            eligible0_sw_ = 0;
            eligible1_sw_ = 0;
        }
        
        if (switch_ptr[i] == 1) {
            if (eligible_ptr[i] == 1) {
                // Switch while eligible - start new regime
                if (treatment_ptr[i] == 1) {
                    started1_ = 1;
                    stop1_ = 0;
                    started0_ = 0;
                    stop0_ = 0;
                    eligible1_sw_ = 1;
                } else if (treatment_ptr[i] == 0) {
                    started0_ = 1;
                    stop0_ = 0;
                    started1_ = 0;
                    stop1_ = 0;
                    eligible0_sw_ = 1;
                }
            } else {
                // Switch while not eligible - mark stop
                stop0_ = started0_;
                stop1_ = started1_;
            }
        }
        
        if (eligible0_sw_ == 0 && eligible1_sw_ == 0) {
            delete_flag[i] = true;
        } else {
            started0[i] = started0_;
            started1[i] = started1_;
            stop0[i] = stop0_;
            stop1[i] = stop1_;
            eligible1_sw[i] = eligible1_sw_;
            eligible0_sw[i] = eligible0_sw_;
            delete_flag[i] = false;
        }
    }
    
    // Return as dictionary
    py::dict result;
    result["started0"] = py::cast(started0);
    result["started1"] = py::cast(started1);
    result["stop0"] = py::cast(stop0);
    result["stop1"] = py::cast(stop1);
    result["eligible0_sw"] = py::cast(eligible0_sw);
    result["eligible1_sw"] = py::cast(eligible1_sw);
    result["delete"] = py::cast(delete_flag);
    
    return result;
}

PYBIND11_MODULE(_censor_func, m) {
    m.doc() = "Censoring function for trial emulation";
    m.def("censor_func", &censor_func, "Artificial censoring C++ function");
}

