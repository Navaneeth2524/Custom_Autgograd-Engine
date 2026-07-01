#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "matrix_ops.hpp"

namespace py = pybind11;

PYBIND11_MODULE(custom_matrix_ops, m) {
    m.doc() = "C++ optimized matrix operations for Python Autograd Engine";

    py::class_<RawMatrix>(m, "RawMatrix")
        .def(py::init<>())
        .def(py::init<size_t, size_t>(), py::arg("rows"), py::arg("cols"))
        .def(py::init<size_t, size_t, const std::vector<float>&>(), py::arg("rows"), py::arg("cols"), py::arg("data"))
        .def_readwrite("data", &RawMatrix::data)
        .def_readwrite("rows", &RawMatrix::rows)
        .def_readwrite("cols", &RawMatrix::cols)
        .def("__repr__", [](const RawMatrix& self) {
            return "RawMatrix(rows=" + std::to_string(self.rows) + ", cols=" + std::to_string(self.cols) + ")";
        });

    m.def("matmul", &matmul, "Performs row-column cache-optimized matrix multiplication", py::arg("A"), py::arg("B"));
    m.def("transpose", &transpose, "Performs shape and data transposition", py::arg("A"));
    m.def("elementwise_add", &elementwise_add, "Performs fast elementwise addition of matching matrices", py::arg("A"), py::arg("B"));
}
