/*
 * This file is part of meas_modelfit.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

#include "ndarray/pybind11.h"

#include "lsst/meas/extensions/informedShape/MomentsModel.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace meas {
namespace extensions {
namespace informedShape {

namespace {
using PyMomentsModel = py::class_<MomentsModel, std::shared_ptr<MomentsModel>>;

PYBIND11_MODULE(momentsModel, mod) {
    PyMomentsModel cls(mod, "MomentsModel");

    cls.def(py::init<MomentsModel::Moments const &>(), "weights"_a);

    cls.def("setParameters", &MomentsModel::setParameters);
    cls.def("computeValues", &MomentsModel::computeValues);
    cls.def("computeJacobian", &MomentsModel::computeJacobian);

    // Wrap the test functions for the anonymous classes
    mod.def("testScalar", &testScalar, "tol"_a = DEFAULT_TEST_TOLERANCE);
    mod.def("testAlphaX", &testAlphaX, "tol"_a = DEFAULT_TEST_TOLERANCE);
    mod.def("testAlphaY", &testAlphaY, "tol"_a = DEFAULT_TEST_TOLERANCE);
    mod.def("testBetaX", &testBetaX, "tol"_a = DEFAULT_TEST_TOLERANCE);
    mod.def("testBetaXY", &testBetaXY, "tol"_a = DEFAULT_TEST_TOLERANCE);
    mod.def("testBetaY", &testBetaY, "tol"_a = DEFAULT_TEST_TOLERANCE);

}
}  // namespace
}  // namespace informedShape
}  // namespace extensions
}  // namespace meas
}  // namespace lsst
