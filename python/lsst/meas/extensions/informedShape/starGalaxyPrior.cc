/*
 * This file is part of meas_extensions_informedShape.
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

#include "lsst/meas/extensions/informedShape/StarGalaxyPrior.h"
#include "lsst/meas/extensions/informedShape/ShapePrior.h"

namespace py = pybind11;

namespace lsst {
namespace meas {
namespace extensions {
namespace informedShape {

namespace {
using PyStarGalaxyPrior = py::class_<StarGalaxyPrior, ShapePrior, std::shared_ptr<StarGalaxyPrior>>;

PYBIND11_MODULE(starGalaxyPrior, mod) {
    PyStarGalaxyPrior cls(mod, "StarGalaxyPrior");
    cls.def(py::init<std::shared_ptr<ShapePrior>, std::shared_ptr<ClassificationPrior>,
                     const std::shared_ptr<lsst::afw::image::Calib const>,
                     lsst::afw::geom::ellipses::Quadrupole,double, double>());
    
    cls.def("setParameters", &StarGalaxyPrior::setParameters);
    cls.def("computeProbability", &StarGalaxyPrior::computeProbability);
    cls.def("computeDerivative", &StarGalaxyPrior::computeDerivative);

}

}  // end anonymous
}}}}  // end lsst::meas::extensions::informedShape
