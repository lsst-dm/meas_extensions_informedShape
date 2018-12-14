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

#include "lsst/meas/extensions/informedShape/GalaxyPrior.h"
#include "lsst/meas/extensions/informedShape/ShapePrior.h"

namespace py = pybind11;

namespace lsst {
namespace meas {
namespace extensions {
namespace informedShape {

namespace {
using PyGalaxyPrior = py::class_<GalaxyPrior, ShapePrior, std::shared_ptr<GalaxyPrior>>;

PYBIND11_MODULE(galaxyPrior, mod) {
    PyGalaxyPrior cls(mod, "GalaxyPrior");
    cls.def(py::init<lsst::afw::geom::ellipses::Quadrupole,
                     const std::shared_ptr<afw::geom::SkyWcs const>,
                     const std::shared_ptr<afw::image::Calib const>,
                     afw::geom::Point2D const &>());
    cls.def("setParameters", &GalaxyPrior::setParameters);
    cls.def("computeProbability", &GalaxyPrior::computeProbability);
    cls.def("computeProjection", &GalaxyPrior::computeProjection);
    cls.def("computeDerivative", &GalaxyPrior::computeDerivative);

}

}  // end anonymous
}}}}  // end lsst::meas::extensions::informedShape
