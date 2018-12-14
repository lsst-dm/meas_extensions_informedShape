// -*- lsst-c++ -*-
/*
 * This file is part of package_name.
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

#ifndef LSST_MEAS_EXTENSIONS_INFORMEDSHAPE_SHAPEPRIOR_H
#define LSST_MEAS_EXTENSIONS_INFORMEDSHAPE_SHAPEPRIOR_H

#include <Eigen/Dense>

#include "lsst/afw/geom/ellipses.h"

namespace lsst {
namespace meas {
namespace extensions {
namespace informedShape {

class ShapePrior {
public:
    using PriorGrad = Eigen::Matrix<double, 4, 1>;
    using Quadrupole = lsst::afw::geom::ellipses::Quadrupole;
    using Triplet = Eigen::Matrix<double, 3, 1>;

    virtual void setParameters(double flux, Quadrupole const & second) = 0;

    virtual double computeProbability() const = 0;

    virtual PriorGrad computeDerivative() const = 0;

    virtual ~ShapePrior() = default;
};

}  // namespace informedShape
}  // namespace extensions
}  // namespace meas
}  // namespace lsst

#endif  // LSST_MEAS_EXTENSIONS_INFORMEDSHAPE_SHAPE_PRIOR_H
