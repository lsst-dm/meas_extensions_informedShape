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

#ifndef LSST_MEAS_EXTENSIONS_INFORMEDSHAPE_GALAXYPRIOR_H
#define LSST_MEAS_EXTENSIONS_INFORMEDSHAPE_GALAXYPRIOR_H

#include "Eigen/Dense"

#include "lsst/afw/geom.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/meas/modelfit/Mixture.h"
#include "lsst/meas/modelfit/UnitSystem.h"
#include "lsst/meas/extensions/informedShape/ShapePrior.h"

namespace lsst {
namespace meas {
namespace extensions {
namespace informedShape {

class GalaxyPrior : public ShapePrior {
public:
    enum Parameter {Flux, Radius, Shape};

    void setParameters(double flux, Quadrupole const & second) override;

    double computeProbability() const override;

    double computeProjection() const;

    PriorGrad computeDerivative() const override;

    GalaxyPrior(lsst::afw::geom::ellipses::Quadrupole psf, const std::shared_ptr<afw::geom::SkyWcs const> wcs,
                const std::shared_ptr<afw::image::Calib const> calib,
                afw::geom::Point2D const & location);

private:
    // mixture model over a box cox transformation with flux, radius, and shape, with
    // dimensions in that order
    std::unique_ptr<lsst::meas::modelfit::Mixture> GMM;
    std::shared_ptr<lsst::meas::modelfit::Mixture> fluxProjection;
    const lsst::afw::geom::ellipses::Quadrupole _psf;
    const std::shared_ptr<afw::geom::SkyWcs const> wcs;
    const std::shared_ptr<afw::image::Calib const> calib;
    Triplet boxCoxParams;
    lsst::meas::modelfit::LocalUnitTransform transformation;
    double magZero = 0;
    // variable to store the probability of the prior, is constructed with 0 to
    // ensure someone calls at before computeProbability
    double probability = 0;
    double proj = 0;
    PriorGrad derivative;
};

}  // namespace informedShape
}  // namespace extensions
}  // namespace meas
}  // namespace lsst

#endif  // LSST_MEAS_EXTENSIONS_INFORMEDSHAPE_GALAXYPRIOR_H
