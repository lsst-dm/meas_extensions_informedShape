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

#ifndef LSST_MEAS_EXTENSIONS_INFORMEDSHAPE_STARGALAXYPRIOR_H
#define LSST_MEAS_EXTENSIONS_INFORMEDSHAPE_STARGALAXYPRIOR_H

#include <limits>

#include "Eigen/Dense"
#include "lsst/afw/geom.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/meas/extensions/informedShape/ShapePrior.h"
#include "lsst/meas/extensions/informedShape/ClassificationPrior.h"

namespace lsst {
namespace meas {
namespace extensions {
namespace informedShape {

class DefaultStarClassificationPrior : public ClassificationPrior {
public:
    double pStar(double mag) const override;

    double pStarGrad(double mag) const override;

    DefaultStarClassificationPrior();

private:
    double _starCoeff;
    double _starExponent;
    double _starZero;
    double _galCoeff;
    double _galExponent;
    double _galZero;
};


class StarGalaxyPrior : public ShapePrior {
public:
    void setParameters(double flux, lsst::afw::geom::ellipses::Quadrupole const & second) override;

    double computeProbability() const override;

    PriorGrad computeDerivative() const override;

    StarGalaxyPrior(std::shared_ptr<ShapePrior> galaxyPrior, std::shared_ptr<ClassificationPrior> classify,
                    const std::shared_ptr<lsst::afw::image::Calib const> calib,
                    lsst::afw::geom::ellipses::Quadrupole psfMoments, double psfFuzzRad, double psfFuzzEllip);

private:
        std::shared_ptr<ShapePrior> _galaxy;
        std::shared_ptr<ClassificationPrior> _classification;
        const std::shared_ptr<lsst::afw::image::Calib const> _calib;
        afw::geom::ellipses::Separable<afw::geom::ellipses::ConformalShear,
                                       afw::geom::ellipses::TraceRadius> _psf;
        Eigen::Matrix<double, 3, 3> _dPsfShear;
        Eigen::Matrix<double, 3, 3> _psfFuzz;
        double _psfFuzzDet = 0;
        afw::geom::ellipses::Separable<afw::geom::ellipses::ConformalShear,
                                       afw::geom::ellipses::TraceRadius> _imageMoments;
        double _flux = 0;
        double _magnitude = 0;
        // Variables to hold values cached from computing probability
        // Calling setParameters will reset these to ininify to make sure computeProbabiliy is
        // called before computeDerivative
        double _cachedStarLikelyhood = std::numeric_limits<double>::infinity();
        double _cachedMomentsProb = std::numeric_limits<double>::infinity();
        double _cachedGalaxyProb = std::numeric_limits<double>::infinity();
};

}  // namespace informedShape
}  // namespace extensions
}  // namespace meas
}  // namespace lsst

#endif  // LSST_MEAS_EXTENSIONS_INFORMEDSHAPE_STARGALAXYPRIOR_H
