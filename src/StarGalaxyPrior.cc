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

#include <cmath>

#include "lsst/afw/geom/Angle.h"
#include "lsst/meas/extensions/informedShape/StarGalaxyPrior.h"

namespace lsst {
namespace meas {
namespace extensions {
namespace informedShape {

namespace {
    namespace afwGeom = lsst::afw::geom;
    namespace afwImage = lsst::afw::image;
    static double constexpr PI = lsst::afw::geom::PI;
}

    double DefaultStarClassificationPrior::pStar(double mag) const {
        double star = _starCoeff*std::pow(mag, _starExponent) + _starZero;
        double galaxy = _galCoeff*std::pow(mag, _galExponent) + _galZero;
        std::cout << "mag " << mag << " star " << star << " galaxy " << galaxy << std::endl;
        return star/(star+galaxy);
    }

    double DefaultStarClassificationPrior::pStarGrad(double mag) const {
        double star = _starCoeff*std::pow(mag, _starExponent) + _starZero;
        double galaxy = _galCoeff*std::pow(mag, _galExponent) + _galZero;
        double dStar = _starCoeff*_starExponent*std::pow(mag, _starExponent-1);
        double dGalaxy = _galCoeff*_galExponent*std::pow(mag, _galExponent-1);
        return ((star+galaxy)*dStar - star *(dStar + dGalaxy))/std::pow(star+galaxy, 2);
    }

    DefaultStarClassificationPrior::DefaultStarClassificationPrior(){
        _starCoeff = 8.31394255e-1;
        _starExponent = 2.61288362;
        _starZero = -8.48726338e2;
        _galCoeff = 2.12918502e-19;
        _galExponent = 1.70694103e1;
        _galZero = 5.96626682e10;
    }


    StarGalaxyPrior::StarGalaxyPrior(std::shared_ptr<ShapePrior> galaxyPrior,
                                     std::shared_ptr<ClassificationPrior> classify,
                                     const std::shared_ptr<afwImage::Calib const> calib,
                                     afwGeom::ellipses::Quadrupole psfMoments,
                                     double psfFuzzRad,
                                     double psfFuzzEllip):_galaxy(galaxyPrior),
                                                          _classification(classify),
                                                          _calib(calib){
       _dPsfShear = _psf.dAssign(psfMoments);
       _psfFuzz.Zero(3, 3);
       _psfFuzz(0, 0) = 1/psfFuzzEllip;
       _psfFuzz(1, 1) = 1/psfFuzzEllip;
       _psfFuzz(2, 2) = 1/psfFuzzRad;
       _psfFuzzDet = _psfFuzz.determinant();
    }

    void StarGalaxyPrior::setParameters(double flux, afwGeom::ellipses::Quadrupole const & second){
        _galaxy->setParameters(flux, second);
        _flux = flux;
        _magnitude = _calib->getMagnitude(flux);
        _imageMoments = second;
        _cachedStarLikelyhood = std::numeric_limits<double>::infinity(); 
        _cachedMomentsProb = std::numeric_limits<double>::infinity();
        _cachedGalaxyProb = std::numeric_limits<double>::infinity();
    }

    double StarGalaxyPrior::computeProbability() const {
        // Get the log probability related to how likely the moments correspond to the psf
        //convert psf and image moments to conformal shear

        afwGeom::ellipses::Separable<afwGeom::ellipses::ConformalShear,
                                     afwGeom::ellipses::TraceRadius> shearImage(_imageMoments);
        afwGeom::ellipses::Separable<afwGeom::ellipses::ConformalShear,
                                     afwGeom::ellipses::TraceRadius> shearPsf(_psf);
        afwGeom::ellipses::BaseCore::ParameterVector residuals(shearImage.getParameterVector() -
                                                               shearPsf.getParameterVector());
        double expMomentsProb = -0.5 * residuals.transpose()*_psfFuzz*residuals;
        double normMomentsProb = 1/(2*PI*_psfFuzzDet);
        std::cout << "expMomentsProb" << expMomentsProb << std::endl;
        std::cout << "normMoments" << normMomentsProb << std::endl;
        double _cachedMomentsProb = normMomentsProb*exp(expMomentsProb);

        // get the likelyhood that an object is a star
        double _cachedStarLikelyhood = _classification->pStar(_magnitude); 

        // get the probability of moments from the Galaxy Probabiliy function
        double _cachedGalaxyProb = _galaxy->computeProbability();
        std::cout << "star likelyhood " << _cachedStarLikelyhood << " moments prop " << _cachedMomentsProb << " Mixture gal prob " << _cachedGalaxyProb << std::endl;
        
        return _cachedStarLikelyhood*_cachedMomentsProb+ (1-_cachedStarLikelyhood)*_cachedGalaxyProb;
    }

    StarGalaxyPrior::PriorGrad StarGalaxyPrior::computeDerivative() const {
        // If any of the bedlow are inf, then the gradient is being called without first
        // calling computeProbabiliy, so some needed values are missing, issue function
        // call
        if (std::isinf(_cachedStarLikelyhood) || std::isinf(_cachedMomentsProb) ||
            std::isinf(_cachedGalaxyProb)) {
                computeProbability();
            }
        double starLikelyhoodGrad = _classification->pStarGrad(_magnitude);

        // Calculate the gradient for the psf term
        PriorGrad psfGrad;
        psfGrad(0, 0) = 0; // This is the flux term, where there is no gradient
        
        // calculation of just the shape terms
        Triplet psfShapeGrad;
        psfShapeGrad(0, 0) = -1 * _cachedMomentsProb * _psfFuzz(0, 0);
        psfShapeGrad(1, 0) = -1 * _cachedMomentsProb * _psfFuzz(1, 0);
        psfShapeGrad(2, 0) = -1 * _cachedMomentsProb * _psfFuzz(2, 0);

        // Assign the transformed gradient into the psfGradient vector
        psfGrad.block<3, 1>(1,0) = psfShapeGrad.transpose()*_dPsfShear;

        // Get the gradient of the Galaxy Prior
        PriorGrad galaxyGrad = _galaxy->computeDerivative();

        PriorGrad completeGradient = _cachedStarLikelyhood*(psfGrad - galaxyGrad) + galaxyGrad;
        
        // Add in the gradient of the classification likelyhood to the flux term
        completeGradient(0, 0) += starLikelyhoodGrad*(_cachedMomentsProb - _cachedGalaxyProb);

        return completeGradient;
    }


}  // namespace informedShape
}  // namespace extensions
}  // namespace meas
}  // namespace lsst

