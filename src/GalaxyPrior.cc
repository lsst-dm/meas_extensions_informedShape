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

#include <math.h>
#include "lsst/meas/extensions/informedShape/GalaxyPrior.h"

namespace lsst {
namespace meas {
namespace extensions {
namespace informedShape {

namespace {
    namespace afwGeom = lsst::afw::geom;
    namespace afwImage = lsst::afw::image;
    namespace measModelfit = lsst::meas::modelfit;
    auto convertShear(afwGeom::ellipses::Quadrupole const & second) {
        afwGeom::ellipses::Separable<afwGeom::ellipses::ConformalShear,
                                     afwGeom::ellipses::TraceRadius> shear;

        Eigen::Matrix<double, 3, 3> dConformalShear = shear.dAssign(second);

        double e = std::sqrt(std::pow(shear.getE1(), 2) + std::pow(shear.getE2(), 2));

        Eigen::Matrix<double, 3, 4> dIsotropy;
        dIsotropy.Zero(3, 4);
        dIsotropy(0, 0) = 1; // Carry through the flux term unchanged
        dIsotropy(1, 1) = shear.getE1()/e;
        dIsotropy(1, 2) = shear.getE2()/e;
        dIsotropy(2, 3) = 1; // Carry through the radial term unchanged

        return std::make_tuple(e, shear.getTraceRadius(), dIsotropy, dConformalShear);
    }

    auto convertBoxCox(double flux, double e, double radius, GalaxyPrior::Triplet lambdaVec) {
        double bcFlux = (std::pow(flux, lambdaVec[0]) - 1)/lambdaVec[0];
        double bcE = (std::pow(e, lambdaVec[1]) - 1)/lambdaVec[1];
        double bcRadius = (std::pow(radius, lambdaVec[2]) - 1)/lambdaVec[2];

        Eigen::Matrix<double, 3, 3> dBoxCox;
        dBoxCox.Zero(3, 3);
        dBoxCox(0, 0) = std::pow(flux, lambdaVec[0]-1);
        dBoxCox(1, 1) = std::pow(e, lambdaVec[1]-1);
        dBoxCox(2, 2) = std::pow(radius, lambdaVec[2]-1);

        return std::make_tuple(bcFlux, bcE, bcRadius, dBoxCox);
    }

} // end anonymous namespace for prior functions

void GalaxyPrior::setParameters(double flux, afwGeom::ellipses::Quadrupole const & second) {
    // Take out psf effects from the quadrupole, converting to a canonical Gaussian psf with width 4
    afwGeom::ellipses::Quadrupole correctedQuad;
    correctedQuad.setParameterVector(second.getParameterVector() - _psf.getParameterVector() +
                                     afwGeom::ellipses::Quadrupole::ParameterVector({4, 4, 0}));
    // Convert to the sky coordinates
    afwGeom::ellipses::Quadrupole::Transformer skyQuadTransform(second.transform(transformation.geometric.getLinear()));

    // Convert flux to calibrated magnitude
    double mag = calib->getMagnitude(flux * transformation.flux);

    afwGeom::ellipses::Quadrupole skyQuad(skyQuadTransform);

    Eigen::Matrix<double, 4, 4> dSky;
    dSky.Zero(4, 4);
    dSky(0, 0) = -2.5/(flux*log(10));
    dSky.block<3, 3>(1, 1) = skyQuadTransform.d();

    double radius, shape;
    Eigen::Matrix<double, 4, 4> dConformalShear;
    dConformalShear.Zero(4, 4);
    // Set the first element to 1, representing dflux / dflux
    dConformalShear(0, 0) = 1;
    Eigen::Matrix<double, 3, 4> dIsotropy;

    // convert to shier coordinates
    auto shapeShearBlock = dConformalShear.block<3, 3>(1, 1);
    std::tie(shape, radius, dIsotropy, shapeShearBlock) = convertShear(skyQuad);

    double bcFlux, bcRadius, bcShape; 
    Eigen::Matrix<double, 3, 3> dBoxCox;

    // convert to box cox
    std::tie(bcFlux, bcRadius, bcShape, dBoxCox) = convertBoxCox(mag, shape, radius, boxCoxParams);

    Eigen::Matrix<double, Eigen::Dynamic, 1> convertedParameters(3, 1);
    convertedParameters << bcFlux, bcRadius, bcShape;
    std::cout << "convertedParameters " << bcFlux << " " << bcRadius << " " << bcShape << std::endl;
    probability = GMM->evaluate(convertedParameters);
    std::cout << "prob pre norm " << probability << std::endl;

    // Normalize by the flux projection
    proj = fluxProjection->evaluate(Eigen::Matrix<double, 1, 1>(bcFlux));
    probability /= proj;
    std::cout << "prob post norm " << probability << std::endl;

    Eigen::Matrix<double, Eigen::Dynamic, 1> probGrad(3, 1);
    GMM->evaluateDerivatives(convertedParameters, probGrad);

    Eigen::Matrix<double, 1, 3> fixedProbGrad(probGrad.transpose());
    derivative = fixedProbGrad*dBoxCox*dIsotropy*dConformalShear*dSky;
}

double GalaxyPrior::computeProbability() const {
    return probability;
}

double GalaxyPrior::computeProjection() const {
    return proj;
}

GalaxyPrior::PriorGrad GalaxyPrior::computeDerivative() const {
    return derivative;
}

GalaxyPrior::GalaxyPrior(afwGeom::ellipses::Quadrupole psf, const std::shared_ptr<afwGeom::SkyWcs const> wcs,
                         const std::shared_ptr<afwImage::Calib const> calib,
                         afwGeom::Point2D const & location): _psf(psf), wcs(wcs), calib(calib),
                         boxCoxParams({7.428651161758321, -1.9465686350090095, 0.2299077796501515}) {
    Eigen::Matrix<double, 5, 1> weights;
    Eigen::Matrix<double, 5, 3> means;
    std::vector<Eigen::Matrix<double, 3, 3>> covariance(5);

    //weights << 0.23710965, 0.21441384, 0.14193948, 0.16343472, 0.2431023;
    weights << 0.24234686, 0.16455491, 0.14205442, 0.2138765 , 0.23716731;

    /*
    means <<  4.24239229e+09, -2.65266056e+00, -1.41590154e+00,
              2.51632238e+09, -2.04936576e+00, -1.42731860e+00,
              1.53342365e+09, -1.49422685e+00, -1.32761860e+00,
              5.11503205e+09, -2.96614696e+00, -1.21340507e+00,
              3.40342232e+09, -2.41794564e+00, -1.47685366e+00;
              */
    means << 3400677695.013583, -2.4172013549523332, -1.4768730474955711,
             5111841192.670658, -2.964293678677747, -1.2143306333627124,
             1533784825.9845226, -1.4944843419568008, -1.3276433492256159,
             2515571423.440081, -2.048969346956344, -1.4272194103498663,
             4238479817.076208, -2.6518764816364517, -1.4165334020086726;

    /*
    covariance[0] <<  1.30046970e+17, -5.33722860e+07,  2.36771634e+07,
                     -5.33722860e+07,  4.41748015e-01,  1.62942995e-01,
                      2.36771634e+07,  1.62942995e-01,  2.15110913e-01;

    covariance[1] <<  1.47660719e+17, -5.16128996e+07, -1.40207120e+07,
                     -5.16128996e+07,  5.07257527e-01,  2.88389185e-01,
                     -1.40207120e+07,  2.88389185e-01,  3.04894417e-01;

    covariance[2] <<  2.09299148e+17, -1.17764896e+08, -3.44092961e+07,
                     -1.17764896e+08,  5.07195816e-01,  2.39505919e-01,
                     -3.44092961e+07,  2.39505919e-01,  3.14374831e-01;

    covariance[3] <<  1.77579714e+17, -1.80040905e+08,  1.31173305e+07,
                     -1.80040905e+08,  5.83555496e-01,  1.22851767e-01,
                      1.31173305e+07,  1.22851767e-01,  1.85293131e-01;

    covariance[4] <<  1.24056253e+17, -3.28853279e+07,  1.20091319e+07,
                     -3.28853279e+07,  4.57908226e-01,  2.34975542e-01,
                      1.20091319e+07,  2.34975542e-01,  2.49636059e-01;
                      */

    covariance[0] << 1.2352676841121253e+17, -32732875.07676366, 11979231.77679869,
                     -32732875.076763663, 0.4578621623063791, 0.23516734685846774,
                     11979231.776798688, 0.2351673468584678, 0.2497701656817987;

    covariance[1] << 1.2352676841121253e+17, -32732875.07676366, 11979231.77679869,
                     -32732875.076763663, 0.4578621623063791, 0.23516734685846774,
                     11979231.776798688, 0.2351673468584678, 0.2497701656817987;

    covariance[2] << 1.2352676841121253e+17, -32732875.07676366, 11979231.77679869,
                     -32732875.076763663, 0.4578621623063791, 0.23516734685846774,
                     11979231.776798688, 0.2351673468584678, 0.2497701656817987;

    covariance[3] << 1.2352676841121253e+17, -32732875.07676366, 11979231.77679869,
                     -32732875.076763663, 0.4578621623063791, 0.23516734685846774,
                     11979231.776798688, 0.2351673468584678, 0.2497701656817987;

    covariance[4] << 1.2352676841121253e+17, -32732875.07676366, 11979231.77679869,
                     -32732875.076763663, 0.4578621623063791, 0.23516734685846774,
                     11979231.776798688, 0.2351673468584678, 0.2497701656817987;

    std::vector<measModelfit::Mixture::Component> ComponentList;
    ComponentList.reserve(weights.size());

    for (auto i = 0; i < weights.size(); ++i) {
        ComponentList.emplace_back(weights[i], means.row(i), covariance[i]);
    }

    GMM = std::make_unique<measModelfit::Mixture>(means.cols(), ComponentList);
    fluxProjection = GMM->project(Parameter::Flux);

    measModelfit::UnitSystem source(wcs, calib);
    measModelfit::UnitSystem destination(wcs->pixelToSky(location), magZero);

    transformation = measModelfit::LocalUnitTransform(location, source, destination);

    derivative.Zero(4);
}

}  // namespace informedShape
}  // namespace extensions
}  // namespace meas
}  // namespace lsst

