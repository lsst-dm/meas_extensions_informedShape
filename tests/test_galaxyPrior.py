# This file is part of meas_extensions_informedShape.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import scipy.stats as sps
import unittest

from sklearn.mixture import GaussianMixture

import lsst.utils
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.geom as geom
from lsst.afw.geom.ellipses import Quadrupole, SeparableConformalShearTraceRadius
from lsst.utils.tests import TestCase
from lsst.meas.extensions.informedShape import GalaxyPrior
from lsst.meas.modelfit import UnitSystem, LocalUnitTransform


class TestGalaxyPrior(TestCase):
    def setUp(self):
        np.random.seed(10)
        self.weights = np.array([0.23710965, 0.21441384, 0.14193948, 0.16343472, 0.2431023])
        self.weights = np.array([0.24234686, 0.16455491, 0.14205442, 0.2138765, 0.23716731])

        self.means = np.array([[4.24239229e+09, -2.65266056e+00, -1.41590154e+00],
                               [2.51632238e+09, -2.04936576e+00, -1.42731860e+00],
                               [1.53342365e+09, -1.49422685e+00, -1.32761860e+00],
                               [5.11503205e+09, -2.96614696e+00, -1.21340507e+00],
                               [3.40342232e+09, -2.41794564e+00, -1.47685366e+00]])
        self.means = np.array([[3.40067770e+09, -2.41720135e+00, -1.47687305e+00],
                               [5.11184119e+09, -2.96429368e+00, -1.21433063e+00],
                               [1.53378483e+09, -1.49448434e+00, -1.32764335e+00],
                               [2.51557142e+09, -2.04896935e+00, -1.42721941e+00],
                               [4.23847982e+09, -2.65187648e+00, -1.41653340e+00]])

        self.cov = np.array([[[1.30046970e+17, -5.33722860e+07, 2.36771634e+07],
                              [-5.33722860e+07, 4.41748015e-01, 1.62942995e-01],
                              [2.36771634e+07, 1.62942995e-01, 2.15110913e-01]],

                             [[1.47660719e+17, -5.16128996e+07, -1.40207120e+07],
                              [-5.16128996e+07, 5.07257527e-01, 2.88389185e-01],
                              [-1.40207120e+07, 2.88389185e-01, 3.04894417e-01]],

                             [[2.09299148e+17, -1.17764896e+08, -3.44092961e+07],
                              [-1.17764896e+08, 5.07195816e-01, 2.39505919e-01],
                              [-3.44092961e+07, 2.39505919e-01, 3.14374831e-01]],

                             [[1.77579714e+17, -1.80040905e+08, 1.31173305e+07],
                              [-1.80040905e+08, 5.83555496e-01, 1.22851767e-01],
                              [1.31173305e+07, 1.22851767e-01, 1.85293131e-01]],

                             [[1.24056253e+17, -3.28853279e+07, 1.20091319e+07],
                              [-3.28853279e+07, 4.57908226e-01, 2.34975542e-01],
                              [1.20091319e+07, 2.34975542e-01, 2.49636059e-01]]])
        self.cov = np.array([[[1.23526768e+17, -3.27328751e+07, 1.19792318e+07],
                              [-3.27328751e+07, 4.57862162e-01, 2.35167347e-01],
                              [1.19792318e+07, 2.35167347e-01, 2.49770166e-01]],

                             [[1.78494732e+17, -1.80294306e+08, 1.35036166e+07],
                              [-1.80294306e+08, 5.83413132e-01, 1.22710951e-01],
                              [1.35036166e+07, 1.22710951e-01, 1.85537774e-01]],

                             [[2.09392529e+17, -1.17823346e+08, -3.44095601e+07],
                              [-1.17823346e+08, 5.07260511e-01, 2.39545886e-01],
                              [-3.44095601e+07, 2.39545886e-01, 3.14372771e-01]],

                             [[1.47139178e+17, -5.13470218e+07, -1.39508477e+07],
                              [-5.13470218e+07, 5.07201215e-01, 2.88384538e-01],
                              [-1.39508477e+07, 2.88384538e-01, 3.04920992e-01]],

                             [[1.29926024e+17, -5.31849025e+07, 2.37251703e+07],
                              [-5.31849025e+07, 4.41661296e-01, 1.63251834e-01],
                              [2.37251703e+07, 1.63251834e-01, 2.15263525e-01]]])
        self.pc = np.array([[[2.84524363e-09, 3.95374635e-10, -6.79790282e-10],
                             [0.00000000e+00, 1.49205809e+00, -1.51822771e+00],
                             [0.00000000e+00, 0.00000000e+00, 2.86131361e+00]],

                            [[2.36694027e-09, 1.59448863e-09, -1.12673836e-09],
                             [0.00000000e+00, 1.57857354e+00, -9.14011649e-01],
                             [0.00000000e+00, 0.00000000e+00, 2.69007685e+00]],

                            [[2.18534198e-09, 8.47362845e-10, -2.61608454e-10],
                             [0.00000000e+00, 1.50591080e+00, -1.11996271e+00],
                             [0.00000000e+00, 0.00000000e+00, 2.24294929e+00]],

                            [[2.60696882e-09, 4.98892535e-10, -2.87736463e-10],
                             [0.00000000e+00, 1.42961822e+00, -1.55245980e+00],
                             [0.00000000e+00, 0.00000000e+00, 2.67918347e+00]],

                            [[2.77429044e-09, 6.31719608e-10, -9.39755987e-10],
                             [0.00000000e+00, 1.54323526e+00, -1.10216718e+00],
                             [0.00000000e+00, 0.00000000e+00, 2.67564379e+00]]])

        self.model = GaussianMixture(5, covariance_type='full')
        self.model.weights_ = self.weights
        self.model.means_ = self.means
        self.model.covariance_ = self.cov
        self.model.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(self.cov)).transpose((0, 2, 1))
        self.model.precisions_cholesky_ = self.pc

        self.num = 10000
        fluxMin = afwImage.fluxFromABMag(27)
        fluxMax = afwImage.fluxFromABMag(0)
        ixxMin = 0
        ixxMax = 30
        iyyMin = 0
        iyyMax = 30
        ixyMin = -10
        ixyMax = 10
        self.samples = np.empty((10000, 4))
        counter = 0
        while counter < 10000:
            ixx = np.random.uniform(ixxMin, ixxMax)
            iyy = np.random.uniform(iyyMin, iyyMax)
            ixy = np.random.uniform(ixyMin, ixyMax)
            flux = np.random.uniform(fluxMin, fluxMax)
            quad = Quadrupole(ixx, iyy, ixy)
            try:
                quad.normalize()
                self.samples[counter] = (ixx, iyy, ixy, flux)
                counter += 1
            except Exception:
                continue

        self.defaultLocation = afwGeom.Point2D(100, 100)
        self.wcs = afwGeom.makeSkyWcs(crpix=self.defaultLocation,
                                      crval=geom.SpherePoint(0, 45, geom.degrees),
                                      cdMatrix=afwGeom.makeCdMatrix(scale=1.0*geom.arcseconds,
                                                                    orientation=0*geom.degrees,
                                                                    flipX=False))

        self.calib = afwImage.Calib(fluxMax)

        sourceSystem = UnitSystem(self.wcs, self.calib)
        destinationSystem = UnitSystem(self.wcs.pixelToSky(self.defaultLocation), 0)
        self.transformation = LocalUnitTransform(self.defaultLocation, sourceSystem, destinationSystem)
        self.boxCoxParams = [7.428651161758321, -1.9465686350090095, 0.2299077796501515]

    def transformSample(self, sample):
        transformedShape = Quadrupole(*sample[0:3]).transform(self.transformation.geometric.getLinear())
        transformedFlux = self.calib.getMagnitude(sample[3] * self.transformation.flux)
        seperableEllipse = SeparableConformalShearTraceRadius(transformedShape)
        shape = np.sqrt(seperableEllipse.getE1()**2 + seperableEllipse.getE2()**2)
        radius = seperableEllipse.getTraceRadius()

        # Convert to box cox
        bcFlux = sps.boxcox(transformedFlux, lmbda=self.boxCoxParams[0])
        bcShape = sps.boxcox(shape, lmbda=self.boxCoxParams[1])
        bcRadius = sps.boxcox(radius, lmbda=self.boxCoxParams[2])
        print("python params: ", transformedFlux, shape, radius)
        print("python bc: ", bcFlux, bcShape, bcRadius)

        return bcFlux, bcShape, bcRadius

    def testScoring(self):
        galaxyPriorInst = GalaxyPrior(Quadrupole(0, 0, 0), self.wcs, self.calib, self.defaultLocation)
        scoresPrior = np.empty(self.num)
        scoresModel = np.empty(self.num)
        for i in range(self.num):
            # Calculate log probabilities from the scikit-learn GMM
            print('python sample: ', self.samples[i])
            bcFlux, bcShape, bcRadius = self.transformSample(self.samples[i])
            scoresModel[i] = self.model.score_samples([[bcFlux, bcRadius, bcShape]])
            galaxyPriorInst.setParameters(self.samples[i, 3], Quadrupole(*self.samples[i, 0:3]))
            proj = np.log(galaxyPriorInst.computeProjection())
            galVal = galaxyPriorInst.computeProbability()
            print("python gal val", galVal)
            scoresModel[i] -= proj
            scoresPrior[i] = np.log(galVal)
        import ipdb; ipdb.set_trace()


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
