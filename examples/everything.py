import numpy as np
import scipy as sp
import scipy.optimize as spo

import lsst.meas.extensions.informedShape as meis
from lsst.afw.geom.ellipses import Quadrupole
import lsst.geom as geom
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage

# Set the seed so everything is reproducible
np.random.seed(10)


def buildTestImage(moments, imsize, noise):
    indY, indX = np.indices((imsize, imsize))
    image = meis.makeGaussian(indX, indY, *moments)
    noiseImage = np.random.normal(0, noise, image.size).reshape(image.shape)
    return image+noiseImage


def fitFunctionFactory(measuredMoments, weightMoments, pixelUncertanty, imSize,
                       psf, wcs, calib, location, psfFuzzRad, psfFuzzShape, history):
    # Moments model stuff
    momentsModel = meis.MomentsModel(weightMoments)
    uncertanty = np.linalg.pinv(meis.buildUncertanty((imSize, imSize), weightMoments, pixelUncertanty))
    uncertanty = np.eye(6)

    # Prior
    galPrior = meis.GalaxyPrior(psf, wcs, calib, location)
    classifier = meis.DefaultStarClassificationPrior()
    prior = meis.StarGalaxyPrior(galPrior, classifier, calib, psf, psfFuzzRad, psfFuzzShape)

    def normMoments(params):
        C = np.array([[weightMoments[3], weightMoments[4]], [weightMoments[4], weightMoments[5]]])
        M = np.array([[params[3], params[4]], [params[4], params[5]]])
        QP = np.linalg.inv(np.linalg.inv(M) - np.linalg.inv(C))
        newParams = np.array((params[0], params[1], params[2], QP[0, 0], QP[0, 1], QP[1, 1]))
        return newParams

    # normedMeasure = normMoments(measuredMoments)

    def fitFunction(params, *args):
        print("python params", params)
        history.append(params)
        flux = params[0]
        moments = params[3:]
        momentsModel.setParameters(params)
        # Only need the quad parameters for the prior, also stupid ordering that should be changed
        # on my part
        prior.setParameters(flux, Quadrupole(moments[0], moments[2], moments[1]))

        # Results of the Moments modeling
        momentsResults = momentsModel.computeValues()
        momentsResultsVec = measuredMoments - momentsResults
        # momentsResultsVec = normedMeasure - normMoments(momentsResults)
        chisq = np.dot(np.dot(np.transpose(momentsResultsVec), uncertanty), momentsResultsVec)

        # Prior
        priorValue = prior.computeProbability()

        val = chisq - 2*np.log(priorValue)
        print("Chi value: ", chisq)
        print("Prior val: ", priorValue)
        print(-2*np.log(priorValue))
        print("Together ", val)
        # if np.isnan(val):
        #    return np.inf
        return chisq

    def gradient(params, *args):
        momentsJacobian = momentsModel.computeJacobian()
        residuals = measuredMoments - momentsModel.computeValues()
        momentsGrad = -2*np.dot(np.dot(np.transpose(momentsJacobian), uncertanty), residuals)

        priorGrad = -2*(1/prior.computeProbability())*prior.computeDerivative()
        priorGradWithPosition = np.array((priorGrad[0], 0, 0, *priorGrad[1:]))

        return momentsGrad + priorGradWithPosition

    return fitFunction, gradient


W = np.array((1, 50, 50, 4, 0.15, 4))
# Create an image with an extended like circular object
Q = np.array((20, 50, 50, 3.8, 0.15, 4.3))
imSize = 101
uncert = 0.00001
tImage = buildTestImage(Q, imSize, uncert)
indy, indx = np.indices((imSize, imSize))

measMoments, weightImage = meis.measureMoments(tImage, W)

# Make a psf that is 10 circular for the psf this will test if this finds a galaxy
psf = Quadrupole(10, 10, 0)
# Make a calib with fake units for now
calib = afwImage.Calib(59631.0)
wcs = afwGeom.makeSkyWcs(crpix=afwGeom.Point2D(50, 50),
                         crval=geom.SpherePoint(0, 45, geom.degrees),
                         cdMatrix=afwGeom.makeCdMatrix(scale=1.0*geom.arcseconds,
                                                       orientation=0*geom.degrees,
                                                       flipX=False))
location = afwGeom.Point2D(50, 50)
psfFuzzRad = float(0.5) # Pixels
psfFuzzShape = float(0.1) # Conformal shear (approximately 10%
history = []

fitFunc, gradFunc = fitFunctionFactory(measMoments, W, 5, imSize, psf, wcs, calib, location,
                                       psfFuzzRad, psfFuzzShape, history)

guess = np.array((18, 49.5, 51.1, 3.5, 0.1, 3.9))
#fit, other = spo.minimize(fun=fitFunc, x0=guess, jac=gradFunc)


def con(params, *extra):
    return params[1]*params[3] - params[2]


#fit, other = spo.minimize(fun=fitFunc, x0=guess, method="COBYLA", constraints={'type': 'ineq', 'fun': con})
solution = spo.minimize(fun=fitFunc, x0=guess, method="Powell")

# MCMC stuff
from demczs import demczs


def fitFunctionFactoryMcmc(measuredMoments, weightMoments, pixelUncertanty, imSize,
                           psf, wcs, calib, location, psfFuzzRad, psfFuzzShape):
    # Moments model stuff
    momentsModel = meis.MomentsModel(weightMoments)
    uncertanty = np.linalg.pinv(meis.buildUncertanty((imSize, imSize), weightMoments, pixelUncertanty))
    #uncertanty = np.eye(6)

    # Prior
    galPrior = meis.GalaxyPrior(psf, wcs, calib, location)
    classifier = meis.DefaultStarClassificationPrior()
    prior = meis.StarGalaxyPrior(galPrior, classifier, calib, psf, psfFuzzRad, psfFuzzShape)

    def normMoments(params):
        C = np.array([[weightMoments[3], weightMoments[4]], [weightMoments[4], weightMoments[5]]])
        M = np.array([[params[3], params[4]], [params[4], params[5]]])
        QP = np.linalg.inv(np.linalg.inv(M) - np.linalg.inv(C))
        newParams = np.array((params[0], params[1], params[2], QP[0, 0], QP[0, 1], QP[1, 1]))
        return newParams

    normedMeasure = normMoments(measuredMoments)

    def fitFunc(parameters, ind, extra):
        momentsModel.setParameters(parameters)
        # Results of the Moments modeling
        momentsResults = momentsModel.computeValues()
        return momentsResults

    def chiFunc(model, data, errors, extra):
        momentsResultsVec = measuredMoments - model
        chisq = np.dot(np.dot(momentsResultsVec, uncertanty), momentsResultsVec)
        return chisq

    def conFunc(parameters, constraints):
        # make sure the parameters are in the bounds
        for i, (p, con) in enumerate(zip(parameters, constraints)):
            if p <= con[0] or p >= con[1]:
                #print('out of bounds {} {}, {}'.format(con[0], con[1], p))
                return 1e99
        # Check the determinant of shapes, if negative this is not a vaild shape and should be rejected
        det = parameters[3]*parameters[5] - parameters[4]
        if det <= 0:
            print('det problem {} {} {}'.format(parameters[3], parameters[5], parameters[4]))
            return 1e99
        flux = parameters[0]
        moments = parameters[3:]
        # Only need the quad parameters for the prior, also stupid ordering that should be changed
        # on my part
        #prior.setParameters(flux, Quadrupole(moments[0], moments[2], moments[1]))
        #priorValue = prior.computeProbability()
        #logPrior = -2*np.log(priorValue)
        #if not np.isfinite(logPrior):
        #    return 1e99
        #return logPrior
        return 0

    
    return fitFunc, chiFunc, conFunc


iterations = 5e5
data = np.array((1))  # dummy for the fitFunc wrapper
ind = np.array((1))  # dummy for the fitFunc wrapper
errors = np.array((1))  # dummy for the fitFunc wrapper
par = np.array((17, 49.5, 51.1, 3.9, 0.15, 4.2))
steps = np.array((10, 0.5, 0.55, 0.2, 0.005, 0.2))
const = ((10, 30), (45.9, 55.1), (45.9, 55.1), (0, 10), (-1, 1), (0, 10))
ex = ()
num_chain = 4
thinning = 10

#fitFunc, chiFunc, conFunc = fitFunctionFactoryMcmc(measMoments, W, 5, imSize, psf, wcs, calib, location,
#                                                   psfFuzzRad, psfFuzzShape)

#output = demczs(iterations, data, ind, errors, fitFunc, chiFunc, conFunc, par, steps, const, ex, num_chain,
#                thinning, hist_mult=100)
'''
gridOut = np.zeros((100, 4, 4, 10, 20, 10))
# dumb loop
for i, iv in enumerate(np.linspace(2950, 3050, 100)):
    print(i)
    for j, jv in enumerate(np.linspace(48, 52, 4)):
        for k, kv in enumerate(np.linspace(48, 52, 4)):
            for l, lv in enumerate(np.linspace(25, 35, 10)):
                for m, mv in enumerate(np.linspace(-1, 1, 20)):
                    for n, nv in enumerate(np.linspace(25, 35, 10)):
                        fit = fitFunc(np.array((iv, jv, kv, lv, mv, nv)), (), ())
                        gridOut[i, j, k, l, m, n] = chiFunc(fit, (), (), ())
'''
