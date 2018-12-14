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

#ifndef LSST_MEAS_EXTENSIONS_INFORMEDSHAPE_CLASSIFICATIONPRIOR_H
#define LSST_MEAS_EXTENSIONS_INFORMEDSHAPE_CLASSIFICATIONPRIOR_H

namespace lsst {
namespace meas {
namespace extensions {
namespace informedShape {

class ClassificationPrior {
public:
    virtual double pStar(double mag) const = 0;
    virtual double pStarGrad(double mag) const = 0; 
    virtual ~ClassificationPrior() = default;
};

}  // namespace informedShape
}  // namespace extensions
}  // namespace meas
}  // namespace lsst

#endif  // LSST_MEAS_EXTENSIONS_INFORMEDSHAPE_CLASSIFICATIONPRIOR_H
