"""Shared companion-law parameters, independent of sampler implementations."""

import fractions
import math

import numpy as np

# Winters et al. (2019), AJ 157, 216, DOI 10.3847/1538-3881/ab05dc,
# §5.2: corrected companion count 0.324 per primary (not the 0.268 system
# multiplicity fraction). §5.4/Fig. 16: Gaussian log10 projected separation,
# peak 20 AU, sigma 1.16 dex. Figs. 19-20, 23-24 delimit the 0.075--0.60
# Msun branch; 0.63 Msun in §5.3 is a different spectral-type boundary.
WINTERS_2019_M_DWARF_MINIMUM_MASS_SOLAR = 0.075
WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR = 0.60
WINTERS_2019_M_DWARF_COMPANION_RATE = 0.324
WINTERS_2019_LOG10_SEPARATION_MEAN_AU = math.log10(20.0)
WINTERS_2019_LOG10_SEPARATION_STANDARD_DEVIATION_DEX = 1.16

# Reggiani & Meyer (2013), A&A 553 A124, DOI 10.1051/0004-6361/201321631,
# §3: dN/dq ∝ q^0.25 for the combined field M-dwarf and solar sample.
# Susemiehl & Meyer (2022), A&A 657 A48, Eq. 4 extrapolate that law below
# q=0.6 and beyond the fitted separations for M-dwarf binaries.
REGGIANI_M_DWARF_Q_EXPONENT = 0.25

# Moe & Di Stefano (2017), ApJS 230, 15, DOI 10.3847/1538-4365/aa6fb6,
# §9.3 Eqs. 20--23: calibration 0.8--40 Msun, q>0.3. TREX's 0.6--0.8
# branch is an explicit extrapolation; above 40 Msun is refused.
MOE2017_PRIMARY_MASS_MAXIMUM_SOLAR = 40.0

# Eq. 23 integrates (u-3.4)/2.1 from 3.4 to m<=5.5:
# (5/21)m²-(34/21)m+289/105; its tail integral is
# (10/3)(1-exp(-0.3(m-5.5))). Fractions avoid rounded polynomial coefficients.
MOE2017_LOGP_QUAD_C2 = float(fractions.Fraction(5, 21))
MOE2017_LOGP_QUAD_C1 = float(fractions.Fraction(34, 21))
MOE2017_LOGP_QUAD_C0 = float(fractions.Fraction(289, 105))
MOE2017_EXP_SCALE = float(fractions.Fraction(10, 3))

# MDS17 §2 Eq. 2 breakpoints, with §9.1's SOLAR-TYPE (0.8--1.2 Msun)
# small/large-q slopes, Eqs. 13/9, and Eq. 6 at M1=1 Msun.
# These are not universal constants: Eqs. 10--11, 14--15 and Fig. 35 give
# different slopes for other spectral types, not implemented by this model.
MOE2017_Q_SLOPE_LOW = 0.3
MOE2017_Q_SLOPE_HIGH = -0.5
MOE2017_TWIN_FRACTION = 0.30
MOE2017_Q_BREAK_LOW = 0.1
MOE2017_Q_BREAK_MID = 0.3
MOE2017_Q_TWIN_THRESHOLD = 0.95
# The stellar-grid secondary needs M2>=0.1 Msun; M1=0.1 has zero-width support.
MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR = np.nextafter(MOE2017_Q_BREAK_LOW, np.inf)

MOE2017_COMPANION_FRACTION_SLOPE = 0.018
MOE2017_COMPANION_LOGPERIOD_BIN_DEX = 0.7
MOE2017_LOGPERIOD_MIN_DAYS = 1.0
MOE2017_LOGPERIOD_FIRST_BREAK_DAYS = 2.0
MOE2017_LOGPERIOD_SECOND_BREAK_DAYS = 3.4
# Integrating alpha*(logP-2.7) over 2.0--m gives
# alpha/2 * (m²-5.4m+6.8); the derivative agrees at both joins.
MOE2017_LOGPERIOD_SECOND_INTEGRAL_LINEAR_COEFFICIENT = 5.4
MOE2017_LOGPERIOD_SECOND_INTEGRAL_CONSTANT = 6.8
MOE2017_LOGPERIOD_THIRD_BREAK_DAYS = 5.5
MOE2017_LOGPERIOD_MAX_DAYS = 8.0
MOE2017_COMPANION_FRACTION_EXPONENTIAL_SLOPE = -0.3
# MDS17 §9.3 Eqs. 20--22: tuples (constant, linear, quadratic) in log10(M1/Msun).
# f1: logP<1; f2: logP=2.7; f3: logP=5.5.
MOE2017_COMPANION_FRACTION_F1_COEFFS = (0.020, 0.04, 0.07)
MOE2017_COMPANION_FRACTION_F2_COEFFS = (0.039, 0.07, 0.01)
MOE2017_COMPANION_FRACTION_F3_COEFFS = (0.078, -0.05, 0.04)

# Eq. 23's first segment starts at logP=0.2, not at its first interior break.
MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS = 0.2
# Eq. 9: solar-type large-q slope -0.5 below logP=5, then -0.3 per dex to 8.
MOE2017_Q_SLOPE_HIGH_PERIOD_BREAK_LOGDAYS = 5.0
MOE2017_Q_SLOPE_HIGH_PERIOD_GRADIENT_PER_DEX = -0.3
# Eqs. 5--7: short-period twin fraction 0.30-0.15log10(M1/Msun),
# taper from logP=1 to cutoff 8-M1 (M1<=6.5), else 1.5.
MOE2017_TWIN_FRACTION_MASS_COEFFICIENT = -0.15
MOE2017_TWIN_PLATEAU_LOGPERIOD_DAYS = 1.0
MOE2017_TWIN_CUTOFF_LOGPERIOD_INTERCEPT = 8.0
MOE2017_TWIN_CUTOFF_LOGPERIOD_MASS_SLOPE_PER_SOLAR_MASS = -1.0
MOE2017_TWIN_CUTOFF_PRIMARY_MASS_BREAK_SOLAR = 6.5
MOE2017_TWIN_CUTOFF_LOGPERIOD_MASSIVE_DAYS = 1.5
# Numerical resolution for monotone Eq. 23 inversion, not a physical constant;
# smooth-interval linear-inversion error is quadratic in node spacing.
MOE2017_PERIOD_CUMULATIVE_GRID_POINTS = 4096

# A correctly rounded secondary-mass product needs at most one ULP lift;
# the larger bound guards the floor-adjustment loop, not physical support.
_MASS_RATIO_FLOOR_MAX_ULP_STEPS = 8
