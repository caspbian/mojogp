"""Constants for GP kernel library."""

from .profiling_config import PROFILING

alias float_dtype = DType.float32

alias PI = 3.14159265358979323846
alias SQRT3 = 1.7320508075688772
alias SQRT5 = 2.23606797749979
alias MAX_SUPPORTED_DIM = 32

# Kernel type constants
alias KERNEL_TYPE_RBF = 0
alias KERNEL_TYPE_MATERN32 = 1
alias KERNEL_TYPE_MATERN52 = 2
alias KERNEL_TYPE_MATERN12 = 3
alias KERNEL_TYPE_PERIODIC = 4
alias KERNEL_TYPE_RQ = 5
alias KERNEL_TYPE_LINEAR = 6
alias KERNEL_TYPE_POLYNOMIAL = 7

# Categorical kernel type constants
alias CAT_KERNEL_GD = 0   # Gower Distance (1 param per variable)
alias CAT_KERNEL_CR = 1   # Continuous Relaxation (L params per variable)
alias CAT_KERNEL_EHH = 2  # Exponential Homoscedastic Hypersphere (L(L-1)/2 params)
alias CAT_KERNEL_HH = 3   # Homoscedastic Hypersphere (L(L-1)/2 params, allows negative)
alias CAT_KERNEL_FE = 4   # Fully Exponential (L(L+1)/2 params)

# Maximum supported categorical variables and levels
alias MAX_CAT_VARS = 16       # Max number of categorical variables
alias MAX_CAT_LEVELS = 32     # Max number of levels per categorical variable
alias MAX_CAT_CORR_SIZE = 1024  # Max total flattened correlation matrix size (sum of L_i^2)
