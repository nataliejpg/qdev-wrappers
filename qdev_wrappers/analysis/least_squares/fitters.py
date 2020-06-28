from qdev_wrappers.analysis.least_squares.base import LeastSquaresBase
from qdev_wrappers.analysis.least_squares import guess

# TODO: add fitter for resonators (nataliejpg)


class ExpDecayFitter(LeastSquaresBase):
    MODELPARAMETERS = {'a': {'label': '$a$'},
                       'T': {'label': '$T$', 'unit': 's'},
                       'c': {'label': '$c$'}}
    MODEL = {'str': r'$f(x) = a \exp(-x/T) + c$',
             'np': 'a * np.exp(-x / T) + c'}
    GUESS = guess.exp_decay
    """
    Least Squares Fitter which fits to an exponential decay using the equation
        a * np.exp(-x / T) + c
    given the measured results and the values of x. Useful for T1 fitting.
    """


class ExpDecayBaseFitter(LeastSquaresBase):
    MODELPARAMETERS = {'a': {'label': '$a$', 'unit': 'V'},
                       'p': {'label': '$p$'},
                       'b': {'label': '$b$', 'unit': 'V'}}
    MODEL = {'str': r'$f(x) = A p^x + B$',
             'np': 'a * p**x + b'}
    GUESS = guess.power_decay
    """
    Least Squares Fitter which fits to an exponential using the equation
        a * p**x + b
    and given the measured results and the values of x. Useful for fitting
    benchmarking results.
    """


class CosFitter(LeastSquaresBase):
    MODELPARAMETERS = {'a': {'label': '$a$'},
                       'w': {'label': r'$\omega$', 'unit': 'Hz'},
                       'p': {'label': r'$\phi$'},
                       'c': {'label': '$c$', 'unit': ''}}
    MODEL = {'str': r'$f(x) = a\cos(\omega x + \phi)+c$',
             'np': 'a * np.cos(w * x + p) + c'}
    GUESS = guess.cosine
    """
    Least Squares Fitter which fits to a cosine using the equation
        a * np.cos(w * x + p) + c
    and given the measured results and the values of x. Useful for fitting
    Rabi oscillations.
    """


class ExpDecaySinFitter(LeastSquaresBase):
    MODELPARAMETERS = {'a': {'label': '$a$', 'unit': ''},
                       'T': {'label': '$T$', 'unit': 's'},
                       'w': {'label': r'$\omega$', 'unit': 'Hz'},
                       'p': {'label': r'$\phi$'},
                       'c': {'label': '$c$'}}
    MODEL = {
        'str': r'$f(x) = a \exp(-x / T) \sin(\omega x + \phi) + c$',
        'np': 'a * np.exp(-x / T) * np.sin(w * x + p) + c'}
    GUESS = guess.exp_decay_sin
    """
    Least Squares Fitter which fits to an exponentially decaying sine using the
    equation
        a * np.exp(-x / T) * np.sin(w * x + p) + c
    and given the measured results and the values of x. Useful for fitting
    Ramsey oscillations to find T2*.
    """
