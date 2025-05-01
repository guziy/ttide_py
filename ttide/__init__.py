"""This package now holds the ttide_py API.

The included functions are:

t_tide : The t_tide harmonic analysis function.

t_predic : The t_tide harmonic fit function.

TTideCon : The t_tide constituents class (returned by t_tide).

"""

from ttide.base import TTideCon

from .t_predic import t_predic
from .t_tide import t_tide

__version__ = "0.3.5"
