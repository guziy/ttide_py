import numpy as np
from datetime import timedelta
import logging

from ttide.t_tide import t_tide
from ttide.t_predic import t_predic
from ttide.tests import base as bmod
import copy

from ttide.time import date2num

cases = copy.deepcopy(bmod.cases)

t = np.arange(0, 60, 0.3)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_tpredic_with_pandas_dates():
    """
    Testing t_predic with dates from pandas
    """
    try:
        import pandas as pd
        dt = timedelta(hours=1)
        dr = pd.date_range("2001-01-01", "2001-01-02", freq=dt)
        logger.debug(dr)

        dl = np.asarray(list(dr))
        const_names = np.asarray(["M2  ".encode(),])
        const_freqs = np.asarray([bmod.m2_freq, ])
        const_ampha = np.asarray([[5, 1, 0, 1], ])

        # set to a small value, but TODO: investigate how 0 could be allowed
        lat = 0.1

        # should not fail
        res = t_predic(dl, const_names, const_freqs, const_ampha, lat=lat)

        logger.debug(res)

        # length of the result should be equal to the length of the time vector
        assert len(res) == len(dl)


        # harmonic analysis
        tcon = t_tide(res, constitnames=["M2"], stime=date2num(dl[0]), lat=lat,
                      synth=0)

        logger.debug(tcon)

        err = np.mean((tcon["xout"].squeeze() - res) ** 2) ** 0.5

        logger.debug([err, tcon["xout"], res])
        assert err <= 1.e-6

    except ImportError:
        logger.info("Not testing t_predict with pandas, probably not installed")


def compare_vec2file(x0, fname):
    x1 = np.loadtxt(bmod.testdir + 'data/predict/' + fname)
    if len(x1) == 2 * len(x0):
        x1 = x1.view(complex)
    assert (np.abs(x0 - x1) < 1e-2).all(), (f"Test failed on file '{fname}'")


def gen_predict_tests(make_data=False):

    for kwargs, fname in cases:
        kwargs['out_style'] = None
        out = t_tide(**kwargs)
        xout = out.t_predic(t)
        if make_data:
            np.savetxt(bmod.testdir + 'data/predict/' + fname, xout.view(float), fmt='%0.5f')
            yield None
        else:
            yield compare_vec2file, xout, fname

def test_predic():
    # no files are written to disk by default (make_data=False)
    for f, vec, fname in gen_predict_tests():
        f(vec, fname)



if __name__ == '__main__':
    pass
    ###
    # This block generates the output files.
    # USE WITH CAUTION!
    # for tst in gen_predict_tests(make_data=True):
    #     pass
    #
    # for f, vec, fname in gen_predict_tests():
    #     f(vec, fname)
