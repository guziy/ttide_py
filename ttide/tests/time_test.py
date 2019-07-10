
"""
Run these tests with py.test

    py.test ttide/tests/time_test.py

"""


from ttide import time
from datetime import datetime, timedelta
import numpy as np

import logging
logging.basicConfig()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def test_date2num_with_pandas():
    """
    date2num with pandas dates
    """
    try:
        import pandas as pd
        dt = timedelta(hours=1)
        dr = pd.date_range("2001-01-01", "2001-01-02", freq=dt)
        logger.debug(dr)

        dl = np.array(list(dr))

        # should not fail
        logger.debug(time.date2num(dl))

    except ImportError as ie:
        logger.info("Skipping pandas related tests, not installed")

def test_date2num():
    d = datetime(1,1,1)
    assert time.date2num(d) == 1
    assert d.toordinal() == 1

    d_arr = np.array([d, ])
    logger.info(time.date2num(d_arr))
