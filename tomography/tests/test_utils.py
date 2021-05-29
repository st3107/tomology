import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import tomography.utils as utils


def test_reshape():
    """Test reshape in a snaking axes case."""
    ds = xr.Dataset({"array_data": [0, 0, 0, 1, 0, 0, 0, 0, 0]})
    ds.attrs["shape"] = (3, 3)
    ds.attrs["extents"] = [(-1, 1), (-1, 1)]
    ds.attrs["snaking"] = [False, True]
    res = utils.reshape(ds, "array_data")
    expected = np.zeros((3, 3))
    expected[1, 2] = 1
    assert np.array_equal(res.values, expected)


def test_plot_real_aspect():
    """Test plot_real_aspect on a zero image."""
    data = xr.DataArray(np.zeros((5, 5)))
    data[2, 2] = 1
    utils.plot_real_aspect(data)
    plt.clf()


def test_annotate_peaks():
    """Test annotate_peaks in a toy case."""
    df = pd.DataFrame(
        {"y": [2], "x": [2], "mass": [1], "size": [1], "ecc": [1], "signal": [1], "raw_mass": [1], "frame": [0]}
    )
    image = xr.DataArray(np.zeros((5, 5)))
    image[2, 2] = 1
    utils.annotate_peaks(df, image)
    plt.clf()
