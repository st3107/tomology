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
    # plt.show()
    plt.clf()


def test_annotate_peaks():
    """Test annotate_peaks in a toy case."""
    df = pd.DataFrame(
        {"y": [2], "x": [2], "mass": [1], "size": [1], "ecc": [1], "signal": [1], "raw_mass": [1], "frame": [0]}
    )
    image = xr.DataArray(np.zeros((5, 5)))
    image[2, 2] = 1
    utils.annotate_peaks(df, image)
    # plt.show()
    plt.clf()


def test_create_atlas():
    """Test create_atlas by plotting the figure out. Test assign_Q_to_atlas"""
    df = pd.DataFrame(
        {"y": [1, 2], "x": [1, 2], "mass": [1, 2], "size": [1, 2], "ecc": [1, 2], "signal": [1, 2],
         "raw_mass": [1, 2], "frame": [1, 3], "particle": [0, 1]}
    )
    df.attrs["shape"] = (2, 2)
    df.attrs["extents"] = [(-3, 3), (-1, 1)]
    df.attrs["snaking"] = (False, True)
    ds = utils.create_atlas(df)
    # print(ds)
    facet = utils.plot_grain_maps(ds)
    facet.fig.set_size_inches(4, 4)
    # facet.fig.show()
    plt.clf()
    # test
    ai = utils.AzimuthalIntegrator(detector="Perkin", wavelength=2 * np.pi)
    ds2 = utils.assign_Q_to_atlas(ds, ai)
    print(ds2)


def test_map_to_Q():
    """Test map_to_Q_vector."""
    # test numpy
    d1 = np.array([0, 1])
    d2 = np.array([0, 1])
    ai = utils.AzimuthalIntegrator(detector="Perkin", wavelength=2 * np.pi)
    q = utils.pixel_to_Q(d1, d2, ai)
    assert q.shape == (2,)


def test_Calculator():
    c = utils.Calculator()

    c.frames_arr = xr.DataArray([[[[1, 0], [0, 0]]], [[[0, 0], [1, 1]]]])
    c.calc_dark_and_light_from_frames_arr()
    expect1 = xr.DataArray([[[0, 0], [0, 0]], [[1, 0], [1, 1]]], coords={"dim_0": ["dark", "light"]})
    assert c.dark_and_light.equals(expect1)

    c.calc_peaks_from_light_frame(1, noise_size=0)
    expect2 = pd.DataFrame(columns=["y", "x", "mass", "size", "ecc", "signal", "raw_mass"])
    assert c.peaks.equals(expect2)

    c.peaks = pd.DataFrame([[1.5, 0.5, 3], [0.5, 0.5, 2], [1.5, 1.5, 1]],
                           columns=["y", "x", "mass"])
    c.calc_windows_from_peaks(100, 0)
    expect3 = pd.DataFrame([[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0]], columns=["y", "dy", "x", "dx"])
    assert c.windows.equals(expect3)

    c.calc_intensity_in_windows()
    expect4 = xr.DataArray([[0., 1.], [1., 0], [0., 1.]], dims=["grain", "frame"])
    assert c.intensity.equals(expect4)

    c.metadata = {"shape": [2, 2], "extents": [(-1, 0), (1, 2)], "snaking": (False, True)}
    c.intensity = xr.DataArray([[1, 2, 3, 4]])
    c.calc_grain_maps_by_reshaping()
    expect5 = xr.DataArray([[[1, 2], [4, 3]]], coords={"dim_1": [-1., 0.], "dim_2": [1., 2.]}, dims=["dim_0", "dim_1", "dim_2"])
    assert c.grain_maps.equals(expect5)
