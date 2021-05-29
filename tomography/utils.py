import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
import xarray as xr


def reshape(dataset: xr.Dataset, name: str) -> xr.DataArray:
    """
    Reshape the xarray dataset[name] into two dimensional array. Return a reshape data array with coordinates.

    Use `shape`, `snaking`, `extents` in the dataset.attrs. The axis axis will be converted to the relative position to samples so that the coordinate is the negative motor position.
    """
    reshaped = _reshape(dataset[name].values, dataset.attrs["shape"], dataset.attrs["snaking"])
    extents = dataset.attrs["extents"]
    shape = dataset.attrs["shape"]
    coords = [np.linspace(*extent, num) for extent, num in zip(extents, shape)]
    coord_data = {"dim_{}".format(i): data for i, data in enumerate(coords)}
    return xr.DataArray(reshaped, coords=coord_data, dims=list(coord_data.keys()))


def _reshape(arr: np.ndarray, shape: typing.List[int], snaking: typing.List[bool]) -> np.ndarray:
    reshaped = arr.reshape(shape)
    for i, row in enumerate(reshaped):
        if snaking[1] and i % 2 == 1:
            reshaped[i] = row[::-1]
    return reshaped


def plot_real_aspect(xarr: xr.DataArray, *args, alpha: float = 1.6, **kwargs) -> xr.plot.FacetGrid:
    """Visualize two dimensional arr as a color map. The color ranges from median - alpha * std to median + alpha * std."""
    facet = xarr.plot(*args, **kwargs, **get_vlim(xarr, alpha))
    facet.axes.set_aspect(1, adjustable="box")
    return facet


def get_vlim(xarr: xr.DataArray, alpha: float) -> dict:
    """Get vmin, vmax using mean and std."""
    mean = xarr.mean()
    std = xarr.std()
    return {"vmin": max(0., mean - alpha * std), "vmax": mean + alpha * std}


def annotate_peaks(df: pd.DataFrame, image: xr.DataArray, ax: plt.Axes = None, alpha: float = 1.6,
                   **kwargs) -> None:
    """A function wrapping the tp.annotate. Use different default setting."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    imshow_style = dict(**get_vlim(image, alpha=alpha), cmap="viridis")
    imshow_style.update(kwargs)
    tp.annotate(df, image, ax=ax, imshow_style=imshow_style)
    ax.set_ylim(*ax.get_ylim()[::-1])
