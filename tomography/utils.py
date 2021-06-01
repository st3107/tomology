import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
import xarray as xr


def reshape(dataset: xr.Dataset, name: str, inverted: bool = True) -> xr.DataArray:
    """
    Reshape the xarray dataset[name] into two dimensional array. Return a reshape data array with coordinates.

    Use `shape`, `snaking`, `extents` in the dataset.attrs. The axis axis will be converted to the relative
    position to samples so that the coordinate is the negative motor position.
    """
    reshaped = _reshape(dataset[name].values, dataset.attrs["shape"], dataset.attrs["snaking"])
    coords = _get_coords(dataset.attrs, inverted=inverted)
    return xr.DataArray(reshaped, coords=coords, dims=list(coords.keys()))


def _reshape(arr: np.ndarray, shape: typing.List[int], snaking: typing.List[bool]) -> np.ndarray:
    reshaped = arr.reshape(shape)
    for i, row in enumerate(reshaped):
        if snaking[1] and i % 2 == 1:
            reshaped[i] = row[::-1]
    return reshaped


def plot_real_aspect(xarr: xr.DataArray, *args, alpha: float = 1.6, **kwargs) -> xr.plot.FacetGrid:
    """Visualize two dimensional arr as a color map. The color ranges from median - alpha * std to median +
    alpha * std."""
    facet = xarr.plot(*args, **kwargs, **get_vlim(xarr, alpha))
    set_real_aspect(facet.axes)
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


def create_atlas(df: pd.DataFrame, start_frame: int = 0, inverted: bool = True) -> xr.Dataset:
    """Create the dataset of the maps of grains.

    The dataset is like below.

        Dimensions:   (dim_0: 2, dim_1: 2, grain: 2)
        Coordinates:
          * dim_0     (dim_0) float64 6.0 0.0
          * dim_1     (dim_1) float64 2.0 0.0
          * grain     (grain) int64 0 1
        Data variables:
            maps      (grain, dim_0, dim_1) float64 1.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0
            y         (grain) int64 1 2
            x         (grain) int64 1 2
            mass      (grain) int64 1 2
            size      (grain) int64 1 2
            ecc       (grain) int64 1 2
            signal    (grain) int64 1 2
            raw_mass  (grain) int64 1 2

    Parameters
    ----------
    df :
        The dataframe of trajectories. It has columns "mass" and "frame" and attrs "extents", "shape", "snaking".
    start_frame :
        The starting number of the frame.
    inverted :
        If True, invert all axis so that maps are viewed in sample frame.

    Returns
    -------
    ds :
        A dataset created.
    """
    all_data = dict()
    groups = df.groupby("particle", sort=False)
    # get a stack of maps
    data_list = []
    names = []
    for name, group in groups:
        data = _fill_in_shape(group, start_frame)
        data_list.append(data)
        names.append(name)
    coords = _get_coords(df.attrs, inverted)
    dims = ["grain"] + list(coords.keys())
    all_data["maps"] = (dims, np.stack(data_list))
    del data_list
    # get additional data
    mean_df = groups.mean()
    for key, lst in mean_df.to_dict("list").items():
        all_data[key] = (["grain"], lst)
    # add grain to coords
    coords["grain"] = names
    return xr.Dataset(all_data, coords=coords)


def _fill_in_shape(df: pd.DataFrame, start_frame: int):
    """Create a map."""
    start_doc = df.attrs
    shape = tuple(start_doc["shape"])
    snaking_tup = tuple(start_doc["snaking"])
    snaking = snaking_tup[1] if len(snaking_tup) > 1 else False
    data = np.zeros(shape)
    for row in df.itertuples():
        seq_num = int(row.frame) - start_frame
        pos = _get_pos(seq_num, shape, snaking)
        data[pos] = row.mass
    return data


def _get_pos(seq_num: int, raster_shape: tuple, snaking: bool):
    """Get the index in the matrix."""
    pos = list(np.unravel_index(seq_num, raster_shape))
    if snaking and (pos[0] % 2):
        pos[1] = raster_shape[1] - pos[1] - 1
    return tuple(pos)


def _get_coords(start_doc: dict, inverted: bool) -> dict:
    """Get coordinates."""
    shape = start_doc["shape"]
    extents = [np.asarray(extent) - np.min(extent) for extent in start_doc["extents"]]
    if inverted:
        extents = [extent[::-1] for extent in extents]
    coords = [np.linspace(*extent, num) for extent, num in zip(extents, shape)]
    return {"dim_{}".format(i): data for i, data in enumerate(coords)}


def plot_grain_maps(atlas: xr.Dataset, **kwargs) -> xr.plot.FacetGrid:
    """Plot the grain maps from the atlas, the output from `create_atlas`."""
    facet = atlas["maps"].plot(col="grain", **kwargs)
    set_real_aspect(facet.axes)
    return facet


def set_real_aspect(axes: typing.Union[plt.Axes, typing.Iterable[plt.Axes]]) -> None:
    """Change all axes to be equal aspect."""
    if isinstance(axes, typing.Iterable):
        for ax in axes:
            set_real_aspect(ax)
    else:
        axes.set_aspect(aspect="equal", adjustable="box")
    return
