import typing

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
import xarray as xr
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

_VERBOSE = 1


def set_verbose(level: int) -> None:
    global _VERBOSE
    _VERBOSE = level


def my_print(*args, **kwargs) -> None:
    if _VERBOSE > 0:
        print(*args, **kwargs)


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
    if len(snaking) > 1 and not snaking[1]:
        new_reshaped = reshaped.copy()
        for i, row in enumerate(reshaped):
            if i % 2 == 1:
                new_reshaped[i] = row[::-1]
        reshaped = new_reshaped
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


def create_atlas(df: pd.DataFrame, start_frame: int = 1, inverted: bool = True,
                 excluded: set = frozenset(["frame"])) -> xr.Dataset:
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
    excluded :
        The excluded the columns in dataframe.

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
        if key not in excluded:
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
        if seq_num < 0:
            raise ValueError(
                "Frame number smaller than the starting number: {}, {}".format(int(row.frame), start_frame))
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


def plot_grain_maps(atlas: xr.Dataset, name: str = "maps", col: str = "grain", **kwargs) -> xr.plot.FacetGrid:
    """Plot the grain maps from the atlas, the output from `create_atlas`."""
    kwargs.setdefault("col", col)
    facet = atlas[name].plot(**kwargs)
    set_real_aspect(facet.axes)
    # automatically adjust size
    ratio = facet.axes.flatten()[0].get_data_ratio()
    num = atlas[name].sizes[col]
    col_wrap = kwargs.get("col_wrap", 1)
    facet.fig.set_size_inches((1 * num / col_wrap, 1 * ratio * col_wrap))
    facet.set_titles("{value}")
    return facet


def set_real_aspect(axes: typing.Union[plt.Axes, typing.Iterable[plt.Axes]]) -> None:
    """Change all axes to be equal aspect."""
    if isinstance(axes, typing.Iterable):
        for ax in axes:
            set_real_aspect(ax)
    else:
        axes.set_aspect(aspect="equal", adjustable="box")
    return


def pixel_to_Q(d1: np.ndarray, d2: np.ndarray, ai: AzimuthalIntegrator) -> xr.DataArray:
    """Map pixel position (d1, d2) to Q in nm-1."""
    arr = xr.DataArray(ai.qCornerFunct(d1, d2))
    arr.attrs["standard_name"] = "Q"
    arr.attrs["units"] = "nm$^{-1}$"
    return arr


def assign_Q_to_atlas(atlas: xr.Dataset, ai: AzimuthalIntegrator) -> xr.Dataset:
    """Assign Q grid to atlas"""
    q = pixel_to_Q(atlas["y"].values, atlas["x"].values, ai)
    dims = atlas["y"].dims
    return atlas.assign({"Q": (dims, q)})


def create_atlas_dask(frames: xr.DataArray, windows: pd.DataFrame, verbose: bool = False) -> xr.DataArray:
    """Create a list of tasks to compute the grain maps. Each task is one grain map.

    The dataframe has columns "x", "y", "dx", "dy". Each row is a window on the frames. The window is
    (x - dx, x + dx) in horizontal and (y - dy, y + dy) in vertical. The return is a list of dask arrays.

    Parameters
    ----------
    windows :
        The dataframe with columns "x", "y", "dx", "dy".
    frames :
        One frame.
    verbose :
        Whether to report status or not.

    Returns
    -------
    tasks :
        A list of dask arrays.
    """
    # make the numbers ints
    windows = windows[["x", "y", "dx", "dy"]].apply(np.round).applymap(int)
    # get limits
    nf, _, ny, nx = frames.shape
    # create tasks
    maps = []
    for i, frame in enumerate(frames):
        if verbose:
            print("process frame {}/{}.".format(i + 1, nf), end="\r")
        frame = frame.compute()
        mean_frames = []
        for row in windows.itertuples():
            slice_y = slice(max(row.y - row.dy, 0), min(row.y + row.dy, ny))
            slice_x = slice(max(row.x - row.dx, 0), min(row.x + row.dx, nx))
            mean_frame = frame[:, slice_y, slice_x].mean()
            mean_frames.append(mean_frame)
        maps.append(xr.concat(mean_frames, dim="grain"))
    return xr.concat(maps, dim="time")


def create_dataset(maps: xr.DataArray, windows: pd.DataFrame, metadata: dict, inverted: bool = True) -> xr.Dataset:
    """Create the dataset from the results of create_atlas_dask."""
    ds: xr.Dataset = windows.to_xarray()
    old_dims = list(ds.dims)
    new_values = reshape_to_matrix(maps.values, metadata)
    new_coords = _get_coords(metadata, inverted=inverted)
    new_dims = list(new_coords.keys()) + old_dims
    ds = ds.assign_coords(new_coords)
    ds = ds.assign({"maps": (new_dims, new_values)})
    ds = ds.rename({old_dims[0]: "grain"})
    return ds


def reshape_to_matrix(arr: np.ndarray, metadata: dict) -> np.ndarray:
    reshaped = np.apply_along_axis(
        lambda x: _reshape(x, metadata["shape"], metadata.get("snaking", [])),
        0,
        arr
    )
    return reshaped


def min_and_max_along_time(data: xr.DataArray) -> xr.DataArray:
    """Extract the minimum and maximum values of each pixel in a series of mean frames. Return a data array.
    First is the min array and the second is the max array."""
    min_arr = max_arr = np.mean(data[0].values, axis=0)
    num = data.shape[0]
    my_print("Process frame: {} / {}".format(1, num), end="\r")
    for i in range(1, len(data)):
        arr = np.mean(data[i].values, axis=0)
        min_arr = np.fmin(min_arr, arr)
        max_arr = np.fmax(max_arr, arr)
        my_print("Process frame: {} / {}".format(i + 1, num), end="\r")
    return xr.DataArray(np.stack([min_arr, max_arr]))


def reshape_to_xarray(arr: xr.DataArray, metadata: dict) -> xr.DataArray:
    """Reshape a 1D data array to an 2D data array according to the metadata about the bluesky plan.
    The coordinates will also be reshaped."""
    data = reshape_to_matrix(arr.values, metadata)
    coords = {name: xr.DataArray(reshape_to_matrix(coord.values, metadata)) for name, coord in arr.coords.items()}
    return xr.DataArray(data, coords=coords)


def reformat_data(ds: xr.Dataset) -> xr.Dataset:
    """Reformat the data loaded from the databroker."""
    frames = np.arange(0, ds["time"].shape[0], 1)
    return ds.rename_dims({"time": "frame"}).reset_index("time").drop("time_").assign_coords({"frame": frames})


def draw_windows(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw windows on the axes."""
    for row in df.itertuples():
        xy = (row.x - row.dx, row.y - row.dy)
        width = row.dx * 2 - 1
        height = row.dy * 2 - 1
        patch = patches.Rectangle(xy, width=width, height=height, linewidth=1, edgecolor="r", fill=False)
        ax.add_patch(patch)
    return


def create_windows_from_size(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    df2 = pd.DataFrame()
    df2["x"] = df["x"].apply(np.round).apply(int)
    df2["y"] = df["y"].apply(np.round).apply(int)
    df2["dx"] = df2["dy"] = (df["size"] * multiplier).apply(np.round).apply(int)
    return df2


def create_windows_from_width(df: pd.DataFrame, width: int) -> pd.DataFrame:
    df2 = pd.DataFrame()
    df2["x"] = df["x"].apply(np.round).apply(int)
    df2["y"] = df["y"].apply(np.round).apply(int)
    df2["dx"] = df2["dy"] = int(np.round(width))
    return df2


def track_peaks(frames: xr.DataArray, windows: pd.DataFrame) -> xr.DataArray:
    """Create a list of tasks to compute the grain maps. Each task is one grain map."""
    # get limits
    nf, _, ny, nx = frames.shape
    # create tasks
    maps = []
    for i, frame in enumerate(frames):
        my_print("Process frame {} / {}.".format(i + 1, nf), end="\r")
        frame = frame.compute()
        mean_frames = []
        for row in windows.itertuples():
            slice_y = slice(max(row.y - row.dy, 0), min(row.y + row.dy, ny))
            slice_x = slice(max(row.x - row.dx, 0), min(row.x + row.dx, nx))
            mean_frame = frame[:, slice_y, slice_x].mean()
            mean_frames.append(mean_frame)
        maps.append(xr.concat(mean_frames, dim="grain"))
    return xr.concat(maps, dim="time")


def create_grain_maps(frames: xr.DataArray, windows: pd.DataFrame, metadata: dict,
                      inverted: bool = True) -> xr.Dataset:
    """Create grain maps from frames.

    Parameters
    ----------
    frames :
        The dask array of raw diffraction frames.
    windows :
        The data frame of x, y, dx, dy. The x, y positions in pixels and the half width of the window.
    metadata :
        The start document of the run.
    inverted :
        Use x[::-1] and y[::-1] in grain maps.

    Returns
    -------
    ds :
        The dataset of grain maps.
    """
    maps = track_peaks(frames, windows)
    return create_dataset(maps, windows, metadata, inverted=inverted)


def select_frames(image_sum_data: xr.DataArray, metadata: dict, start_row: int = 0, end_row: int = None, **kwargs) -> None:
    """Select the frames according to the summation of the intensity on the image."""
    image_sum_matrix = reshape_to_xarray(image_sum_data, metadata)
    kwargs.setdefault("size", 8)
    if end_row is None:
        end_row = image_sum_matrix.shape[0]
    image_sum_matrix = image_sum_matrix[start_row:end_row]
    facet = image_sum_matrix.plot(**kwargs)
    index = image_sum_matrix.frame.values
    start_index, end_index = index.min(), index.max()
    facet.axes.set_title("From frame {} to frame {}".format(start_index, end_index))
    set_real_aspect(facet.axes)
