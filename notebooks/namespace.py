import dask
import typing
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp
import pandas as pd
import databroker
from bluesky.callbacks.stream import LiveDispatcher
import event_model
from bluesky.callbacks.best_effort import BestEffortCallback


DB = databroker.catalog["xpd"]
UID = pd.read_csv("data/uid.csv")
DASK_SETTING = dask.config.set(scheduler='threads')  # overwrite default with threaded scheduler


class TrackPyStream(LiveDispatcher):
    """A secondary stream using trackpy.

    It will read the image in event, subtract the background image, fill the negative pixel with zero and find the peaks on the image. At last, it will save the peak info in a data frame.
    """

    def __init__(self, output_dir: str, image_key: str, background: np.ndarray, **kwargs):
        self.output_dir = output_dir
        self.background = background
        self.image_key = image_key
        self.kwargs = kwargs
        self._dfs = []
        self.summary = pd.DataFrame()

    def start(self, doc, _md=None):
        self._dfs = []
        self.summary = pd.DataFrame()
        return super(TrackPyStream, self).start(doc, _md)

    def event(self, doc, _md=None):
        frames = doc["data"][image_key]
        avg_frame = np.mean(frames, axis=0)
        clean_frame = avg_frame - self.background
        df = tp.locate(clean_frame, **kwargs)
        df = df.assign(frame=doc["seq_num"])
        self._dfs.append(df)
        self.process_event({'data': doc["data"], 'descriptor': doc["descriptor_id"]})

    def event_page(self, doc, _md=None):
        for event in event_model.unpack_event_page(doc):
            self.event(event)

    def stop(self, doc, _md=None):
        self.summary = pd.concat(self._dfs)
        self.summary.to_csv("{}-features.csv".format(doc["run_start"]))
        return super(TrackPyStream, self).stop(doc, _md)


def process_images_and_track_peaks(images: xr.DataArray, background_image: xr.DataArray, *args, **kwargs) -> pd.DataFrame:
    """Process a series of image and track peaks. Return a summarized feature dataframe."""
    dfs = [process_image_and_find_peaks(image, background_image, *args, **kwargs) for image in images]
    return pd.concat(dfs)


def process_image_and_find_peaks(image: xr.DataArray, background_image: xr.DataArray, *args, **kwargs) -> pd.DataFrame:
    """Process the image and find peak on it."""
    processed_image = average_subtract_fill_zero(image, background_image).compute()
    df = my_locate(processed_image, *args, **kwargs)
    return df


def average_subtract_fill_zero(image1: xr.DataArray, image2: xr.DataArray) -> xr.DataArray:
    """Average the frames, subtract image1 by image2 and fill the negative value with zero. Return the result."""
    image1 = image1.mean(axis=-3).squeeze()
    image2 = image2.mean(axis=-3).squeeze()
    image3 = image1.copy()
    image3.values = image1.values - image2.values
    image3 = image3.where(image3 > 0., np.zeros_like(image3))
    return image3.compute()


def my_locate(frame: xr.DataArray, diameter: int = 15, percentile=90, separation=100, **kwargs) -> pd.DataFrame:
    """A wrapper of the tp.locate. It uses a different default values. It will assign the scalar coordinates to data frame as a column."""
    df = tp.locate(frame.values, diameter, percentile=90, separation=100, **kwargs)
    columns = {key: arr.item() for key, arr in frame.coords.items() if arr.ndim == 0}
    df = df.assign(**columns)
    return df


def reshape(dataset: xr.Dataset, name: str) -> xr.DataArray:
    """
    Reshape the xarray dataset[name] into two dimensional array. Return a reshape data array with coordinates.

    Use `shape`, `snaking`, `extents` in the dataset.attrs. The axis axis will be converted to the relative position
    to samples so that the coordinate is the negative motor position.
    """
    reshaped = _reshape(dataset[name].values, dataset.attrs["shape"], dataset.attrs["snaking"])
    extents = dataset.attrs["extents"]
    shape = dataset.attrs["shape"]
    coords = [-np.linspace(*extent, num) for extent, num in zip(extents, shape)]
    return xr.DataArray(reshaped, coords={"y": coords[0], "x": coords[1]}, dims=["y", "x"])


def _reshape(arr: np.ndarray, shape: typing.List[int], snaking: typing.List[bool]) -> np.ndarray:
    reshaped = arr.reshape(shape)
    for i, row in enumerate(reshaped):
        if snaking[1] and i % 2 == 1:
            reshaped[i] = row[::-1]
    return reshaped


def my_color_map(xarr: xr.DataArray, *args,  alpha: float=1.6, **kwargs) -> xr.plot.FacetGrid:
    """Visualize two dimensional arr as a color map. The color ranges from median - alpha * std to median + alpha * std."""
    facet = xarr.plot(*args, **kwargs, **get_vlim(xarr, alpha))
    facet.axes.set_aspect(1, adjustable="box")
    return facet


def get_vlim(xarr: xr.DataArray, alpha: float = 1.6) -> dict:
    """Get vmin, vmax using median and std."""
    median = xarr.median()
    std = xarr.std()
    return {"vmin": median - alpha * std, "vmax": median + alpha * std}


def my_annotate_image(df, image: xr.DataArray, ax, **kwargs):
    """A function wrapping the tp.annotate. Use different default setting."""
    imshow_style = dict(**get_vlim(image), cmap="viridis")
    imshow_style.update(kwargs)
    tp.annotate(df, image, ax=ax, imshow_style=imshow_style)
    ax.set_ylim(ax.get_ylim()[::-1])
