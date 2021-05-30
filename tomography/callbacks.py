import typing
from collections import ChainMap

import event_model as em
import numpy as np
import pandas as pd
from bluesky.callbacks.stream import LiveDispatcher
from databroker.core import BlueskyEventStream
from databroker.v2 import Broker
from trackpy import link, locate


def subtract_images(minuend: np.ndarray, subtrahend: np.ndarray) -> np.ndarray:
    """Subtract the image and fill zero in negative pixles."""
    res = np.subtract(minuend, subtrahend)
    res[res < 0.] = 0.
    return res


class ImageProcessor(LiveDispatcher):
    """A callback to average frames of images, subtract it by another image, and emit the document."""

    def __init__(self, data_key: str, subtrahend: np.ndarray, scale: float = 1.):
        """Initiate the instance.

        Parameters
        ----------
        data_key :
            The key of the data to use.
        subtrahend :
            The 2d image as a subtrahend.
        scale :
            The scale factor of the subtrahend.
        """
        super(ImageProcessor, self).__init__()
        self.data_key = data_key
        self.subtrahend = np.asarray(subtrahend) * scale

    def start(self, doc, _md=None):
        if _md is None:
            _md = {}
        _md = ChainMap({"analysis_stage": ImageProcessor.__name__}, _md)
        super(ImageProcessor, self).start(doc, _md=_md)

    def event_page(self, doc):
        for event_doc in em.unpack_event_page(doc):
            self.event(event_doc)

    def event(self, doc, **kwargs):
        minuend = self.get_mean_frame(doc)
        result = subtract_images(minuend, self.subtrahend)
        new_data = {k: v for k, v in doc["data"].items() if k != self.data_key}
        new_data[self.data_key] = result.tolist()
        self.process_event({'data': new_data, 'descriptor': doc["descriptor"]})

    def get_mean_frame(self, doc) -> np.ndarray:
        frames = np.asarray(doc["data"][self.data_key])
        n = np.ndim(frames)
        if n < 2:
            raise ValueError("The dimension of {} < 2.".format(self.data_key))
        elif n == 2:
            mean_frame = frames
        elif n == 3:
            mean_frame = np.mean(frames, axis=0)
        else:
            mean_frame = np.mean(frames, axis=tuple((i for i in range(n - 2))))
        return mean_frame


class PeakTracker(LiveDispatcher):
    """Track the peaks on a series of images and summarize their position and intensity in a dataframe."""

    def __init__(self, data_key: str, diameter: typing.Union[int, tuple], **kwargs):
        """Initiate the instance.

        Parameters
        ----------
        data_key :
            The key of the data to use.
        diameter :
            The pixel size of the peak.
        kwargs :
            The other kwargs for the `trackpy.locate`.
        """
        kwargs["diameter"] = diameter
        super(PeakTracker, self).__init__()
        self.data_key = data_key
        self.config = kwargs

    def start(self, doc, _md=None):
        _md = {"analysis_stage": PeakTracker.__name__}
        super(PeakTracker, self).start(doc, _md=_md)

    def event_page(self, doc):
        for event_doc in em.unpack_event_page(doc):
            self.event(event_doc)

    def event(self, doc, **kwargs):
        image = doc["data"][self.data_key]
        df = locate(image, **self.config)
        df = df.assign(frame=doc["seq_num"])
        for data in df.to_dict("records"):
            self.process_event({"data": data, "descriptor": doc["descriptor"]})


def get_dataframe(stream: BlueskyEventStream, drop_time: bool = True) -> pd.DataFrame:
    """Get the dataframe from the stream. Drop the time column."""
    df: pd.DataFrame = stream.read().to_dataframe()
    return df.reset_index(drop=drop_time)


class TrackLinker(LiveDispatcher):
    """Track the peaks in frame and link them in trajectories.

    When a stop is received, the data will be pulled from the databroker and processed. Then, the dataframe will
    be emitted row by row.
    """

    def __init__(self, *, db: Broker = None, search_range: typing.Union[float, tuple], **kwargs):
        """Create the instance.

        Parameters
        ----------
        db :
            The databroker. If None, this callback does nothing.
        search_range :
            The search_range in `trackpy.link`.
        kwargs :
            Other kwargs in `trackpy.link`.
        """
        super(TrackLinker, self).__init__()
        kwargs["search_range"] = search_range
        self.config = kwargs
        self.db = db

    def start(self, doc, _md=None):
        _md = {"analysis_stage": TrackLinker.__name__}
        super(TrackLinker, self).start(doc, _md=_md)

    def event_page(self, doc):
        return

    def event(self, doc, **kwargs):
        return

    def stop(self, doc, _md=None):
        features = get_dataframe(self.db[doc["run_start"]].primary)
        df = link(features, **self.config)
        descriptor = next(iter(self.raw_descriptors.keys()))
        for data in df.to_dict("records"):
            self.process_event({"data": data, "descriptor": descriptor})
        super(TrackLinker, self).stop(doc, _md=None)
