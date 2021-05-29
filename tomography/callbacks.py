import pathlib
import typing

import event_model as em
import numpy as np
import pandas as pd
import trackpy as tp
from bluesky.callbacks.core import CallbackBase
from bluesky.callbacks.stream import LiveDispatcher


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

    def event_page(self, doc):
        for event_doc in em.unpack_event_page(doc):
            self.event(event_doc)

    def event(self, doc, **kwargs):
        minuend = self.get_mean_frame(doc)
        result = np.subtract(minuend, self.subtrahend)
        result[result < 0.] = 0.
        new_data = {k: v for k, v in doc["data"].items() if k != self.data_key}
        new_data[self.data_key] = result
        self.process_event({'data': new_data, 'descriptor': doc["descriptor"]})
        return super(LiveDispatcher, self).event(doc)

    def get_mean_frame(self, doc):
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


class PeakTracker(CallbackBase):
    """Track the peaks on a series of images and summarize their position and intensity in a dataframe."""

    def __init__(self, data_key: str, output_dir: str, config: typing.Dict[str, typing.Any] = None):
        """Initiate the instance.

        Parameters
        ----------
        data_key :
            The key of the data to use.
        output_dir :
            The path to the directory to export cif files of the peak tracking results.
        config :
            The kwargs for the `trackpy.locate`. The "diameter" is required. If not provided, use (3, 3).
        """
        if not config:
            config = {}
        config.setdefault("diameter", (3, 3))
        super(PeakTracker, self).__init__()
        self.data_key = data_key
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self._cache = []

    def start(self, doc):
        self._cache = []
        return super(PeakTracker, self).start(doc)

    def event_page(self, doc):
        for event_doc in em.unpack_event_page(doc):
            self.event(event_doc)
        return super(PeakTracker, self).event_page(doc)

    def event(self, doc):
        image = doc["data"][self.data_key]
        df = tp.locate(image, **self.config)
        df = df.assign(frame=doc["seq_num"])
        self._cache.append(df)
        return super(PeakTracker, self).event(doc)

    def stop(self, doc):
        filename = str(self.output_dir.joinpath("{}_features.csv".format(doc["run_start"])))
        df = pd.concat(self._cache)
        df.to_csv(filename)
        return super(PeakTracker, self).stop(doc)
