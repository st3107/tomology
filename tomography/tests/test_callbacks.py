import pytest
import numpy as np
import tomography.callbacks as cbs
from pdfstream.callbacks.composer import gen_stream
import pandas as pd


def test_ImageProcessor():
    """Check that ImageProcessor is correctly subtracting images."""
    data_key = "pe1_image"

    def verify(_name, _doc):
        if _name != "event":
            return
        assert np.array_equal(_doc["data"][data_key], np.zeros((3, 3)))

    ip = cbs.ImageProcessor(data_key=data_key, subtrahend=np.ones((3, 3)))
    ip.subscribe(verify, name="event")

    frames = np.ones((2, 3, 3))
    for name, doc in gen_stream([{data_key: frames}], {}):
        ip(name, doc)


def test_PeakTracker(tmpdir):
    """Check that PeakTrack works without errors."""
    # make images
    images = [np.zeros((16, 16), dtype=int) for _ in range(3)]
    # test
    data_key = "pe1_image"
    pt = cbs.PeakTracker(data_key=data_key, output_dir=str(tmpdir))
    data = [{data_key: image} for image in images]
    for name, doc in gen_stream(data, {}):
        pt(name, doc)
    # print
    df = pd.read_csv(tmpdir.listdir()[0])
    print(df.to_string())
