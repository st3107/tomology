from pprint import pformat

import databroker
import matplotlib.pyplot as plt
import numpy as np
from pdfstream.callbacks.composer import gen_stream
from pkg_resources import resource_filename

import tomography.callbacks as cbs


def print_doc(name, doc):
    print(name, "\n", pformat(doc))


def test_ImageProcessor():
    """Check that ImageProcessor is correctly subtracting images."""
    data_key = "pe1_image"

    def verify(_name, _doc):
        if _name != "event":
            return
        data = _doc["data"][data_key]
        assert isinstance(data, list)
        assert np.array_equal(np.asarray(data), np.zeros((3, 3)))

    ip = cbs.ImageProcessor(data_key=data_key, subtrahend=np.ones((3, 3)))
    ip.subscribe(verify, name="event")

    frames = np.ones((2, 3, 3))
    for name, doc in gen_stream([{data_key: frames}], {}):
        ip(name, doc)


def test_PeakTracker(tmpdir):
    """Check that PeakTrack works without errors."""
    # make images
    image_file = resource_filename("tomography", "data/image.png")
    image = plt.imread(image_file)
    images = [image] * 3
    # check if db friendly
    db = databroker.v2.temp()
    # test
    data_key = "pe1_image"
    pt = cbs.PeakTracker(data_key=data_key, diameter=(11, 11))
    pt.subscribe(db.v1.insert)
    data = [{data_key: image} for image in images]
    for name, doc in gen_stream(data, {}):
        pt(name, doc)
    # check output
    df = cbs.get_dataframe(db[-1].primary)
    print(df.to_string())
