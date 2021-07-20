from pathlib import Path

import numpy as np
from pdfstream.callbacks.composer import gen_stream

import tomography.servers as servers


def test_Extremum(tmpdir):
    """Test if it computes the correct min and max."""
    base = Path(str(tmpdir))
    # callback
    config = servers.ExtremumConfig()
    data_key = "image"
    config.parser["EXTREMUM"]["directory"] = str(base)
    config.parser["EXTREMUM"]["data_key"] = data_key
    extremum = servers.Extremum(config)
    # create date stream and let it flow through the callback
    shape = (2, 3, 3)
    data = [{data_key: np.zeros(shape)}, {data_key: np.ones(shape)}]
    metadata = {"sample_name": "A"}
    stream = gen_stream(data, metadata)
    for name, doc in stream:
        extremum(name, doc)
    # check the files
    file1 = base.joinpath("A_1_min.npy")
    file2 = base.joinpath("A_1_max.npy")
    file3 = base.joinpath("A_2_min.npy")
    file4 = base.joinpath("A_2_max.npy")
    for f in (file1, file2, file3, file4):
        assert f.is_file
    # check the output
    arr1 = np.load(file1)
    arr2 = np.load(file2)
    arr3 = np.load(file3)
    arr4 = np.load(file4)
    shape2 = (3, 3)
    assert np.array_equal(arr1, np.zeros(shape2, float))
    assert np.array_equal(arr2, np.zeros(shape2, float))
    assert np.array_equal(arr3, np.zeros(shape2, float))
    assert np.array_equal(arr4, np.ones(shape2, float))


def test_ExtremumServer(tmpdir):
    """Test if it can write and read the cfg file"""
    base = Path(str(tmpdir))
    ss = servers.Servers()
    cfg_file = base.joinpath("test.ini")
    ss.create_extremum_config(str(cfg_file))
    print(cfg_file.read_text())
    ss.run_extremum(str(cfg_file), test=False)
