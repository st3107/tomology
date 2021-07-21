import typing
from pathlib import Path
from datetime import datetime
from configparser import ConfigParser

import fire
import event_model as em
from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky.callbacks.core import CallbackBase
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.callbacks.broker import LiveImage
import numpy as np


def myprint(*args, **kwargs) -> None:
    now = datetime.now()
    now_str = now.strftime("%d/%m/%Y %H:%M:%S")
    print("[{}]".format(now_str), *args, **kwargs)
    return


class ServerConfig:

    def __init__(self):
        self.parser = ConfigParser()
        self.parser.add_section("SERVER")
        section = self.parser["SERVER"]
        section["host"] = "localhost"
        section["port"] = "5567"
        section["prefix"] = "raw"
        section["verbose"] = "1"
        section["name"] = "unnamed"

    @property
    def address(self) -> typing.Tuple[str, int]:
        return self.parser.get("SERVER", "host"), self.parser.getint("SERVER", "port")

    @property
    def prefix(self) -> typing.ByteString:
        return self.parser.get("SERVER", "prefix").encode()

    @property
    def verbose(self) -> int:
        return self.parser.getint("SERVER", "verbose")

    @property
    def name(self) -> str:
        return self.parser.get("SERVER", "name")

    @name.setter
    def name(self, value: str) -> None:
        self.parser.set("SERVER", "name", value)
        return

    def read(self, cfg_file: str) -> None:
        path = Path(cfg_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError("No such a file '{}'.".format(cfg_file))
        self.parser.read(cfg_file)

    def write(self, cfg_file: str) -> None:
        path = Path(cfg_file)
        path = path.expanduser()
        if path.is_file():
            raise FileExistsError("File '{}' exists.".format(cfg_file))
        with path.open("w") as f:
            self.parser.write(f)


class ServerBase:

    def __init__(self, config: ServerConfig):
        self.config = config
        self.dispatcher = RemoteDispatcher(config.address, prefix=config.prefix)

    def print(self, *args, **kwargs) -> None:
        if self.config.verbose > 0:
            myprint(*args, **kwargs)

    def run(self) -> None:
        self.print("Start the {} server.".format(self.config.name))
        try:
            self.dispatcher.start()
        except KeyboardInterrupt:
            self.print("Terminate the {} server.".format(self.config.name))


class ExtremumConfig(ServerConfig):

    def __init__(self):
        super().__init__()
        self.start = None
        self.event = None
        self.name = "extremum"
        self.parser.add_section("EXTREMUM")
        section = self.parser["EXTREMUM"]
        section["data_key"] = "dexela_image"
        section["directory"] = "{start[sample_name]}_{start[uid][:6]}"
        section["file_prefix"] = "{start[sample_name]}_{event[seq_num]}"

    def copy_start(self, start) -> None:
        self.start = dict(start)
        self.start.setdefault("sample_name", "None")

    def set_event(self, event) -> None:
        self.event = event

    @property
    def directory(self) -> Path:
        return Path(self.parser.get("EXTREMUM", "directory").format(start=self.start))

    @property
    def file_prefix(self) -> str:
        return self.parser.get("EXTREMUM", "file_prefix").format(start=self.start, event=self.event)

    @property
    def min_path(self) -> Path:
        rendered = self.file_prefix.format(start=self.start, event=self.event)
        file_name = "{}_min".format(rendered)
        return self.directory.joinpath(file_name).with_suffix(".npy")

    @property
    def max_path(self) -> Path:
        rendered = self.file_prefix.format(start=self.start, event=self.event)
        file_name = "{}_max".format(rendered)
        return self.directory.joinpath(file_name).with_suffix(".npy")

    @property
    def data_key(self) -> str:
        return self.parser.get("EXTREMUM", "data_key")


class Extremum(CallbackBase):

    def __init__(self, config: ExtremumConfig):
        super().__init__()
        self.config: ExtremumConfig = config
        self.min: np.ndarray = None
        self.max: np.ndarray = None

    def print(self, *args, **kwargs) -> None:
        if self.config.verbose > 0:
            myprint(*args, **kwargs)
        return

    def start(self, doc):
        uid = doc.get("uid", "missing")
        self.print("Start processing (id: '{}').".format(uid))
        self.min = None
        self.max = None
        self.config.copy_start(doc)
        self.config.directory.mkdir(parents=True, exist_ok=True)
        return

    def event_page(self, doc):
        for event_doc in em.unpack_event_page(doc):
            self.event(event_doc)
        return

    def event(self, doc):
        self.config.set_event(doc)
        seq_num = doc.get("seq_num", 0)
        self.print("Process event {}.".format(seq_num))
        frames = doc.get("data", {}).get(self.config.data_key, None)
        frames_np = np.asanyarray(frames)
        image = np.mean(frames_np, axis=0)
        if image is None:
            date_key = self.config.data_key
            self.print("No {} data in the event {}.".format(date_key, seq_num))
            return
        self.min = np.fmin(self.min, image) if self.min is not None else image
        self.max = np.fmax(self.max, image) if self.max is not None else image
        np.save(self.config.min_path, self.min)
        np.save(self.config.max_path, self.max)
        return

    def stop(self, doc):
        uid = doc.get("run_start", "missing")
        self.print("Finish processing (id: '{}').".format(uid))
        return


class ExtremumServer(ServerBase):

    def __init__(self, config: ExtremumConfig):
        super().__init__(config)
        extremum = Extremum(config)
        self.dispatcher.subscribe(extremum)


class BestEffortConfig(ServerConfig):

    def __init__(self):
        super().__init__()
        self.name = "best_effort"
        section = "BEST EFFORT"
        self.parser.add_section(section)
        self.parser.set(section, "image_key", "dexela_image")

    @property
    def image_key(self):
        return self.parser.get("BEST EFFORT", "image_key")


class BestEffortServer(ServerBase):

    def __init__(self, config: BestEffortConfig):
        super().__init__(config)
        bec = BestEffortCallback()
        li = LiveImage(config.image_key)
        self.dispatcher.subscribe(bec)
        self.dispatcher.subscribe(li)


class Commands:
    """A collection of commands.
    """

    def run_extremum(self, cfg_file: str, test: bool = False) -> None:
        """Run extremum server.

        Parameters
        ----------
        cfg_file : str
            The configuration file. It is an .ini file.
        test : bool, optional
            If True, it is a pytest mode, by default False.
        """
        config = ExtremumConfig()
        config.read(cfg_file)
        server = ExtremumServer(config)
        if not test:
            server.run()
        return

    def run_best_effort(self, cfg_file: str, test: bool = False) -> None:
        """Run best effort server.

        Parameters
        ----------
        cfg_file : str
            The configuration file. It is an .ini file.
        test : bool, optional
            If True, it is a pytest mode, by default False.
        """
        config = BestEffortConfig()
        config.read(cfg_file)
        server = BestEffortServer(config)
        if not test:
            server.run()
        return

    def create_extremum_config(self, cfg_file: str) -> None:
        """Create the configuration file for extremum server.

        Parameters
        ----------
        cfg_file : str
            The configuration file. It is an .ini file.
        """
        config = ExtremumConfig()
        config.write(cfg_file)
        return

    def create_best_effort_config(self, cfg_file: str) -> None:
        """Create the configuration file for best efforts server.

        Parameters
        ----------
        cfg_file : str
            The configuration file. It is an .ini file.
        """
        config = BestEffortConfig()
        config.write(cfg_file)
        return


def run_cli():
    c = Commands()
    fire.Fire(
        {
            "run_extremum": c.run_extremum,
            "run_best_effort": c.run_best_effort,
            "create_extremum_config": c.create_extremum_config,
            "create_best_effort_config": c.create_best_effort_config
        }
    )


if __name__ == "__main__":
    run_cli()
