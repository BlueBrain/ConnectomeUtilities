import tempfile
import shutil

from git import Repo
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def setup_tempdir(cleanup=True):
    temp_dir = str(Path(tempfile.mkdtemp()).resolve())
    try:
        yield temp_dir
    finally:
        if cleanup:
            shutil.rmtree(temp_dir)

@contextmanager
def get_snap_test_circuit():
    with setup_tempdir() as tmp_dir:
        repo = Repo.init(tmp_dir)
        snap_origin = repo.create_remote("origin", "https://github.com/BlueBrain/snap.git")
        snap_origin.pull("release_v0.13.2")

        circ_path = Path(tmp_dir) / "tests" / "data" / "circuit_config.json"
        yield circ_path

