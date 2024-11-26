# tests/test_utils.py
import os
import sys

current_file_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.join(current_file_dir, "..")
sys.path.insert(0, root_dir)


def test_version():
    # import slo_lr_detection module
    import src.slo_lr_detection as pkg

    # test version
    assert pkg.__version__ == "1.0.0"
