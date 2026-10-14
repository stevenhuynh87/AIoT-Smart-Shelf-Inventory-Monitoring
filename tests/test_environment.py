"""Verification tests for required Python environment dependencies."""

import pytest


def test_numpy_installed():
    import numpy as np
    assert np.__version__ is not None
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6


def test_pandas_installed():
    import pandas as pd
    assert pd.__version__ is not None
    df = pd.DataFrame({"item": ["product_a", "product_b"], "qty": [10, 5]})
    assert len(df) == 2


def test_opencv_installed():
    import cv2
    assert cv2.__version__ is not None


def test_paho_mqtt_installed():
    import paho.mqtt.client as mqtt
    assert mqtt is not None


def test_ultralytics_installed():
    import ultralytics
    assert ultralytics.__version__ is not None


def test_project_packages_importable():
    import vision
    import fusion
    assert vision.__version__ == "0.1.0"
    assert fusion.__version__ == "0.1.0"
