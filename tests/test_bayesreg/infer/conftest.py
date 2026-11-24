"""Registers the `slow` marker of the infer tests (review M4 item 8): the S = 1e5 whitening gate and the
real-certificate test (numba compilation).  They run by default; a daily run may deselect them with
-m "not slow"."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long test (whitening gate S = 1e5, numba certificate compilation)")
