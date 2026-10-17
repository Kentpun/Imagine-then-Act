from swm.data.libero_local import _as_float, _as_int


def test_as_int_float():
    assert _as_int(3.0) == 3
    assert _as_float(1.5) == 1.5
