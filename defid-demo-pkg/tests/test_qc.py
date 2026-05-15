from defid_demo.qc import check_rep


def _swipe(n):
    return [{"t": i * 0.02, "x": float(i), "y": float(i)} for i in range(n)]


def _keys(n):
    out = []
    for i in range(n):
        out.append({"t": i * 0.2, "phase": "down"})
        out.append({"t": i * 0.2 + 0.08, "phase": "up"})
    return out


def test_good_rep_passes():
    r = check_rep(_swipe(20), _keys(6))
    assert r.ok and r.reason is None


def test_too_few_touch_points_is_a_tap_not_a_swipe():
    r = check_rep(_swipe(2), _keys(6))
    assert not r.ok and "touch" in r.reason


def test_no_typing_rejected():
    r = check_rep(_swipe(20), [])
    assert not r.ok and "key" in r.reason


def test_non_monotonic_touch_rejected():
    bad = _swipe(10)
    bad[5]["t"] = bad[4]["t"] - 0.01
    r = check_rep(bad, _keys(6))
    assert not r.ok and "monotonic" in r.reason
