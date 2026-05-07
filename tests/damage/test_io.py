import json

import pytest

from bvidfe.damage.io import (
    CScanSchemaError,
    damage_state_from_dict,
    damage_state_to_dict,
    load_cscan_json,
    save_cscan_json,
)
from bvidfe.damage.state import DamageState, DelaminationEllipse


def _make_state() -> DamageState:
    return DamageState(
        [
            DelaminationEllipse(3, (75, 50), 28, 18, 45),
            DelaminationEllipse(4, (78, 52), 32, 20, 50),
        ],
        dent_depth_mm=0.45,
        fiber_break_radius_mm=3.0,
    )


def test_round_trip_to_dict_and_back():
    ds = _make_state()
    ds2 = damage_state_from_dict(damage_state_to_dict(ds))
    assert ds2.dent_depth_mm == ds.dent_depth_mm
    assert ds2.fiber_break_radius_mm == ds.fiber_break_radius_mm
    assert len(ds2.delaminations) == 2
    assert ds2.delaminations[0].interface_index == 3
    assert ds2.delaminations[1].major_mm == 32


def test_round_trip_file(tmp_path):
    ds = _make_state()
    fp = tmp_path / "cscan.json"
    save_cscan_json(ds, fp)
    ds2 = load_cscan_json(fp)
    assert ds2.delaminations[1].orientation_deg == 50


def test_rejects_bad_schema_version(tmp_path):
    fp = tmp_path / "bad.json"
    fp.write_text(json.dumps({"schema_version": "99.0", "delaminations": [], "dent_depth_mm": 0}))
    with pytest.raises(CScanSchemaError):
        load_cscan_json(fp)


def test_rejects_missing_schema_version(tmp_path):
    fp = tmp_path / "bad.json"
    fp.write_text(json.dumps({"delaminations": [], "dent_depth_mm": 0}))
    with pytest.raises(CScanSchemaError):
        load_cscan_json(fp)


def test_rejects_negative_ellipse(tmp_path):
    fp = tmp_path / "bad.json"
    fp.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dent_depth_mm": 0.0,
                "delaminations": [
                    {
                        "interface_index": 0,
                        "centroid_mm": [0, 0],
                        "major_mm": -1,
                        "minor_mm": 5,
                        "orientation_deg": 0,
                    }
                ],
            }
        )
    )
    with pytest.raises(CScanSchemaError):
        load_cscan_json(fp)


def test_rejects_negative_dent(tmp_path):
    fp = tmp_path / "bad.json"
    fp.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dent_depth_mm": -0.1,
                "delaminations": [],
            }
        )
    )
    with pytest.raises(CScanSchemaError):
        load_cscan_json(fp)


def _valid_payload(**overrides):
    base = {
        "schema_version": "1.0",
        "dent_depth_mm": 0.4,
        "fiber_break_radius_mm": 1.5,
        "delaminations": [
            {
                "interface_index": 1,
                "centroid_mm": [10.0, 5.0],
                "major_mm": 8.0,
                "minor_mm": 4.0,
                "orientation_deg": 0.0,
            }
        ],
    }
    base.update(overrides)
    return base


def test_rejects_string_in_numeric_field(tmp_path):
    """Issue #10: a string passed in dent_depth_mm used to crash the
    < 0 comparison with a TypeError; it now raises CScanSchemaError."""
    fp = tmp_path / "bad.json"
    fp.write_text(json.dumps(_valid_payload(dent_depth_mm="abc")))
    with pytest.raises(CScanSchemaError, match="dent_depth_mm"):
        load_cscan_json(fp)


def test_rejects_nan_in_numeric_field(tmp_path):
    """NaN is JSON-encodable via Python's allow_nan=True default; the
    loader must reject it explicitly so it doesn't reach numpy."""
    fp = tmp_path / "bad.json"
    # json.dumps with default allow_nan=True emits 'NaN' as a literal
    fp.write_text(json.dumps(_valid_payload(dent_depth_mm=float("nan"))))
    with pytest.raises(CScanSchemaError, match="finite"):
        load_cscan_json(fp)


def test_rejects_inf_in_centroid(tmp_path):
    """Inf must be rejected for any numeric field, including centroid coords."""
    payload = _valid_payload()
    payload["delaminations"][0]["centroid_mm"] = [float("inf"), 5.0]
    fp = tmp_path / "bad.json"
    fp.write_text(json.dumps(payload))
    with pytest.raises(CScanSchemaError, match="finite"):
        load_cscan_json(fp)


def test_rejects_negative_fiber_break_radius(tmp_path):
    fp = tmp_path / "bad.json"
    fp.write_text(json.dumps(_valid_payload(fiber_break_radius_mm=-1.5)))
    with pytest.raises(CScanSchemaError, match="fiber_break_radius_mm"):
        load_cscan_json(fp)


def test_rejects_delamination_that_is_not_an_object(tmp_path):
    """A bare list or string in the delaminations array used to crash the
    'k not in d' check; now raises a clear CScanSchemaError."""
    payload = _valid_payload()
    payload["delaminations"].append("oops")
    fp = tmp_path / "bad.json"
    fp.write_text(json.dumps(payload))
    with pytest.raises(CScanSchemaError, match="must be an object"):
        load_cscan_json(fp)


def test_unknown_top_level_field_warns_but_loads(tmp_path):
    """Forward-compat: an unknown top-level field (e.g. a future
    'panel_id' annotation) emits a UserWarning but does not abort loading."""
    import warnings as _warnings

    payload = _valid_payload(panel_id="ABC-1234")
    fp = tmp_path / "ok.json"
    fp.write_text(json.dumps(payload))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        ds = load_cscan_json(fp)
    assert ds.dent_depth_mm == 0.4
    assert any("unknown top-level field" in str(w.message) for w in caught)


def test_unknown_delamination_field_warns_but_loads(tmp_path):
    import warnings as _warnings

    payload = _valid_payload()
    payload["delaminations"][0]["confidence"] = 0.95
    fp = tmp_path / "ok.json"
    fp.write_text(json.dumps(payload))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        ds = load_cscan_json(fp)
    assert len(ds.delaminations) == 1
    assert any("unknown field" in str(w.message) for w in caught)


def test_rejects_non_int_interface_index(tmp_path):
    """interface_index must be an int, not a float — silent int(2.5) → 2
    truncation hides a real schema error."""
    payload = _valid_payload()
    payload["delaminations"][0]["interface_index"] = 2.5
    fp = tmp_path / "bad.json"
    fp.write_text(json.dumps(payload))
    with pytest.raises(CScanSchemaError, match="must be an int"):
        load_cscan_json(fp)
