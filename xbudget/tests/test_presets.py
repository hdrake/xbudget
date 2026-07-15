import warnings
from pathlib import Path

import pytest
import yaml

from xbudget.presets import load_yaml, save_yaml
from xbudget.parse import BudgetParseError

CONVENTIONS = sorted((Path(__file__).parent.parent / "conventions").glob("*.yaml"))

def test_load_all_convention_presets():
    """Test that all convention YAML files can be loaded successfully."""
    conventions_dir = Path(__file__).parent.parent / "conventions"
    yaml_files = sorted(conventions_dir.glob("*.yaml"))

    assert len(yaml_files) > 0, "No YAML files found in conventions directory"

    for yaml_file in yaml_files:
        preset_dict = load_yaml(str(yaml_file))
        assert isinstance(preset_dict, dict), f"Failed to load {yaml_file.name} as dictionary"
        assert len(preset_dict) > 0, f"Loaded preset {yaml_file.name} is empty"


def test_load_yaml_raises_on_invalid_yaml(tmp_path):
    """A broken file must surface the YAML error, not an opaque NameError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("heat: {rhs: [unclosed\n")
    with pytest.raises(yaml.YAMLError):
        load_yaml(str(bad))


@pytest.mark.parametrize("path", CONVENTIONS, ids=lambda p: p.name)
def test_save_yaml_round_trips_shipped_conventions(path, tmp_path):
    """load -> save -> load returns the same recipe, key order included."""
    original = load_yaml(str(path))
    out = tmp_path / path.name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tolerated placeholder terms
        save_yaml(original, str(out))
    reloaded = load_yaml(str(out))
    assert reloaded == original
    # Key order is meaningful (it drives operand order), so pin it too.
    assert list(reloaded) == list(original)
    for budget in original:
        assert list(reloaded[budget]) == list(original[budget])


def test_save_yaml_does_not_mutate_the_recipe(tmp_path):
    import copy

    d = {"heat": {"rhs": {"var": None, "sum": {"var": None, "f": {"var": "diag"}}}}}
    before = copy.deepcopy(d)
    save_yaml(d, str(tmp_path / "out.yaml"))
    assert d == before


def test_save_yaml_rejects_malformed_and_writes_nothing(tmp_path):
    """Validation is the point: a bad recipe must not reach disk."""
    out = tmp_path / "never.yaml"
    with pytest.raises(BudgetParseError):
        save_yaml({"heat": "not a dict"}, str(out))
    assert not out.exists()


def test_save_yaml_preserves_operand_order(tmp_path):
    """safe_dump's default sort_keys=True would reorder these alphabetically."""
    d = {
        "heat": {
            "rhs": {
                "var": None,
                "sum": {
                    "var": None,
                    "zonal": {"var": "z_diag"},
                    "advection": {"var": "a_diag"},
                    "meridional": {"var": "m_diag"},
                },
            }
        }
    }
    out = tmp_path / "order.yaml"
    save_yaml(d, str(out))
    reloaded = load_yaml(str(out))
    assert list(reloaded["heat"]["rhs"]["sum"]) == [
        "var",
        "zonal",
        "advection",
        "meridional",
    ]
