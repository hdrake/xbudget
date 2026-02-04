import pytest
import os
from pathlib import Path
from xbudget.presets import load_yaml

def test_load_all_convention_presets():
    """Test that all convention YAML files can be loaded successfully."""
    conventions_dir = Path(__file__).parent.parent / "conventions"
    yaml_files = sorted(conventions_dir.glob("*.yaml"))
    
    assert len(yaml_files) > 0, "No YAML files found in conventions directory"
    
    for yaml_file in yaml_files:
        preset_dict = load_yaml(str(yaml_file))
        assert isinstance(preset_dict, dict), f"Failed to load {yaml_file.name} as dictionary"
        assert len(preset_dict) > 0, f"Loaded preset {yaml_file.name} is empty"
