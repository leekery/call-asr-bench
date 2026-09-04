from importlib.metadata import entry_points


def test_callasr_console_entry_point_is_installed() -> None:
    matches = entry_points(group="console_scripts", name="callasr")
    assert len(matches) == 1
    entry_point = next(iter(matches))
    assert entry_point.value == "callasr.cli:main"
