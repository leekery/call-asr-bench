from pathlib import Path

path = Path("tests/test_concurrent_cli.py")
text = path.read_text(encoding="utf-8")
old = '''    [
        ("explicit", "callasr", "openai", "explicit"),
        (None, "callasr", "openai", "callasr"),
        (None, None, "openai", "openai"),
        (None, None, None, None),
    ],'''
new = '''    [
        (
            "secret-explicit-41f7",
            "secret-callasr-62bd",
            "secret-provider-93ae",
            "secret-explicit-41f7",
        ),
        (None, "secret-callasr-62bd", "secret-provider-93ae", "secret-callasr-62bd"),
        (None, None, "secret-provider-93ae", "secret-provider-93ae"),
        (None, None, None, None),
    ],'''
if text.count(old) != 1:
    raise SystemExit("secret parameter block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
