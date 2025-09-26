from pathlib import Path


def test_import_alias_and_symbols() -> None:
    import Marketing_analytics as canonical
    import marketing_analytics as lowercase

    assert canonical is lowercase

    module_path = Path(canonical.__file__).resolve()
    assert module_path.name == "__init__.py"
    assert module_path.parent.name == "Marketing_analytics"

    from marketing_analytics import generate_verified_brief

    assert callable(generate_verified_brief)
