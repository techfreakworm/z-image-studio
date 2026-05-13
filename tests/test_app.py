import app


def test_on_model_change_returns_base_defaults():
    assert app._on_model_change("Base") == (25, 4.0)


def test_on_model_change_returns_turbo_defaults():
    assert app._on_model_change("Turbo") == (8, 1.0)


def test_on_model_change_unknown_falls_back_to_turbo():
    assert app._on_model_change("Edit") == (8, 1.0)
