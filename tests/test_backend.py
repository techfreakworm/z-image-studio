import backend


def test_duration_t2i_turbo_is_short():
    d = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=8, width=1024, height=1024))
    assert 60 <= d <= 90


def test_duration_t2i_base_is_longer():
    d = backend.duration_for(mode="t2i", params=dict(model="Base", steps=25, width=1024, height=1024))
    assert d > 60


def test_duration_clamps_at_180():
    d = backend.duration_for(mode="t2i", params=dict(model="Base", steps=200, width=2048, height=2048))
    assert d == 180


def test_duration_clamps_at_60():
    d = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=1, width=256, height=256))
    assert d == 60


def test_duration_multiplier_scales_up():
    base = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=8, width=1024, height=1024))
    retry = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=8, width=1024, height=1024),
                                  multiplier=2.0)
    assert retry > base


def test_duration_upscale_has_realesrgan_overhead():
    t2i = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=8, width=1024, height=1024))
    upsc = backend.duration_for(mode="upscale", params=dict(refine_steps=5, width=1024, height=1024))
    assert upsc > t2i
