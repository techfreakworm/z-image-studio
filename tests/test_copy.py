import copy as zis_copy  # name shadows the stdlib `copy` but the module is unrelated

REQUIRED_KEYS = {
    "prompt", "negative_prompt", "model", "lora", "lora_strength",
    "steps", "cfg", "width", "height", "seed",
    "controlnet_image", "controlnet_preprocessor", "controlnet_scale",
    "upscale_image", "refine_steps", "refine_denoise", "output",
}

def test_tooltips_has_all_required_keys():
    assert REQUIRED_KEYS <= set(zis_copy.TOOLTIPS)

def test_tooltips_values_are_non_empty_strings():
    for key, val in zis_copy.TOOLTIPS.items():
        assert isinstance(val, str) and val.strip(), f"{key} is empty or non-string"

def test_tooltips_values_are_short_enough_for_a_tooltip():
    for key, val in zis_copy.TOOLTIPS.items():
        assert len(val) <= 200, f"{key} is too long for a tooltip ({len(val)} chars)"
