import importlib.util
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make top-level modules importable in tests
sys.path.insert(0, str(_PROJECT_ROOT))

# ── copy.py shadow fix ────────────────────────────────────────────────────────
# The project ships a file named ``copy.py`` which shadows the stdlib module.
# Third-party packages (pydantic, gradio) use ``from copy import deepcopy``;
# they must see the stdlib module.  test_copy.py uses ``import copy as zis_copy``
# and must see the project module.
#
# Fix: eagerly import all third-party packages that use stdlib copy so their
# module-level ``from copy import ...`` calls are resolved against stdlib NOW,
# before we swap sys.modules["copy"].  After the swap only new ``import copy``
# statements are affected — and test_copy.py is the only file that issues one.
try:
    import torch     # noqa: F401  — torch._tensor uses `from copy import deepcopy`
    import pydantic  # noqa: F401  — resolves `from copy import deepcopy` in pydantic.main
    import fastapi   # noqa: F401  — depends on pydantic
    import gradio    # noqa: F401  — depends on fastapi/pydantic
except ImportError:
    pass  # optional deps; if absent the swap is harmless

# Now replace sys.modules["copy"] with our project module so that any
# subsequent ``import copy`` (i.e., in test_copy.py) gets TOOLTIPS.
_project_copy_path = _PROJECT_ROOT / "copy.py"
_spec = importlib.util.spec_from_file_location("copy", _project_copy_path)
_project_copy_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_project_copy_module)
sys.modules["copy"] = _project_copy_module
