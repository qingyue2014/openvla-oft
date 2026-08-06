"""Pin the LIBERO namespace to an explicitly approved source checkout.

This module is loaded only when its directory is deliberately prepended to
``PYTHONPATH``.  Cosmos Policy's venv contains a regular top-level ``libero``
package, which otherwise wins over the namespace-package layout used by the
shared LIBERO checkout and silently changes the resolved native asset files.
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
import types
from pathlib import Path


source_root_value = os.environ.get("LIBERO_NATIVE_SOURCE_ROOT", "")
if source_root_value:
    source_root = Path(source_root_value).expanduser().resolve()
    namespace_path = source_root / "libero"
    expected_inner_init = namespace_path / "libero" / "__init__.py"
    if not expected_inner_init.is_file():
        raise RuntimeError(
            "Approved LIBERO source checkout is incomplete: "
            f"{expected_inner_init}"
        )

    package = types.ModuleType("libero")
    package.__file__ = None
    package.__package__ = "libero"
    package.__path__ = [str(namespace_path)]
    spec = importlib.machinery.ModuleSpec("libero", loader=None, is_package=True)
    spec.submodule_search_locations = package.__path__
    package.__spec__ = spec
    sys.modules["libero"] = package
