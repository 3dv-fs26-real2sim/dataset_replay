"""Shared viewport utility import helper.

Must be imported after SimulationApp is created.
"""

import importlib


def get_viewport_utility():
    """Import and return ``omni.kit.viewport.utility``, enabling the extension if needed.

    Returns ``None`` if the module is unavailable.
    """
    try:
        return importlib.import_module("omni.kit.viewport.utility")
    except ModuleNotFoundError:
        pass
    try:
        app = importlib.import_module("omni.kit.app")
        app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "omni.kit.viewport.utility", True,
        )
    except Exception:
        return None
    try:
        return importlib.import_module("omni.kit.viewport.utility")
    except ModuleNotFoundError:
        return None
