import contextlib

from .amber_utils import assign_chainids

with contextlib.suppress(ImportError):
    from .parsl_settings import (
        AuroraSettings,
        LocalSettings,
        PolarisSettings,
    )
