# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unified PMC data access layer.

Defines the ``PmcDataAccessor`` protocol and the ``PmcDataCache``
concrete implementation that wraps ``dict`` or ``DataFrame`` inputs
and caches repeated lookups at every nesting level.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PmcDataAccessor(Protocol):
    """
    Read-only interface for accessing raw PMC counter data.

    All metric-evaluation code should accept this protocol instead of
    ``Union[pd.DataFrame, dict, PmcDataCache]``.
    """

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        ...

    def __contains__(self, key: object) -> bool: ...

    def get(
        self,
        key: str,
        default: Any = None,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        ...

    def has_column(self, table_key: str, col_name: str) -> bool: ...


class PmcDataCache:
    """
    Caches lookups from raw PMC data at every nesting level.

    Wraps a ``raw_pmc_df`` (DataFrame with MultiIndex columns, plain
    dict, or single-level DataFrame) and caches ``__getitem__``
    results.  When a lookup returns a DataFrame, the result is itself
    wrapped in a new ``PmcDataCache`` so that subsequent column access
    (e.g. ``cache['pmc_perf']['SQ_WAVES']``) is also cached.  Series,
    scalars, and other results are stored directly.

    Attribute access (e.g. ``.columns``, ``hasattr(cache, 'COL')``)
    is delegated to the underlying data structure.
    """

    def __init__(self, raw_pmc_df: pd.DataFrame | dict) -> None:
        self._raw_pmc_df = raw_pmc_df
        self._cache: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        if key not in self._cache:
            value = self._raw_pmc_df[key]
            if isinstance(value, pd.DataFrame):
                value = PmcDataCache(value)
            self._cache[key] = value
        return self._cache[key]

    def get(
        self,
        key: str,
        default: Any = None,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Return cached value for *key*, or *default* on miss."""
        try:
            return self[key]
        except (KeyError, TypeError):
            return default

    def __contains__(self, key: object) -> bool:
        return key in self._raw_pmc_df

    def has_column(self, table_key: str, col_name: str) -> bool:
        """Check whether *table_key* exists and contains *col_name*."""
        if table_key not in self:
            return False
        nested = self.get(table_key)
        if nested is None:
            return False
        return hasattr(nested, col_name)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self._raw_pmc_df, name)
