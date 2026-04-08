# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import numpy as np
import pandas as pd
import pytest

from utils.metrics import MetricEvaluator, PmcDataCache


def _make_pmc_dict():
    return {
        "pmc_perf": pd.DataFrame({
            "SQ_WAVES": [100, 200, 150],
            "GRBM_GUI_ACTIVE": [1000, 2000, 1500],
        })
    }


def _make_pmc_multiindex_df():
    sub_df = pd.DataFrame({
        "SQ_WAVES": [100, 200, 150],
        "GRBM_GUI_ACTIVE": [1000, 2000, 1500],
    })
    return pd.concat({"pmc_perf": sub_df}, axis=1)


def test_pmc_data_cache_dict_input():
    cache = PmcDataCache(_make_pmc_dict())

    nested = cache["pmc_perf"]
    assert isinstance(nested, PmcDataCache)

    series = nested["SQ_WAVES"]
    assert isinstance(series, pd.Series)
    assert list(series) == [100, 200, 150]


def test_pmc_data_cache_dataframe_input():
    cache = PmcDataCache(_make_pmc_multiindex_df())

    nested = cache["pmc_perf"]
    assert isinstance(nested, PmcDataCache)

    series = nested["SQ_WAVES"]
    assert isinstance(series, pd.Series)
    assert list(series) == [100, 200, 150]


def test_pmc_data_cache_level1_cache_hit():
    cache = PmcDataCache(_make_pmc_dict())

    first = cache["pmc_perf"]
    second = cache["pmc_perf"]
    assert first is second


def test_pmc_data_cache_level2_cache_hit():
    cache = PmcDataCache(_make_pmc_dict())

    nested = cache["pmc_perf"]
    first = nested["SQ_WAVES"]
    second = nested["SQ_WAVES"]
    assert first is second


def test_pmc_data_cache_getattr_delegation():
    cache = PmcDataCache(_make_pmc_dict())
    nested = cache["pmc_perf"]

    assert hasattr(nested, "SQ_WAVES")
    assert list(nested.columns) == ["SQ_WAVES", "GRBM_GUI_ACTIVE"]
    assert not hasattr(nested, "NONEXISTENT")


def test_pmc_data_cache_contains_and_get():
    cache = PmcDataCache(_make_pmc_dict())

    assert "pmc_perf" in cache
    assert "nonexistent" not in cache

    assert isinstance(cache.get("pmc_perf"), PmcDataCache)
    assert cache.get("nonexistent") is None
    assert cache.get("nonexistent", "fallback") == "fallback"


def test_pmc_data_cache_has_column_present():
    cache = PmcDataCache(_make_pmc_dict())
    assert cache.has_column("pmc_perf", "SQ_WAVES") is True


def test_pmc_data_cache_has_column_missing_table():
    cache = PmcDataCache(_make_pmc_dict())
    assert cache.has_column("nonexistent", "SQ_WAVES") is False


def test_pmc_data_cache_has_column_missing_column():
    cache = PmcDataCache(_make_pmc_dict())
    assert cache.has_column("pmc_perf", "NONEXISTENT") is False


def test_pmc_data_cache_getitem_missing_key():
    cache = PmcDataCache(_make_pmc_dict())
    with pytest.raises(KeyError):
        cache["nonexistent"]


def test_pmc_data_cache_get_type_error():
    cache = PmcDataCache(_make_pmc_dict())
    nested = cache["pmc_perf"]
    assert nested.get([1, 2]) is None
    assert nested.get([1, 2], "fallback") == "fallback"


def test_pmc_data_cache_scalar_value_not_wrapped():
    raw = {"version": 42, "pmc_perf": pd.DataFrame({"A": [1]})}
    cache = PmcDataCache(raw)

    assert cache["version"] == 42
    assert not isinstance(cache["version"], PmcDataCache)


def test_pmc_data_cache_with_metric_evaluator():
    cache = PmcDataCache(_make_pmc_dict())

    evaluator = MetricEvaluator(cache, {}, {})
    result = evaluator.eval_expression("to_avg(raw_pmc_df['pmc_perf']['SQ_WAVES'])")

    assert result != "N/A"
    assert np.isclose(result, 150.0)
