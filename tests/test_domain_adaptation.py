"""Tests for the unified domain_adaptation interface."""

import numpy as np
import pytest
from da4bci.methods import domain_adaptation


class TestUnifiedInterface:
    def test_dispatches_all_methods(self, src_tgt_large_42):
        src, tgt = src_tgt_large_42
        for m in ["tca", "sa", "mida", "rd", "coral", "gfk", "art", "pt", "ot"]:
            res = domain_adaptation(src, tgt, method=m)
            assert isinstance(res, dict), f"method={m}"
            assert "weighted_source_data" in res, f"method={m}"

    def test_m3d_requires_labels(self, src_tgt_large_42):
        src, tgt = src_tgt_large_42
        with pytest.raises(ValueError):
            domain_adaptation(src, tgt, method="m3d")

    def test_m3d_with_labels(self):
        rng = np.random.RandomState(42)
        src = rng.randn(80, 8)
        tgt = rng.randn(80, 8) + 1
        labels = rng.choice([1, 2], size=80)
        res = domain_adaptation(src, tgt, method="m3d",
                                control={"source_labels": labels,
                                         "l_iter": 2, "max_dim": 6})
        assert isinstance(res, dict)
        assert "weighted_source_data" in res

    def test_invalid_method(self, src_tgt_42):
        src, tgt = src_tgt_42
        with pytest.raises(ValueError):
            domain_adaptation(src, tgt, method="invalid_method")

    def test_default_is_sa(self, src_tgt_42):
        src, tgt = src_tgt_42
        res_default = domain_adaptation(src, tgt)
        res_sa = domain_adaptation(src, tgt, method="sa")
        np.testing.assert_array_equal(res_default["weighted_source_data"],
                                       res_sa["weighted_source_data"])

    def test_control_params_pass_through(self, src_tgt_42):
        src, tgt = src_tgt_42
        res_k3 = domain_adaptation(src, tgt, method="tca",
                                   control={"k": 3, "sigma": 1, "mu": 0.1})
        res_k5 = domain_adaptation(src, tgt, method="tca",
                                   control={"k": 5, "sigma": 1, "mu": 0.1})
        assert res_k3["weighted_source_data"].shape[1] == 3
        assert res_k5["weighted_source_data"].shape[1] == 5

    def test_ot_control(self, src_tgt_42):
        src, tgt = src_tgt_42
        res = domain_adaptation(src, tgt, method="ot",
                                control={"eps": 0.2, "maxit": 100})
        assert res["epsilon"] == 0.2

    def test_reference(self):
        rng = np.random.RandomState(2024)
        src = rng.randn(10, 5)
        tgt = rng.randn(10, 5) + 1
        for m in ["sa", "coral", "rd"]:
            res = domain_adaptation(src, tgt, method=m)
            assert np.all(np.isfinite(res["weighted_source_data"])), f"method={m}"
