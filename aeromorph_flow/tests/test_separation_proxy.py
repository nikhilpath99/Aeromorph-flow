import numpy as np

from aeromorph_flow.src.evaluation.separation_proxy import cp_separation_proxy, pearson_corr


def test_cp_separation_proxy_shapes():
    x = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    upper = -1.0 + x
    lower = 0.5 * np.ones_like(x)
    cp = np.concatenate([upper, lower])[None, :]
    proxy = cp_separation_proxy(cp)
    assert proxy["cp_min_upper"].shape == (1,)
    assert proxy["cp_recovery_gradient_upper"][0] > 0
    assert proxy["max_positive_dcpdx_upper_after_min"][0] > 0


def test_pearson_corr_handles_constant_inputs():
    assert np.isnan(pearson_corr(np.ones(3), np.arange(3)))
