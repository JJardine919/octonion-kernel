import numpy as np

from octonion_kernel.compress_controls import (
    load_and_split_digits, fit_pca, pca_encode, pca_decode, run_compress_control,
)


def test_load_and_split_digits_shapes():
    train_chunks, test_chunks = load_and_split_digits(seed=0, test_frac=0.2)
    assert train_chunks.shape[1] == 8
    assert test_chunks.shape[1] == 8
    assert train_chunks.shape[0] + test_chunks.shape[0] == 1797 * 8


def test_load_and_split_digits_deterministic():
    a1, b1 = load_and_split_digits(seed=5)
    a2, b2 = load_and_split_digits(seed=5)
    assert np.array_equal(a1, a2)
    assert np.array_equal(b1, b2)


def test_fit_pca_shapes_and_orthonormal_components():
    rng = np.random.default_rng(9)
    train = rng.standard_normal((200, 8))
    mean, components = fit_pca(train, k=3)
    assert mean.shape == (8,)
    assert components.shape == (3, 8)
    assert np.allclose(components @ components.T, np.eye(3), atol=1e-9)


def test_pca_roundtrip_at_k8_is_near_exact():
    rng = np.random.default_rng(10)
    train = rng.standard_normal((200, 8))
    mean, components = fit_pca(train, k=8)
    codes = pca_encode(train, mean, components)
    decoded = pca_decode(codes, mean, components)
    assert np.allclose(decoded, train, atol=1e-7)


def test_run_compress_control_returns_well_formed_result():
    result = run_compress_control(k=3, seed=0)
    assert set(result["mean_mse"].keys()) == {
        "octonion", "random_rotation", "raw_truncation", "pca"
    }
    for mse in result["mean_mse"].values():
        assert np.isfinite(mse) and mse >= 0.0
    for baseline in ("random_rotation", "raw_truncation"):
        v = result["verdict_by_baseline"][baseline]
        assert isinstance(v["octonion_wins"], bool)
        assert v["ci_lo"] <= v["ci_hi"]
    assert isinstance(result["octonion_beats_fixed_baselines"], bool)


def test_run_compress_control_deterministic():
    r1 = run_compress_control(k=3, seed=7)
    r2 = run_compress_control(k=3, seed=7)
    assert r1["mean_mse"] == r2["mean_mse"]


def test_run_compress_control_information_density_is_well_formed():
    result = run_compress_control(k=3, seed=0)
    density = result["information_density"]
    assert set(density.keys()) == {"octonion", "random_rotation", "raw_truncation", "pca"}
    for method_stats in density.values():
        assert np.isfinite(method_stats["reconstruction_fidelity"])
        assert np.isfinite(method_stats["max_h1"])
        assert method_stats["max_h1"] >= 0.0
        assert np.isfinite(method_stats["information_density"])
