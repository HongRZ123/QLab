"""Kalman spread 信号单元测试"""
import numpy as np
import pytest

from signals.kalman import compute_kalman_spread


def test_output_keys_and_shapes():
    """输出应包含 6 个键，所有数组形状正确"""
    x = np.random.randn(100)
    y = np.random.randn(100)
    result = compute_kalman_spread(x, y)

    expected_keys = {'beta_slope', 'beta_intercept', 'e', 'Q', 'sqrt_Q', 'spread'}
    assert set(result.keys()) == expected_keys

    for key in expected_keys:
        assert result[key].shape == (100,)


def test_beta_convergence():
    """y = 2x + 1 + noise 时，beta_slope 应收敛到 ~2.0"""
    np.random.seed(42)
    T = 500
    x = np.cumsum(np.random.randn(T)) + 10
    y = 2 * x + 1 + 0.1 * np.random.randn(T)

    result = compute_kalman_spread(x, y)

    # 检查尾部中位数
    tail = result['beta_slope'][-100:]
    median = np.median(tail)
    assert 1.8 <= median <= 2.2, f"beta_slope median {median} not in [1.8, 2.2]"


def test_q_positive():
    """Q 应全为正数"""
    x = np.random.randn(100)
    y = np.random.randn(100)
    result = compute_kalman_spread(x, y)

    assert np.all(result['Q'] > 0)


def test_numerical_equality_with_s9():
    """与 S9 内联计算结果应数值相等"""
    # 生成测试数据
    np.random.seed(42)
    T = 200
    x = np.cumsum(np.random.randn(T)) + 10
    y = 2 * x + 1 + 0.1 * np.random.randn(T)

    # 使用新函数
    result_new = compute_kalman_spread(x, y, delta=0.0001, ve=0.001)

    # 复制 S9 的内联计算（第 90-148 行）
    x_aug = np.column_stack([x, np.ones(T)])
    beta = np.zeros(2)
    R = np.zeros((2, 2))
    Vw = 0.0001 / (1.0 - 0.0001) * np.eye(2)
    Ve = 0.001

    beta_slope_old = np.zeros(T)
    beta_intercept_old = np.zeros(T)
    e_arr_old = np.zeros(T)
    Q_arr_old = np.zeros(T)

    for t in range(T):
        if t > 0:
            R = R + Vw
        x_t = x_aug[t]
        yhat = np.dot(x_t, beta)
        Q_t = np.dot(x_t, np.dot(R, x_t)) + Ve
        Q_arr_old[t] = Q_t
        e_t = y[t] - yhat
        e_arr_old[t] = e_t
        K = np.dot(R, x_t) / Q_t
        beta = beta + K * e_t
        R = R - np.outer(K, np.dot(x_t, R))
        beta_slope_old[t] = beta[0]
        beta_intercept_old[t] = beta[1]

    sqrt_Q_old = np.sqrt(Q_arr_old)
    spread_old = y - beta_slope_old * x

    # 比较
    assert np.allclose(result_new['beta_slope'], beta_slope_old)
    assert np.allclose(result_new['beta_intercept'], beta_intercept_old)
    assert np.allclose(result_new['e'], e_arr_old)
    assert np.allclose(result_new['Q'], Q_arr_old)
    assert np.allclose(result_new['sqrt_Q'], sqrt_Q_old)
    assert np.allclose(result_new['spread'], spread_old)


def test_mismatched_lengths():
    """x 和 y 长度不等应抛出 ValueError"""
    x = np.random.randn(100)
    y = np.random.randn(50)

    with pytest.raises(ValueError, match="长度必须相等"):
        compute_kalman_spread(x, y)
