"""CPU-only checks of the unrescaled two-task CAGrad contract."""

import math
import random

import pytest


pytestmark = pytest.mark.unit


def test_zero_c_returns_the_mean_gradient():
    from gradientwam.cagrad import cagrad_coefficients

    assert cagrad_coefficients([[4.0, -1.0], [-1.0, 9.0]], c=0.0) == (0.5, 0.5)


def _solve(gram, c=0.4):
    from gradientwam.cagrad import cagrad_coefficients

    return cagrad_coefficients(gram, c)


def _dot(left, right):
    return math.fsum(x * y for x, y in zip(left, right))


def _gram(video, action):
    cross = _dot(video, action)
    return [[_dot(video, video), cross], [cross, _dot(action, action)]]


def _compose(video, action, coefficients):
    return tuple(coefficients[0] * v + coefficients[1] * a for v, a in zip(video, action))


def _primal_reference(video, action, c):
    """Solve the 2D primal by circle extrema/intersections, not the dual solver."""
    mean = tuple((v + a) / 2 for v, a in zip(video, action))
    radius = c * math.hypot(*mean)
    candidates = [mean]
    for gradient in (video, action):
        norm = math.hypot(*gradient)
        if norm:
            candidates.append(tuple(m + radius * g / norm for m, g in zip(mean, gradient)))
    difference = tuple(v - a for v, a in zip(video, action))
    norm = math.hypot(*difference)
    if radius and norm:
        axis = tuple(g / norm for g in difference)
        projection = -_dot(difference, mean) / (radius * norm)
        if abs(projection) <= 1:
            perpendicular = (-axis[1], axis[0])
            offset = math.sqrt(max(0.0, 1 - projection * projection))
            for sign in (-1, 1):
                candidates.append(tuple(
                    m + radius * (projection * u + sign * offset * p)
                    for m, u, p in zip(mean, axis, perpendicular)
                ))
    return max(candidates, key=lambda d: min(_dot(video, d), _dot(action, d)))


@pytest.mark.parametrize("video,action,expected", [
    ((1.0, 0.0), (1.0, 0.0), (1.4, 0.0)),
    ((2.0, 0.0), (1.0, 0.0), (2.1, 0.0)),
    ((2.0, 0.0), (-1.0, 0.0), (0.3, 0.0)),
    ((1.0, 0.0), (-1.0, 0.0), (0.0, 0.0)),
    ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    ((0.0, 0.0), (2.0, 0.0), (1.0, 0.0)),
    ((2.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
    ((1.0, 0.0), (0.0, 1.0), (0.7, 0.7)),
])
def test_hand_derived_unrescaled_and_degenerate_directions(video, action, expected):
    coefficients = _solve(_gram(video, action))
    assert all(isinstance(value, float) and math.isfinite(value) for value in coefficients)
    assert _compose(video, action, coefficients) == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize("c", [0.0, 0.01, 0.4, 0.95, 0.999999])
def test_random_gram_matches_independent_primal_optimum(c):
    rng = random.Random(20261009)
    for _ in range(128):
        video = (rng.uniform(-4, 4), rng.uniform(-4, 4))
        action = (rng.uniform(-4, 4), rng.uniform(-4, 4))
        actual = _compose(video, action, _solve(_gram(video, action), c))
        expected = _primal_reference(video, action, c)
        scale = max(math.hypot(*video), math.hypot(*action), 1.0)
        mean = tuple((v + a) / 2 for v, a in zip(video, action))
        assert math.dist(actual, mean) <= c * math.hypot(*mean) + 2e-10 * scale
        actual_score = min(_dot(video, actual), _dot(action, actual))
        expected_score = min(_dot(video, expected), _dot(action, expected))
        assert actual_score == pytest.approx(expected_score, rel=2e-10, abs=2e-10 * scale**2)


def test_task_swap_is_symmetric_and_action_is_not_protected():
    video, action = (2.0, 0.0), (-1.0, 0.0)
    coefficients = _solve(_gram(video, action))
    swapped = _solve(_gram(action, video))
    assert coefficients == pytest.approx(swapped[::-1])
    assert _dot(action, _compose(video, action, coefficients)) < 0


@pytest.mark.parametrize("factor", [1e-280, 1e-100, 1e100, 1e280])
def test_common_gradient_scale_does_not_change_coefficients(factor):
    gram = [[4.0, 1.0], [1.0, 2.0]]
    expected = _solve(gram)
    assert _solve([[factor * value for value in row] for row in gram]) == pytest.approx(expected)


def test_large_finite_gram_does_not_overflow_in_the_mean():
    assert _solve([[1e308, 0.0], [0.0, 1e308]]) == pytest.approx((0.7, 0.7))


def test_nearly_opposed_gradients_remain_feasible():
    video, action = (1.0, 0.0), (-1.0, 1e-5)
    actual = _compose(video, action, _solve(_gram(video, action)))
    expected = _primal_reference(video, action, 0.4)
    assert actual == pytest.approx(expected, abs=2e-11)


@pytest.mark.parametrize("gram", [
    None, [], [1.0, 2.0], [[1.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    [[1.0, 0.2], [0.5, 1.0]], [[-1.0, 0.0], [0.0, 1.0]],
    [[1.0, 2.0], [2.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]],
    [[1.0, "bad"], ["bad", 1.0]], [[1.0, complex(0, 1)], [complex(0, 1), 1.0]],
    [[math.nan, 0.0], [0.0, 1.0]], [[1.0, math.inf], [math.inf, 1.0]],
])
def test_invalid_gram_is_rejected_even_when_c_is_zero(gram):
    for c in (0.0, 0.4):
        with pytest.raises(ValueError):
            _solve(gram, c)


@pytest.mark.parametrize("c", [-0.01, 1.0, 2.0, math.nan, math.inf, "bad", complex(0, 1)])
def test_invalid_c_is_rejected(c):
    with pytest.raises(ValueError):
        _solve([[1.0, 0.0], [0.0, 1.0]], c)
