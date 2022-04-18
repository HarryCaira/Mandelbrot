import pytest
from mandelbrot import calc_zeta, calc_zeta_n


@pytest.mark.parametrize(
    "zeta_n_minus_one, c, d, want",
    [
        (0 + 0j, 1 + 1j, 2, 1 + 1j),  # happy simple
        (51 - 25j, 51 - 25j, 2, 2027 - 2575j),  # happy negative j
        (
            -12 + 4j,
            -12 + 4j,
            20,
            1.0868634934025113e22 - 1.6629361480080462e21j,
        ),  # happy large n
    ],
)
def test_calc_zeta_pass(zeta_n_minus_one: complex, c: complex, d: int, want: complex):
    assert calc_zeta(zeta_n_minus_one, c, d) == want


@pytest.mark.parametrize(
    "zeta_n_minus_one, c, d",
    [(-12 + 4j, -12 + 4j, 20000)],  # overflows
)
def test_calc_zeta_fail(zeta_n_minus_one: complex, c: complex, d: int):
    with pytest.raises(OverflowError):
        calc_zeta(zeta_n_minus_one, c, d)


@pytest.mark.parametrize(
    "c, d, n, divergence_limit, want",
    [
        (1 + 1j, 2, 4, 2, (1 + 3j, 2)),  # divergent at iter 2
        (
            0.2 + 0.2j,
            2,
            4,
            2,
            (0.12877055999999998 + 0.30083840000000006j, 4),
        ),  # convergent
    ],
)
def test_calc_zeta_n_pass(
    c: complex, d: int, n: int, divergence_limit: int, want: tuple[complex, int]
):
    assert calc_zeta_n(c, d, n, divergence_limit) == want
