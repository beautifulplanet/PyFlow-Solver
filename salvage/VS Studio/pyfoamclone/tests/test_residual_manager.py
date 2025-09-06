from pyfoamclone.residuals.manager import ResidualManager


def test_plateau_detection_simple():
    rm = ResidualManager()
    # First 50 values modest decrease, second 50 essentially flat / tiny increase -> plateau
    first = [1.0 - 0.01 * i for i in range(50)]  # 1.0 -> 0.51
    # Start second half near 0.52 to keep geometric means close
    second = [0.52 + 5e-5 * i for i in range(50)]  # 0.52 -> ~0.5225
    vals = first + second
    for v in vals:
        rm.track("u", v)
    assert rm.plateau("u", window=100) is True


def test_drop_orders():
    rm = ResidualManager()
    for v in [10.0, 1.0]:
        rm.track("u", v)
    assert rm.series["u"].drop_orders() == 1.0
