from pyfoamclone.core.ghost_fields import allocate_state, interior_view, ghost_shapes, State


def test_allocate_state_and_interior():
    state = allocate_state(4, 3)  # nx, ny
    assert isinstance(state, State)
    shapes = ghost_shapes(state)
    assert shapes["u"] == (3+2, 4+2)
    interior = interior_view(state["u"])
    assert interior.shape == (3, 4)
    # ensure ghost isolation (writing interior not touching ghost)
    interior[:] = 5.0
    assert state["u"][0, :].sum() == 0.0  # top ghost row untouched
