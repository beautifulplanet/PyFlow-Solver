from pyfoamclone.mesh.mesh_builder import GridSpec, build_grid
from pyfoamclone.core.ghost_fields import allocate_state, interior_view, State
from pyfoamclone.boundary_conditions.apply import apply_all
from pyfoamclone.boundary_conditions import registry


def test_build_grid():
    spec = GridSpec(4, 5, 2.0, 1.0)
    g = build_grid(spec)
    assert g["dx"] == 0.5 and g["dy"] == 0.2
    assert g["Xc"].shape == (4, 5)


def test_bc_registry_and_apply():
    state = allocate_state(4, 4)
    assert isinstance(state, State)
    apply_all({"u": "wall", "v": "moving_wall"}, state)
    assert interior_view(state["u"]).sum() == 0.0  # wall sets zeros
    assert interior_view(state["v"]).sum() > 0.0  # moving wall set to 1
    # ensure registry exposes expected keys
    avail = registry.available()
    assert "wall" in avail and "moving_wall" in avail
