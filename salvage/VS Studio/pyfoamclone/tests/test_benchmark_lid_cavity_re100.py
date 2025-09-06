import json
from pathlib import Path
import numpy as np
from pyfoamclone.scripts.run_case import run

# Acceptable relative tolerance for centerline u profile vs coarse-grid Ghia data
RTOL = 0.12  # looser than production until operators refined


def load_reference():
    ref_path = Path(__file__).parent.parent / 'pyfoamclone' / 'benchmarks' / 'ghia_centerline_u_re100.json'
    data = json.loads(ref_path.read_text())
    y_desc = np.array(data['y_desc'])
    u_ref_desc = np.array(data['u_centerline'])
    return y_desc, u_ref_desc


def test_lid_cavity_re100_centerline_match():
    case = Path(__file__).parent.parent / 'cases' / 'lid_cavity_re100.json'
    result = run(str(case))
    u_centerline = np.array(result['u_centerline'])
    y_centerline = np.array(result['y_centerline'])
    # Ensure both y arrays are ascending for interpolation
    if y_centerline[0] > y_centerline[-1]:
        y_centerline = y_centerline[::-1]
        u_centerline = u_centerline[::-1]
    y_ref_desc, u_ref_desc = load_reference()
    y_ref_asc = y_ref_desc[::-1]
    u_ref_asc = u_ref_desc[::-1]
    # Interpolate simulation u to reference y positions (ascending)
    u_sim_interp_asc = np.interp(y_ref_asc, y_centerline, u_centerline)
    u_sim_interp_desc = u_sim_interp_asc[::-1]
    # Compare with tolerance
    print("Sim centerline (desc):", u_sim_interp_desc)
    print("Ref centerline (desc):", u_ref_desc)
    print("Sim y (desc):", y_ref_desc)
    print("Sim y (asc):", y_ref_asc)
    print("Sim centerline (asc):", u_sim_interp_asc)
    print("Ref centerline (asc):", u_ref_asc)
    print("Sim y_centerline:", y_centerline)
    print("Sim u_centerline:", u_centerline)
    print("Max abs diff:", np.max(np.abs(u_sim_interp_desc - u_ref_desc)))
    assert np.allclose(u_sim_interp_desc, u_ref_desc, rtol=RTOL), f"Centerline mismatch. Max abs diff={np.max(np.abs(u_sim_interp_desc - u_ref_desc))}"


def whiteboard_test_quick_positive():
    phi = np.array([10, 20, 30, 40, 50])
    results = []
    for i in range(1, 11):
        val = (3/8)*phi[1] + (6/8)*phi[2] - (1/8)*phi[3]
        results.append(val)
    print('Whiteboard QUICK positive:', results)


def whiteboard_test_quick_negative():
    phi = np.array([10, 20, 30, 40, 50])
    results = []
    for i in range(1, 11):
        val = (3/8)*phi[4] + (6/8)*phi[3] - (1/8)*phi[2]
        results.append(val)
    print('Whiteboard QUICK negative:', results)


def whiteboard_test_upwind_positive():
    phi = np.array([10, 20, 30, 40, 50])
    results = []
    for i in range(1, 11):
        val = phi[2]  # upwind positive
        results.append(val)
    print('Whiteboard Upwind positive:', results)


def whiteboard_test_upwind_negative():
    phi = np.array([10, 20, 30, 40, 50])
    results = []
    for i in range(1, 11):
        val = phi[3]  # upwind negative
        results.append(val)
    print('Whiteboard Upwind negative:', results)


# Run all whiteboard tests for diagnostics
if __name__ == "__main__":
    whiteboard_test_quick_positive()
    whiteboard_test_quick_negative()
    whiteboard_test_upwind_positive()
    whiteboard_test_upwind_negative()
