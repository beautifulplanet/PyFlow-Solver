from pyflow.benchmark.harness import benchmark

def test_benchmark_harness_small():
    res = benchmark([8, 12], steps=5)
    assert res['benchmark'] == 'lid_driven_cavity'
    assert len(res['runs']) == 2
    for r in res['runs']:
        assert 'total_time_s' in r and r['total_time_s'] >= 0.0
        assert 'avg_time_per_step_s' in r and r['avg_time_per_step_s'] > 0.0
        assert 'avg_cg_iterations' in r and r['avg_cg_iterations'] >= 0