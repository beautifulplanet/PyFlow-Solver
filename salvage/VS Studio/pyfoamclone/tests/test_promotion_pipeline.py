from pathlib import Path
from pyfoamclone.tools.promotion import promote


def test_promotion_report_pass():
    module = Path('experiments') / 'sample_module_proto.py'
    rep = promote(module, apply=False)
    d = rep.to_dict()
    assert d['status'] == 'pass'
    assert d['doc_coverage'] == 100.0
    assert d['functions'] >= 1