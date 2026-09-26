"""Pass counts of the Schur iteration over every case of the oracle suite `schur` (the table of the
WP 8.5-P16 report): `python3 crates/nalgebra/src/linalg/schur_model/stats.py`."""
import sys, json, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
import model
from model import *
model.STUCK_K=8; model.NEAREST=True; model.HESS_NEAREST=False; model.EXC=True
import model_f64 as F
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 5)))
d = json.load(open(os.path.join(ROOT, 'tools', 'oracle', 'vectors', 'schur.json')))
ops={o['name']:o for o in d['ops']}
print("| n | family | cases | passes median / p90 / max | f64 upstream median / max |")
print("|---|---|---:|---|---|")
for n in (3,4,5,6):
    for fam in ['', '_real', '_complex', '_nonnormal', '_clustered', '_defective', '_near_triangular']:
        op=f"schur{n}_eigenvalues{fam}"
        its=[]; fits=[]
        for c in ops[op]['cases']:
            a=c['in'][0]
            r=schur(Mat(n,[x[:] for x in a]), thr_abs=64, compute_q=False, max_guard=5000)
            its.append(r[2] if r[0]!='DIVERGED' else 99999)
            rf=F.schur(F.Mat(n,[[v/2**32 for v in x] for x in a]), eps=F.EPS, thr_abs=F.EPS**2, max_guard=5000)
            fits.append(rf[2] if rf[0]!='DIVERGED' else 99999)
        its.sort(); fits.sort()
        print(f"| {n} | {fam[1:] or 'general'} | {len(its)} | {its[len(its)//2]} / {its[int(len(its)*0.9)]} / {its[-1]} | {fits[len(fits)//2]} / {fits[-1]} |")
