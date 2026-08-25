"""Environment smoke test -- run this first, in any new environment.

Motivation: a conda-forge py3.9 environment on Windows solved to a numpy/scipy
pair whose OpenBLAS DLL failed to load.  Plain arithmetic and BLAS matmuls
worked, so imports all succeeded and nothing looked wrong, but *every* LAPACK
call terminated the interpreter with no traceback and exit code 127:

    np.linalg.svd              -> dead     (needed for POD)
    scipy.linalg.solve         -> dead
    interp1d(kind='cubic')     -> dead     (needed by loadData(yInterp=...))
    GaussianProcessRegressor.predict(return_std=True) -> dead

Each check below runs in a subprocess, so a hard crash is reported instead of
taking the whole run down with it.

    python dev/00_env_check.py
"""

import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CHECKS = [
    ('versions', """
        import os, sys, numpy, scipy, sklearn, pandas, joblib
        print('python  ', sys.version.split()[0])
        print('numpy   ', numpy.__version__)
        print('scipy   ', scipy.__version__)
        print('sklearn ', sklearn.__version__)
        print('pandas  ', pandas.__version__)
        print('MKL_THREADING_LAYER', os.environ.get('MKL_THREADING_LAYER', '<unset>'))
    """),
    ('BLAS matmul', """
        import numpy as np
        A = np.random.rand(300, 300)
        assert np.isfinite(A @ A).all()
    """),
    ('LAPACK: svd  (POD)', """
        import numpy as np
        U, S, Vt = np.linalg.svd(np.random.rand(300, 50), full_matrices=False)
        assert S[0] > 0
    """),
    ('LAPACK: cholesky + triangular solve  (GPR return_std)', """
        import numpy as np
        from scipy.linalg import solve_triangular
        A = np.random.rand(200, 200); A = A @ A.T + 200*np.eye(200)
        L = np.linalg.cholesky(A)
        assert np.isfinite(solve_triangular(L, np.ones(200), lower=True)).all()
    """),
    ('LAPACK: banded solve  (interp1d cubic)', """
        import numpy as np
        from scipy.interpolate import interp1d
        x = np.linspace(0, 1, 50)
        assert np.isfinite(interp1d(x, np.sin(x), kind='cubic')(0.5))
    """),
    ('pymoo', """
        from pymoo.algorithms.moo.nsga2 import NSGA2
        NSGA2(pop_size=8)
    """),
    ('repo import', """
        import sys; sys.path.insert(0, r'{root}')
        import modelDefinition, hyperparametersGPR
    """),
    ('loadData (raw)', """
        import sys; sys.path.insert(0, r'{root}')
        import os; os.chdir(r'{root}')
        from modelDefinition import loadData
        d = loadData([0.04], [0.6], [52], 1.0, './GPRDatabase', 15.0)
        assert len(d) > 100, d.shape
    """),
    ('loadData (yInterp -> cubic)', """
        import sys; sys.path.insert(0, r'{root}')
        import os; os.chdir(r'{root}')
        import numpy as np
        from modelDefinition import loadData
        d = loadData([0.04], [0.6], [52], 1.0, './GPRDatabase', 15.0,
                     np.linspace(0.01, 1.0, 100))
        assert len(d) == 100, d.shape
    """),
    ('unpickle GPR + predict(return_std=True)', """
        import sys; sys.path.insert(0, r'{root}')
        import os; os.chdir(r'{root}')
        import numpy as np, joblib
        p = '../GPRModels/0p6_intensities_u.pkl'
        if not os.path.exists(p):
            print('SKIP -- ' + p + ' not present'); raise SystemExit(0)
        m = joblib.load(p)
        print('  kernel:', m.kernel_, ' n_train:', m.X_train_.shape[0])
        mu, sd = m.predict(np.zeros((3, 3)), return_std=True)
        assert np.isfinite(mu).all() and np.isfinite(sd).all()
    """),
]


def main():
    width = max(len(n) for n, _ in CHECKS)
    failed = []
    for name, code in CHECKS:
        src = textwrap.dedent(code).format(root=ROOT)
        r = subprocess.run([sys.executable, '-W', 'ignore', '-c', src],
                           capture_output=True, text=True)
        ok = r.returncode == 0
        print(f'{name:<{width}}  {"ok" if ok else "FAIL":>4}')
        for line in (r.stdout or '').splitlines():
            print(f'{"":<{width}}    {line}')
        if not ok:
            failed.append(name)
            tail = (r.stderr or '').strip().splitlines()[-3:]
            if tail:
                for line in tail:
                    print(f'{"":<{width}}    ! {line}')
            else:
                print(f'{"":<{width}}    ! hard crash, exit {r.returncode}, '
                      f'no traceback (native/DLL failure)')

    print()
    if failed:
        print(f'{len(failed)} check(s) failed: {", ".join(failed)}')
        return 1
    print('environment looks good')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
