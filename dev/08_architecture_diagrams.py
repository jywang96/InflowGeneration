# %% [markdown]
# # Architecture diagrams: old vs new
#
# Three figures for circulation, one per component of the framework, each
# contrasting the shipped structure with the reformulated one.  Every count is
# taken from the artefacts on disk rather than from the text:
#
#   * old upstream   : 10 pickles, X_train_ = (3000, 3), 72 MB each
#   * old downstream : 76 pickles, X_train_ = (2000, 3), 32 MB each
#   * new            : POD basis (N_y x M) per QoI, one GP per (QoI, mode)
#
# Storage for the new models is the Cholesky factor, N_G^2 doubles per GP.

# %%
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

if os.path.basename(os.getcwd()) == 'dev':
    os.chdir('..')
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'dev')]

FIGDIR = 'dev/figs'
os.makedirs(FIGDIR, exist_ok=True)

C_OLD, C_NEW = '#c44e52', '#4c72b0'
C_DATA, C_OUT = '0.45', '#55a868'
FS = 9


def box(ax, xy, w, h, text, fc, ec=None, fs=FS, alpha=0.13, weight='normal'):
    ec = ec or fc
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.012',
                                fc=fc, ec=ec, alpha=alpha, lw=1.4, zorder=1))
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.012',
                                fc='none', ec=ec, lw=1.4, zorder=3))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center',
            fontsize=fs, zorder=4, weight=weight, color='0.15')


def arrow(ax, p0, p1, color='0.4', text=None, fs=FS - 1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=13,
                                 lw=1.2, color=color, zorder=2,
                                 connectionstyle=f'arc3,rad={rad}'))
    if text:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.018, text,
                ha='center', va='bottom', fontsize=fs, color=color, style='italic')


def frame(ax, title, subtitle, color):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.975, title, ha='center', va='top', fontsize=12,
            weight='bold', color=color)
    ax.text(0.5, 0.925, subtitle, ha='center', va='top', fontsize=FS,
            color='0.35', style='italic')


def cost_note(ax, lines, color):
    ax.text(0.5, 0.055, '\n'.join(lines), ha='center', va='bottom',
            fontsize=FS, color=color,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=color, lw=1.1))


# %% [markdown]
# ## 1. Upstream surrogate

# %%
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 7.2))

# ---- old ----
frame(axL, 'Upstream: as shipped',
      'one model per pseudo-station, per QoI', C_OLD)
box(axL, (0.06, 0.76), 0.88, 0.10,
    r'upstream database: 20 $(h,r)$ pairs $\times$ 2 planes',
    C_DATA, alpha=0.10)
arrow(axL, (0.30, 0.76), (0.24, 0.68), text='split by plane')
arrow(axL, (0.70, 0.76), (0.76, 0.68))
for cx, lbl in [(0.24, r'$x=-4.95$'), (0.76, r'$x=-2.85$')]:
    box(axL, (cx - 0.20, 0.58), 0.40, 0.10,
        lbl + '\n' + r'20 profiles $\times$ $N_y{=}150$', C_OLD, alpha=0.10)
    for k in range(5):
        box(axL, (cx - 0.19 + k * 0.077, 0.30), 0.068, 0.20,
            f'GP\n{["u","uu","vv","ww","uv"][k]}', C_OLD, fs=FS - 1)
    arrow(axL, (cx, 0.58), (cx, 0.505))
axL.text(0.5, 0.245, r'features $(y,\,h,\,r)$   $\cdot$   '
                     r'$N_{\mathcal{G}}=20\times150=3000$ per model',
         ha='center', fontsize=FS, color='0.2')
cost_note(axL, [r'$\mathbf{10}$ Gaussian processes',
                r'$N_{\mathcal{G}} = 3000$,  $3000^2$ Cholesky',
                r'$10 \times 72$ MB $= \mathbf{720}$ MB on disk'], C_OLD)

# ---- new ----
frame(axR, 'Upstream: pooled by row index',
      'both planes in one model, indexed by $n$', C_NEW)
box(axR, (0.06, 0.76), 0.88, 0.10,
    r'40 profiles indexed by row $n$   ($n=r{-}7$ inlet, $n=r$ ALF)',
    C_DATA, alpha=0.10)
arrow(axR, (0.5, 0.76), (0.5, 0.70), text='POD per QoI')
box(axR, (0.12, 0.58), 0.76, 0.11,
    r'basis $\mathbf{\Phi}^q \in \mathbb{R}^{150\times5}$  +  mean $\bar{q}$'
    '\n' r'(one per QoI, $E_5 \geq 99.9\%$)', C_NEW, alpha=0.10)
arrow(axR, (0.5, 0.58), (0.5, 0.515), text=r'regress $a^q_j$')
for k, q in enumerate(['u', 'uu', 'vv', 'ww', 'uv']):
    for m in range(5):
        box(axR, (0.10 + k * 0.164 + m * 0.026, 0.30 + m * 0.008),
            0.10, 0.16, '', C_NEW, fs=FS - 3, alpha=0.10)
    axR.text(0.10 + k * 0.164 + 0.11, 0.375, f'{q}\n' r'$\times5$',
             ha='center', va='center', fontsize=FS - 1, color='0.15')
axR.text(0.5, 0.245, r'features $(h,\,n)$   $\cdot$   '
                     r'$N_{\mathcal{G}}=20\times2=40$ per model',
         ha='center', fontsize=FS, color='0.2')
cost_note(axR, [r'$\mathbf{25}$ Gaussian processes + 5 bases',
                r'$N_{\mathcal{G}} = 40$,  $40^2$ Cholesky',
                r'$\approx \mathbf{0.4}$ MB on disk  '
                r'($\mathbf{1800\times}$ smaller)'], C_NEW)

fig.suptitle(r'Upstream surrogate $\mathcal{S}^U$: '
             r'$(y,h,r)\!\rightarrow\!$ scalar  vs  $(h,n)\!\rightarrow\!$ '
             r'POD coefficients', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/arch_upstream.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 2. Downstream surrogate

# %%
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 7.2))

frame(axL, 'Downstream: as shipped',
      r'one model per station, per QoI  $\Rightarrow$  $x_B$ is discrete', C_OLD)
box(axL, (0.06, 0.78), 0.88, 0.09,
    r'downstream database: 20 $(h,r)$ pairs $\times$ 19 stations',
    C_DATA, alpha=0.10)
arrow(axL, (0.5, 0.78), (0.5, 0.72), text=r'split by station $x_B$')
for k in range(6):
    ec = C_OLD
    box(axL, (0.05 + k * 0.155, 0.44), 0.135, 0.26,
        (r'$x_B=$' + f'\n{[0.3, 0.6, 0.9, 1.2, "...", 13.0][k]}' +
         '\n\n' + r'4 GPs' if k != 4 else '\n\n$\\cdots$'),
        ec, fs=FS - 1, alpha=0.10)
axL.text(0.5, 0.38, r'19 stations $\times$ 4 QoI', ha='center', fontsize=FS,
         color='0.2')
axL.text(0.5, 0.31, r'features $(y,\,h,\,r)$   $\cdot$   '
                    r'$N_{\mathcal{G}}=20\times100=2000$ per model',
         ha='center', fontsize=FS, color='0.2')
box(axL, (0.20, 0.17), 0.60, 0.09,
    r'$x_B$ enters as a model index $\Rightarrow$ integer decision variable',
    C_OUT, alpha=0.10)
cost_note(axL, [r'$\mathbf{76}$ Gaussian processes',
                r'$N_{\mathcal{G}} = 2000$,  $2000^2$ Cholesky',
                r'$76 \times 32$ MB $= \mathbf{2438}$ MB on disk'], C_OLD)

frame(axR, 'Downstream: POD + coefficient regression',
      r'stations pooled  $\Rightarrow$  $x_B$ is continuous', C_NEW)
box(axR, (0.06, 0.78), 0.88, 0.09,
    r'380 profiles $= 20$ pairs $\times$ 19 stations', C_DATA, alpha=0.10)
arrow(axR, (0.5, 0.78), (0.5, 0.72), text='POD per QoI')
box(axR, (0.12, 0.58), 0.76, 0.13,
    r'basis $\mathbf{\Phi}^q \in \mathbb{R}^{150\times5}$  +  mean $\bar{q}$'
    '\n' r'(one per QoI, $E_5 \geq 99.9\%$)', C_NEW, alpha=0.10)
arrow(axR, (0.5, 0.58), (0.5, 0.50), text=r'regress $a^q_j(h,r,\log x_B)$')
for k, q in enumerate(['u', r'$I_{stream}$', r'$I_{vert}$', r'$I_{span}$']):
    for m in range(5):
        box(axR, (0.08 + k * 0.215 + m * 0.028, 0.29 + m * 0.008),
            0.13, 0.16, '', C_NEW, alpha=0.10)
    axR.text(0.08 + k * 0.215 + 0.13, 0.365, q + '\n' r'$\times5$',
             ha='center', va='center', fontsize=FS - 1, color='0.15')
axR.text(0.5, 0.235, r'features $(h,\,r,\,\log x_B)$   $\cdot$   '
                     r'$N_{\mathcal{G}}=380$ per model',
         ha='center', fontsize=FS, color='0.2')
box(axR, (0.20, 0.15), 0.60, 0.07,
    r'$x_B$ is a feature $\Rightarrow$ continuous decision variable',
    C_OUT, alpha=0.10)
cost_note(axR, [r'$\mathbf{20}$ Gaussian processes + 4 bases',
                r'$N_{\mathcal{G}} = 380$,  $380^2$ Cholesky',
                r'$\approx \mathbf{23}$ MB on disk  '
                r'($\mathbf{106\times}$ smaller)'], C_NEW)

fig.suptitle(r'Downstream surrogate $\mathcal{S}^D$: '
             r'$x_B$ from model index to regression input',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/arch_downstream.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3. Inverse problem

# %%
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 7.2))

frame(axL, 'Inverse problem: as shipped',
      'five decision variables, stochastic search', C_OLD)
box(axL, (0.10, 0.79), 0.80, 0.09,
    r'target ABL  $\rightarrow$  5 decision variables'
    '\n' r'$(h,\ r,\ x_B,\ y^{Max},\ s_U)$', C_DATA, alpha=0.10)
arrow(axL, (0.5, 0.79), (0.5, 0.71))
box(axL, (0.14, 0.55), 0.72, 0.15,
    'NSGA-II\n' r'population 128 $\times$ 220 generations',
    C_OLD, alpha=0.10)
arrow(axL, (0.5, 0.55), (0.5, 0.47))
box(axL, (0.10, 0.31), 0.80, 0.15,
    r'$\mathbf{28{,}160}$ objective evaluations'
    '\n' r'$\mathbf{112{,}640}$ surrogate calls'
    '\n' r'(re-evaluated for every new $(y^{Max}, s_U)$)', C_OLD, alpha=0.10)
arrow(axL, (0.5, 0.31), (0.5, 0.23))
box(axL, (0.14, 0.13), 0.72, 0.09,
    r'sampled Pareto front, 128 points  $\cdot$  seed-dependent',
    C_OUT, alpha=0.10)
cost_note(axL, [r'no global optimality guarantee',
                r'convergence must be argued',
                r'response surface not retained'], C_OLD)

frame(axR, 'Inverse problem: reformulated',
      r'$s_U$ eliminated in closed form, remainder enumerated', C_NEW)
box(axR, (0.10, 0.79), 0.80, 0.09,
    r'target ABL  $\rightarrow$  4 decision variables'
    '\n' r'$(h,\ r,\ x_B,\ y^{Max})$,   $s_U = s_U^\dagger$ analytic',
    C_DATA, alpha=0.10)
arrow(axR, (0.5, 0.79), (0.5, 0.71))
box(axR, (0.05, 0.55), 0.42, 0.15,
    r'grid $\Theta$'
    '\n' r'$13_h \times 41_r \times 18_{x_B}$'
    '\n' r'$= \mathbf{9{,}594}$', C_NEW, alpha=0.10)
box(axR, (0.53, 0.55), 0.42, 0.15,
    r'scan $\mathcal{Y}$'
    '\n' r'$y^{Max} \in [0.25, 0.73]$'
    '\n' r'step $0.001 \Rightarrow \mathbf{481}$', C_NEW, alpha=0.10)
arrow(axR, (0.26, 0.55), (0.42, 0.47))
arrow(axR, (0.74, 0.55), (0.58, 0.47))
box(axR, (0.10, 0.31), 0.80, 0.15,
    r'$\mathbf{4{,}614{,}714}$ objective evaluations'
    '\n' r'$\mathbf{38{,}376}$ surrogate calls'
    '\n' r'(one per $\boldsymbol{\theta}$, reused across $\mathcal{Y}$)',
    C_NEW, alpha=0.10)
arrow(axR, (0.5, 0.31), (0.5, 0.23))
box(axR, (0.14, 0.13), 0.72, 0.09,
    r'complete Pareto front  $\cdot$  deterministic  $\cdot$  15 s',
    C_OUT, alpha=0.10)
cost_note(axR, [r'global optimum on $\Theta \times \mathcal{Y}$',
                r'no convergence criterion needed',
                r'full response surface available'], C_NEW)

fig.suptitle(r'Inverse problem: 5-variable metaheuristic  vs  '
             r'closed-form $s_U$ + enumeration', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/arch_optimisation.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
print('written:')
for f in ['arch_upstream', 'arch_downstream', 'arch_optimisation']:
    p = f'{FIGDIR}/{f}.png'
    print(f'  {p}  {os.path.getsize(p)/1024:.0f} KB')
