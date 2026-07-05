> **ARCHIVED — AIM SUPERSEDED by `plans/run13-selfreproducing-cell.md` §2** (a
> generation-0 *optimizer* is the wrong target). The **math** here (curvature-measure
> coordinate §2) is still live: run13 §1 reuses it verbatim. Kept as the coordinate spec.

# Optimizing generation 0 — formulation and parameterized construction

Goal: pin down *exactly* what "generation 0" is, in coordinates an optimizer can move in,
so that maximizing $J$ over those coordinates approaches the best single-generation value.
This document is specification only — the objective, the degrees of freedom, and the
feasibility constraints. No solver, no code.

---

## 1. The functional and the constraints

On $Q=[-1,1]\times[0,1]$,

$$
J[f,g]=\int_0^1\!\!\int_{-1}^1 f_t(x,t)\,g_{xx}(x,t)\,dx\,dt .
$$

| | $f$ | $g$ |
|---|---|---|
| convex in $x$ | $f_{xx}\ge0$ | $g_{xx}\ge0$ |
| monotone in $t$ | $f_t\ge0$ | $g_t\le0$ |
| Lipschitz | $\lvert f_x\rvert\le1$ | $\lvert g_x\rvert\le1$ |
| spatial boundary | $f(\pm1,t)=0$ | $g(\pm1,t)=0$ |
| temporal boundary | $f(x,1)=0$ | $g(x,0)=0$ |

Both integrands are $\ge0$. Two budgets bound everything:

$$
\underbrace{\int_0^1 f_t\,dt=-f(x,0)\le 1-\lvert x\rvert}_{\text{rise, per column}},
\qquad
\underbrace{\int_{-1}^1 g_{xx}\,dx=g_x(1,t)-g_x(-1,t)\le 2}_{\text{curvature, per slice}} .
$$

---

## 2. A convex slice **is** a curvature measure (the working coordinate)

Fix $t$. A convex $f(\cdot,t)$ with $f(\pm1,t)=0$ is exactly

$$
f(x,t)=\int_{-1}^{1} G(x,y)\,\mu^f_t(dy),\qquad
G(x,y)=\tfrac12\bigl(\min(x,y)+1\bigr)\bigl(\max(x,y)-1\bigr)\le 0,
$$

where $\mu^f_t=f_{xx}(\cdot,t)\ge0$ is the **curvature measure**. Key facts:

- $G(\pm1,y)=0$, so **$f(\pm1,t)=0$ holds automatically** for any $\mu^f_t\ge0$; the spatial
  boundary is free.
- $G_{xx}=\delta$, so $f_{xx}=\mu^f_t$ (convexity $\Leftrightarrow \mu^f_t\ge0$).
- Arm slopes: $f_x(-1,t)=\tfrac12(\langle y\rangle-\mathrm{mass})$, $f_x(1,t)=\tfrac12(\langle y\rangle+\mathrm{mass})$,
  with $\mathrm{mass}=\int\mu^f_t$, $\langle y\rangle=\int y\,\mu^f_t(dy)$.
- **Lipschitz** $\lvert f_x\rvert\le1 \Leftrightarrow \bigl\lvert\langle y\rangle\pm\mathrm{mass}\bigr\rvert\le2$.
  In the melt regime the arms are **pinned** at $\pm1$, which is exactly

$$
\boxed{\ \mathrm{mass}=\int\mu^f_t=2,\qquad \text{first moment}=\int y\,\mu^f_t(dy)=0\ }
$$

i.e. total curvature $2$, barycenter $0$. The **tent** is the extreme point $\mu=2\delta_0$
(all curvature at the center); $G(x,0)\cdot2=-(1-\lvert x\rvert)$.

Everything above holds verbatim for $g$ with $\mu^g_t=g_{xx}(\cdot,t)$.

> Reading: "melting" = spreading $\mu$ off a single atom; "drifting" = moving $\mu$'s mass
> off-center — but the barycenter must stay $0$, so drifting one lump right **requires**
> a balancing lump left. That balancing mass is the "full-tent end slice" the optimum parks.

---

## 3. Time is a monotone flow of curvature measures

The two temporal constraints become monotonicity of the recovered slice, plus a vanishing
endpoint:

- **$f$:** $f_t\ge0 \Leftrightarrow t\mapsto f(x,t)=\int G\,d\mu^f_t$ is **non-decreasing** for
  every $x$; and $f(x,1)=0 \Leftrightarrow \mu^f_1=0$. So $\mu^f_t$ flows from $2\delta_0$
  (tent, $t=0$) down to $0$ (flat, $t=1$): curvature **dissolves** as $f$ rises.
- **$g$:** $g_t\le0 \Leftrightarrow t\mapsto g(x,t)$ is **non-increasing** for every $x$; and
  $g(x,0)=0 \Leftrightarrow \mu^g_0=0$. So $\mu^g_t$ flows from $0$ (flat, $t=0$) up to its
  full configuration (tent-like, $t=1$): curvature **accumulates** as $g$ deepens.

The mass need not equal $2$ at every $t$ — near the flat ends the arms may droop
($\mathrm{mass}<2$). Mass $=2$, moment $=0$ is the *interior/melt* regime where the harvest lives.

---

## 4. The objective is a pairing — and the obstruction

Because $g_{xx}(\cdot,t)=\mu^g_t$,

$$
J=\int_0^1\!\Bigl(\int_{-1}^1 f_t(y,t)\,\mu^g_t(dy)\Bigr)dt .
$$

**Alignment principle.** At each $t$, $g$ contributes $\int f_t\,d\mu^g_t$. Given the rise field
$f_t(\cdot,t)\ge0$, this is maximized by placing $\mu^g_t$'s mass (total $2$, barycenter $0$)
on the **argmax of $f_t$**. Symmetrically, $f$ should route its rise to wherever $\mu^g_t$ sits.
Optimizing $g$ against fixed $f$, or $f$ against fixed $g$, is *linear*; the coupling is the
whole difficulty.

**Tent cap.** If $\mu^g_t$ is stationary (a fixed tent at $x_0$),
$J=\int_0^1 m(t)\,f_t(x_0,t)\,dt\le2(1-\lvert x_0\rvert)\le2$. **Any $J>2$ forces $\mu^g_t$ to move.**

---

## 5. What "generation 0" is: **one sweep**

> **Definition (generation 0).** The active curvature makes a **single monotone pass** of its
> center across the interval over $[0,1]$ — one sweep, no reversal — balanced by mass parked
> near the ends. Formally, the drift coordinate $c(t)$ (barycenter of the moving cluster) is
> monotone in $t$. Special cases: $c\equiv0$ is the *centered melt*; a point mass is the *tent*.

Generation $k$ reverses the sweep $2^k-1$ times (i.e. $2^k$ passes at time-scale $2^{-k}$,
riding the coarser drift). Restricting to a monotone $c(t)$ is precisely the "no self-similar
time refinement" that isolates generation 0. Its value is bounded (empirically $J\approx2.6$);
adding one reversal is generation 1.

---

## 6. The generation-0 ansatz (degrees of freedom)

Represent each curvature measure by a fixed, small number $K$ of **moving atoms** — the
single-scale realization. (An atom of $f_{xx}$ is a kink of $f$; $K$ atoms give a convex,
piecewise-linear slice with $K$ kinks and arms $\pm1$.)

$$
\mu^f_t=\sum_{j=1}^{K} a_j(t)\,\delta_{\alpha_j(t)},\qquad
\mu^g_t=\sum_{i=1}^{K} b_i(t)\,\delta_{\beta_i(t)} .
$$

**Recover the fields** by superposing Green's kernels:

$$
f(x,t)=\sum_{j} a_j(t)\,G\!\bigl(x,\alpha_j(t)\bigr),\qquad
g(x,t)=\sum_{i} b_i(t)\,G\!\bigl(x,\beta_i(t)\bigr).
$$

**Free parameters** (the optimization variables) are the trajectories

$$
a_j(t)\ge0,\ \ \alpha_j(t)\in[-1,1],\qquad
b_i(t)\ge0,\ \ \beta_i(t)\in[-1,1],\qquad t\in[0,1].
$$

**Feasibility constraints.**

- **(F1) arms / Lipschitz (per $t$).** Pinned: $\ \sum_j a_j=2,\ \ \sum_j a_j\alpha_j=0$;
  and likewise $\sum_i b_i=2,\ \sum_i b_i\beta_i=0$. General (drooping) form:
  $\bigl\lvert\sum_j a_j\alpha_j\pm\sum_j a_j\bigr\rvert\le2$.
- **(F2) ordering.** $\alpha_1(t)\le\cdots\le\alpha_K(t)$ in $[-1,1]$ (same for $\beta$).
- **(F3) time monotonicity.** $f(x,t)=\sum_j a_jG(x,\alpha_j)$ non-decreasing in $t$ for all $x$
  ($f_t\ge0$); $g(x,t)$ non-increasing in $t$ for all $x$ ($g_t\le0$).
- **(F4) terminal.** $a_j(1)=0$ (so $f(\cdot,1)=0$); $b_i(0)=0$ (so $g(\cdot,0)=0$).
- **(F5) generation-0 restriction.** the barycenter $\sum_i b_i\beta_i$ is pinned to $0$ by
  (F1), so the sweep is carried by the **position of the active atom** — the one holding most
  of the curvature. Let $c(t)=\beta_{i^\*(t)}(t)$ with $i^\*=\arg\max_i b_i$. Generation 0
  requires $c(t)$ to be **monotone** in $t$ (single pass, no reversal).

**Objective in these parameters.** Since $\mu^g_t$ is atomic,

$$
\boxed{\ J=\int_0^1\ \sum_{i=1}^{K} b_i(t)\,f_t\!\bigl(\beta_i(t),\,t\bigr)\,dt\ }
$$

with the rise field read off from $f$'s trajectories,

$$
f_t(x,t)=\sum_{j=1}^{K}\Bigl[\ \dot a_j(t)\,G\!\bigl(x,\alpha_j(t)\bigr)
+ a_j(t)\,G_y\!\bigl(x,\alpha_j(t)\bigr)\,\dot\alpha_j(t)\ \Bigr],
\qquad
G_y(x,\alpha)=
\begin{cases}\tfrac12(x+1), & x<\alpha\\[2pt]\tfrac12(x-1), & x>\alpha.\end{cases}
$$

That is a complete optimal-control problem: state = the atom trajectories, dynamics/limits =
(F1)–(F5), payoff = $J$. Increasing $K$ enriches the band shape; keeping $c(t)$ monotone (F5)
keeps it generation 0.

---

## 7. Design targets (what a maximizer is chasing)

- **Alignment.** Drive $\beta_i(t)$ (where $g$ is kinked) onto the argmax of $f_t$, and route
  $f$'s rise to the same moving location $c(t)$. Misaligned drift scores *below* the tent cap.
- **Amplification.** Give one "active" atom most of the curvature and let it travel a long
  distance $L$ while staying thin (small effective width $w$); gain $\sim L/w$ against the
  fresh rise budget $1-\lvert x\rvert$ it visits.
- **Balance = parked mass.** To hold $\sum b_i\beta_i=0$ while the active atom sits at $c\ne0$,
  a companion atom must park near the opposite end — this is the "full-tent end slice" that
  stores rise/curvature stock.
- **Depth cap.** A slice whose minimum sits at $x=c$ can be at most $1-\lvert c\rvert$ deep;
  drifting far therefore forces shallow, which is why one sweep is bounded.
- **Ceiling.** A single monotone sweep saturates around $J\approx2.6$. Letting $c(t)$ reverse
  once (two passes) is the move to generation 1; the recursion of reversals is the cascade.

---

## 8. Smallest expressive instance ($K=2$)

Twlo atoms are the minimum that can drift off-center (one atom is forced to $x=0$ by the
barycenter constraint). For $g$: an **active** atom $(b_1,\beta_1)$ with $\beta_1(t)=c(t)$ the
monotone sweep, and a **balancer** $(b_2,\beta_2)$, tied by

$$
b_1+b_2=2,\qquad b_1\,c + b_2\,\beta_2=0\ \Rightarrow\ \beta_2=-\frac{b_1\,c}{\,2-b_1\,}.
$$

The recovered $g(\cdot,t)$ is then a convex, two-kink slice (a slanted-bottom basin) with arms
$\pm1$, whose bottom tracks $c(t)$. Mirror the same $K=2$ construction for $f$ (with its rise
concentrated at $c(t)$), and the pair is the leanest generation-0 competitor; raising $K$ only
softens the band into a smoother melt. The free schedules left to optimize are $c(t)$
(monotone), the mass split $b_1(t)$, and the matching $f$-side trajectories — everything else
is fixed by (F1)–(F5).
