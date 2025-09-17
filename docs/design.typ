
#import "@preview/elsearticle:1.0.0": *
#import "symbols.typ": *

#show: elsearticle.with(
  title: "Design of this project",
  authors: (
    (
      name: "Hanamy",
      affiliation: "SYSU",
      corr: "rongerch@outlook.com",
      id: "0000-0002-1825-0097",
    ),
  ),
  journal: "Proceedings of Machine Learning Research",
  abstract: [],
  keywords: ("Reinforcement Learning", "VTOL", "Model Predictive Control"),
  format: "review",
  // line-numbering: true,
)

= Goal
Current Goal is to find a *control policy* from a differentiable simulation (with dynamics are simplified and known) to complete a forward transition control policy for a VTOL aircraft (in the prototype of a tailsitter).

The reward or the loss that guides the training process is defined as follows:

The look-ahead time horizon is $T$, the control period is predefined as $T_c < T$ and divides $T$. Thus, the continuous forward transition problem is discretized into $N = T / T_c$ steps, with a simulation time step of $Delta t$. $t_k = k * Delta t$ for $k = 0, 1, \ldots, N$.

$
  cal(L) = sum_(k = 0)^(N) cal(L)_(k, "height hold term") + cal(L)_(k, "time penalize") + cal(L)_(k, "control energy penalize")
$

== Specific Goal
it should be catergorized into two type of goals: long-sighted goal and short-sighted goal.

The long-sighted goal is defined as follows:
1. reach a target position $Pos_"target" = (x_"target", y_"target", z_"target")$ at time $T$;
2. keep a target speed $Vel_"target" = (v_x_"target", v_y_"target", v_z_"target")$ at time $T$;
3. keep a target attitude $Attitude_"target" = (phi_"target", theta_"target", psi_"target")$ at time $T$;
4. keep a altitude $z_"target"$ during the whole flight.

The short-sighted goal is defined as follows:
1. (depth-image based) collision avoidance;
2. control energy minimization. (The smoothness of the trajectory is also considered here.)


Normalized Control Signal are preferred.



= State and Action
== cardinality
$
  |cal(A)| = m quad |cal(S)| = n
$

$
  State = (Pos, Attitude, Vel, BodyRateB, ...)
$

The state is defined as follows:
$
  State = (Pos, Vel, Acc_"thr", Attitude, BodyRateB) oplus ("Aerodynamic model")
$
where $oplus$ means possible concatenation.


The continuous transition function is defined as follows:
$
  DotState = f(State, Action)
$


i.e.,
$
  dot(Vel) = underbrace(Acc_"thr" + AccGrav, "preditable") + underbrace(Acc_"aero" + Acc_"dist", "slowly varying on the time scale of control period")
$

$
  dot(Acc)_"thr" = 1/tau (Acc_"thr,cmd" - Acc_"thr") // first order response with time constant tau. I feel that this could be fast.
$


$
  DotBodyRateB = f(BodyRateB, BodyRateB_"cmd")
  \
  "this would be a second-order closed-loop dynamics"
$
Another way to think about this sub-dynamics is from physical perspective:
$
  DotBodyRateB = J^(-1) (Moment_"thr" +
    underbrace(Moment_"aero" - BodyRateB times (J BodyRateB), "slowly-varying")
  )
$

$
  DotBodyRateB = J^(-1) (Moment_"thr") + "Bias"
$


So, reduce $||dot.double(BodyRateB)||$ over time is equivalent to reduce $||dot(Moment_"thr")||$ over time.

// The action is defined as follows:
// $
//   Action = (Acc_"thr,cmd", BodyRateB_"cmd")
// $

Maybe we should have a fined-grained control model for control energy penalize term (short-sighted).

$
  BodyRateB_(k+1) = BodyRateB_(k) + DotBodyRateB_(k) T_(c') + 1/2 dot.double(BodyRateB)_(k) T_(c')^(2)
$

$
  DotBodyRateB_(k+1) = DotBodyRateB_k
  + 1/2 (
    dot.double(BodyRateB)_(k+1)
    +
    dot.double(BodyRateB)_(k)
  ) T_(c')
$


== Action Design
$
  Action = (Acc_"thr,cmd", BodyRateB_"cmd")
$

= Penalties Design


== Height Hold Term
We beilieve that this term can be merged into the control energy penalize term, which depends on the differentiable model we define.

At now, we define it as follows:
$
  cal(L)_(k, "height hold term") = w_h * (z(t_k) - )^(2)
$


== Time Penalize
$
  cal(L)_(k, "time penalize") = rho_T "softplus"(t_k - T + d_"shift")
  \ "if pos is outside the target zone"
$
