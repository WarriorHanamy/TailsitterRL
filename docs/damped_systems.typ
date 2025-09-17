#import "symbols.typ": *

= Theory

== Common Special Case: Mass-Spring-Damper System

For a mechanical system with mass $m$, velocity damping coefficient $b$
and spring constant
$k$, the equation of motion is
$
  m dot.double(x) + b dot(x) + k x = f(t)
$


== Standardized Control System Form
$
  dot.double(x) + 2 zeta omega_n dot(x) + omega_n^2 x = u (t)
$
where, $omega_n = sqrt(k/m)$, $zeta = b/(2 sqrt(m k))$

Types of Responses (depending on damping ratio $zeta$):
1. Overdamped ($zeta > 1$)
2. Critically damped ($zeta = 1$)
3. Underdamped ($0 < zeta < 1$)
4. Undamped ($zeta = 0$)


== Best damping ratio
$zeta = sqrt(2)$ is often considered as the best damping ratio. This is because it provides a good balance between response speed and overshoot.
1. Low damping ratio ($zeta < sqrt(2)$) can lead to overshoot and oscillations.
2. High damping ratio ($zeta > sqrt(2)$) can lead to sluggish response.
See. [docs/damped_systems.ipynb]

=== Mag. Response of Second-Order System
$
  |H(j omega)| = omega_n^2 / sqrt((omega_n^2 - omega^2)^2 + (2 zeta omega_n omega)^2)
$
The bandwidth is defined as
$
  omega_"BW" = omega_n sqrt(1 - 2 zeta^2 + sqrt(4 zeta^4 - 4 zeta^2 + 2))
$
When $zeta = 1/sqrt(2)$, $omega_"BW" = omega_n$.

=== Maximially Flat Response
Let $x = omega/omega_n$ (Normalized Frequency, low-frquency: $x << 1$)
$
  |H(j omega)|^2
  = 1 / ((1 - x^2)^2 + (2 zeta x)^2) = 1/(1+(4 zeta^2 - 2)x^2 + x^4)
  \
  = 1 - (4 zeta^2 - 2)x^2 + ( (4 zeta^4 - 2)^2 - 1 )x^4 + O(x^6)
$

Maximially flat means : $(4 zeta^2 - 2) = 0$, i.e., $zeta = 1/sqrt(2)$

这正是二阶 Butterworth（最大平坦） 条件
$
  omega_c = omega_n
$



= Practice
In Practice, the closed-loop dynamics of the angular velocity of a flying machine is often modeled as a second-order system with a certain damping ratio $zeta$ and closed-loop bandwidth $omega_"BW"$.



$
  s^2 + 2 zeta omega_n s + omega_n^2 = omega_n^2 u
$
特别地,
$
  dot.double(BodyRateB) + 2 zeta omega_n DotBodyRateB + omega_n^2 BodyRateB = omega_n^2 BodyRateB^cal(A)
$
where $BodyRateB^cal(A)$ is normalied action signal.


The Euler integration of the above equation is, if sim integration timestep is $Delta t$ and let $u = BodyRateB^cal(A)$

$
  BodyRateB_(k+1) &= BodyRateB_k + DotBodyRateB_k Delta t
  \
  DotBodyRateB_(k+1) &= DotBodyRateB_k + ( -2 zeta omega_n DotBodyRateB_k - omega_n^2 BodyRateB_k + omega_n^2 u_k ) Delta t
  \
  &=(1-2zeta omega_n Delta t) DotBodyRateB_k - omega_n^2 Delta t BodyRateB_k + omega_n^2 Delta t u_k
$

if take $
          z = mat(
            BodyRateB;
            DotBodyRateB
          )
        $
, then
$
  z_(k+1) = A z_k + B u_k
$
where
$
  A = mat(
    1, Delta t;
    -omega_n^2 Delta t, 1 - 2 zeta omega_n Delta t
  )
  \
  B = mat(
    0;
    omega_n^2 Delta t
  )
$
