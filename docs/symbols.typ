

#let AccGrav = $bold(g)$
#let AttitudeB = $upright(bold(R))_(cal(B))$
#let Attitude = $upright(bold(R))$
#let Iner = $upright(bold(J))$
#let BodyRate = $bold(Omega)$
#let StabilityRate = $attach(bold(Omega)_s, tl: cal(S))$
#let AttitudeS = $Attitude_(cal(S))$


#let Pos = $bold(p)$
#let Vel = $bold(v)$
#let Acc = $bold(a)$
#let Jerk = $bold(j)$
#let AccP = $bold(a)_p$
#let AccPS = $attach(AccP, tl: cal(S))$
#let CtrlEffective = $bold(B)$
#let Ctrl = $bold(u)$

#let xb = $bold(x)_b$
#let zb = $bold(z)_b$

#let DotState = $dot(bold(x))$

#let DotATmax = $dot(a)_T, "max"$
#let DotAt = $dot(a)_T$
#let BodyRateBmax = $attach(bold(Omega), tl: cal(B))_(b, "max")$

#let BodyRateBx = $attach(bold(Omega)_(b_x), tl: cal(B))$
#let BodyRateBy = $attach(bold(Omega)_(b_y), tl: cal(B))$
#let BodyRateBz = $attach(bold(Omega)_(b_z), tl: cal(B))$

#let BodyAccBx = $attach(dot(bold(Omega))_(b_x), tl: cal(B))$
#let BodyAccBy = $attach(dot(bold(Omega))_(b_y), tl: cal(B))$
#let BodyAccBz = $attach(dot(bold(Omega))_(b_z), tl: cal(B))$

#let BodyAccBxMax = $attach(dot(bold(Omega))_(b_(x,"max")), tl: cal(B))$
#let BodyAccByMax = $attach(dot(bold(Omega))_(b_(y,"max")), tl: cal(B))$
#let BodyAccBzMax = $attach(dot(bold(Omega))_(b_(z,"max")), tl: cal(B))$


#let BodyRateBxCmd = $attach(bold(Omega)_(b_x)^(upright(c)), tl: cal(B))$
#let BodyRateByCmd = $attach(bold(Omega)_(b_y)^(upright(c)), tl: cal(B))$
#let BodyRateBzCmd = $attach(bold(Omega)_(b_z)^(upright(c)), tl: cal(B))$


#let DotBodyRateB = $attach(dot(bold(Omega))_(b), tl: cal(B))$

#let DotBodyRateBx = $attach(dot(bold(Omega))_(b_x), tl: cal(B))$
#let DotBodyRateBy = $attach(dot(bold(Omega))_(b_y), tl: cal(B))$
#let DotBodyRateBz = $attach(dot(bold(Omega))_(b_z), tl: cal(B))$

#let Mx = $bold(M)_x$
#let My = $bold(M)_y$
#let Mz = $bold(M)_z$


#let DotAccPSx = $dot(AccPS_x)$
#let DotAccPSz = $dot(AccPS_z)$
#let DotAccPS = $dot(AccPS)$

#let AttitudeSB = $attach(AttitudeB, tl: cal(S))$
#let AttitudeSBExpansion = $
  mat(
    cos alpha, 0, sin alpha;
    0, 1, 0;
    -sin alpha, 0, cos alpha
  )
$

#let AttitudeBS = $attach(AttitudeS, tl: cal(B))$
#let AttitudeBSExpansion = $
  mat(
    cos alpha, 0, -sin alpha;
    0, 1, 0;
    sin alpha, 0, cos alpha
  )
$



#let DomainDF = $cal(D)_"df"$


#let BodyX = $bold(x)_b$
#let BodyY = $bold(y)_b$
#let BodyZ = $bold(z)_b$




#let BodyYRef = $bold(y)_b^("ref")$
#let BodyYxRef = $bold(y)_(b_x)^("ref")$
#let BodyYyRef = $bold(y)_(b_y)^("ref")$
#let BodyYzRef = $bold(y)_(b_z)^("ref")$



#let InerMat = $upright(bold(J))$
#let MomentB = $attach(bold(M), tl: cal(B))$
#let Moment = $bold(M)$

#let forceT = $bold(f)_T$
#let forceTB = $attach(bold(f)_T, tl: cal(B))$
#let forceAero = $bold(f)_a$
#let forceAeroB = $attach(bold(f)_a, tl: cal(B))$
#let forceAeroS = $attach(bold(f)_a, tl: cal(S))$

#let IdeMat = $upright(bold(I))$
#let quaternion = $bold(q)$
#let yaw = $psi$
#let pitch = $theta$
#let roll = $phi.alt$
#let bigP = body => math.lr(size: 1.5em)[(#body)]

#let difft = $upright(d)/(upright(d) t)$

#let AeroForceB = $attach(bold(f)_a, tl: cal(B))$
#let VelB = $attach(bold(v), tl: cal(B))$
#let VelS = $attach(bold(v), tl: cal(S))$

#let VelBx = $VelB_x$
#let VelBy = $VelB_y$
#let VelBz = $VelB_z$

#let DotVelBx = $dot(VelB_x)$
#let DotVelBy = $dot(VelB_y)$
#let DotVelBz = $dot(VelB_z)$


#let DFforward = $cal(D)_(cal(F))$
#let DFbackward = $cal(D)_(cal(B))$

#let BodyFrame = $cal(B)$
#let StabilityFrame = $cal(S)$

#let right = text(fill: green)[✔]
#let wrong = text(fill: red)[❌]
// #let f_a = $bold(f_a)$
#let VirCtrl = $bold(nu)$

// inequalities
#let ExtGt = $succ.eq$
#let ExtLt = $prec.eq$
#let ExtLtStrict = $prec$
#let ExtGtStrict = $succ$

// Optimzation
#let DecVars = $bold(x)$
#let Jacobian = $bold(J)$


// aerodynamics
#let AeroDensity = $rho$


// attitude dynamics
#let EulerAngles = $bold(Theta)$
#let DisturbAngles = $upright(bold(d))_(EulerAngles)$
#let BodyRateB = $attach(bold(Omega)_b, tl: cal(B))$
#let DisturbRates = $upright(bold(d))_(BodyRate)$
#let Rotation = $bold(R)$
#let RotSinB = $attach(Rotation_(cal(S)), tl: cal(B))$
#let RotBinS = $attach(Rotation_(cal(B)), tl: cal(S))$


#let RB = $bold(R)_(cal(B))$


// controls
#let Actuators = $bold(delta)$
#let CtrlActuators = $upright(bold(B))_(Actuators)$
#let Torque = $bold(tau)$

#let Chirp = $upright(bold(d))_("chirp")$
#let Control = $bold(u)$
#let Action = $bold(u)$

#let StabilityX = $bold(x)_s$
#let StabilityY = $bold(y)_s$
#let StabilityZ = $bold(z)_s$
// General
#let State = $bold(x)$
#let Input = $bold(u)$

#let SE3 = $text("SE(3)", font: "DejaVu Sans")$
#let SO3 = $text("SO(3)", font: "DejaVu Sans")$


#let Ex = $bold(e_x)$
#let Ey = $bold(e_y)$
#let Ez = $bold(e_z)$
#let rr = $bold(r)$



// thrust
#let MomentT = $bold(M)_T$
#let MomentTB = $attach(MomentT, tl: cal(B))$
#let MomentAero = $bold(M)_a$
#let MomentAeroB = $attach(MomentAero, tl: cal(B))$
#let ForceTBx = $attach(forceT_x, tl: cal(B))$
#let MomentTBx = $attach(MomentT_x, tl: cal(B))$
#let MomentTBy = $attach(MomentT_y, tl: cal(B))$
#let MomentTBz = $attach(MomentT_z, tl: cal(B))$




// planning
#let IntermediateWaypoints = $upright(bold(Q))$
#let TimeAllocation = $upright(bold(T))$
#let Waypoint = $bold(q)$

#let MINCO = $frak(T)_"MINCO"$
#let Coefs = $upright(bold(C))$


#let DynamicConstr = $cal(G)_cal(D)$
#let CorridorConstr = $cal(G)_cal(C)$
#let dt = $upright("d") t$



#let AccPC = $AccP^(upright("c"))$
#let AttitudeBC = $Attitude_(cal(B))^(upright("c"))$
#let fTC = $f_T^(upright("c"))$



#let VelRef = $Vel^("ref")$
#let PosRef = $Pos^("ref")$
#let BodyYRef = $BodyY^("ref")$
#let AccPRef = $AccP^("ref")$

#let PosRefk = $Pos^("ref")_k$
#let VelRefk = $Vel^("ref")_k$
#let AccPRefk = $AccP^("ref")_k$
#let BodyYRefk = $attach(bold(y), tr: "ref", br: b_"k")$
#let BodyRateBRefk = $attach(BodyRateB, tr: "ref")$
#let SubjectTo = $"s.t."$



#let throttle = $"throttle"^(upright(c))$
#let tauCmd = $bold(tau)^(upright(c))$

#let forceAeroByExpansion = $attach(bold(c)_y, tl: cal(B))$
#let forceAeroBxExpansion = $attach(bold(c)_x, tl: cal(B))$
#let forceAeroBzExpansion = $attach(bold(c)_z, tl: cal(B))$


#let cz = $attach(bold(c)_z, tl: cal(B))$
#let cx = $attach(bold(c)_x, tl: cal(B))$
#let cy = $attach(bold(c)_y, tl: cal(B))$


#let fc = $bold(f)_c$



#let AccPBz = $attach(AccP_z, tl: cal(B))$
#let AccPBx = $attach(AccP_x, tl: cal(B))$
#let AccPBy = $attach(AccP_y, tl: cal(B))$



#let MPCState = $bold(s)$

#let czm = $attach(bold(c)_z^(upright(m)), tl: cal(B))$


#let MPCStateAll = $bold(upright(S))$
#let MPCControlAll = $bold(upright(U))$


#let BodyRateBC = $attach(bold(Omega)_b^(upright(c)), tl: cal(B))$
#let aTC = $a_T^(upright("c"))$


#let StateMeas = $bold(x)^(upright(m))$
#let BodyYMeas = $BodyY^(upright(m))$
#let PosMeas = $Pos^(upright(m))$
#let VelMeas = $Vel^(upright(m))$
#let AttitudeBMeas = $AttitudeB^(upright(m))$
#let BodyRateBMeas = $BodyRateB^(upright(m))$
#let AccPMeas = $AccP^(upright(m))$

#let czMeas = $cz^(upright(m))$
#let cxMeas = $cx^(upright(m))$
#let cyMeas = $cy^(upright(m))$

#let VMeas = $V^(upright(m))$


#let AccPBzMeas = $attach(bold(Acc)_(p_z)^(upright(m)), tl: cal(B))$
#let AccPBxMeas = $attach(bold(Acc)_(p_x)^(upright(m)), tl: cal(B))$


#let AccPBMeas = $attach(bold(Acc)_(p)^(upright(m)), tl: cal(B))$

#let AccPBxC = $attach(bold(Acc)_(p_x)^(upright(c)), tl: cal(B))$





//  Operations
#let oplus = $plus.circle$
#let oplusBig = $plus.circle.big$
