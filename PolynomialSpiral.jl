module PolynomialSpiral

using Constants, LatticeState, Utils, Geometry, JuMP, Ipopt

export SpiralControlAction, optimizeSpiral, sampleSpiral, getControlId, 
  transformControlActionEnd, transformControlActionFromState, 
  transformControlActionPoint, getPredecessorState, sampleSpiral, fbe, optimizeSpiral, getSpiralCoefficients

struct SpiralControlAction
  xf::Float64
  yf::Float64
  ti::Float64
  tf::Float64
  ki::Float64
  kf::Float64
  coefficients::Array{Float64}
  path::Array{Float64, 2}
  curvature_vals::Array{Float64}
  line_segment_count::Int64
  feasible::Bool
  arc_length::Float64
  bending_energy::Float64

  function SpiralControlAction(xf::Float64, yf::Float64, ti::Float64, tf::Float64,
    ki::Float64, kf::Float64, kmax::Float64=0.5)

    ti = wrapTo2Pi(ti)
    tf = wrapTo2Pi(tf)
    

    opt_result = optimizeSpiral(xf, yf, ti, tf, ki, kf, kmax)
    params = opt_result[1]
    if (opt_result[2] == :Optimal) || (opt_result[2] == :UserLimit)
      feasible = true
    else
      feasible = false
      @printf("Opt result = ")
      show(opt_result)
      @printf("\n")
    end

    # If the path is too long, it looped back onto itself, and is not relevant.
    straight_line_norm = norm([xf, yf])
    if params[5] > 1.5*straight_line_norm
      feasible = false
    end
  
    bending_energy::Float64 = fbe(params...)      
    coefficients = getSpiralCoefficients(params)
    path_vals = sampleSpiral(coefficients, ti)
    path = path_vals[1]
    line_segment_count = path_vals[2]
    curvature_vals = path_vals[3]

    s_f::Float64 = params[5]
    t_f::Float64 = fTheta(coefficients, s_f, ti)
    x_f::Float64 = xSimpson(params...)
    y_f::Float64 = ySimpson(params...)

    x_f_rotated::Float64 = x_f*cos(ti) - y_f*sin(ti)
    y_f_rotated::Float64 = x_f*sin(ti) + y_f*cos(ti)

    @assert(size(path, 1) - 1 == line_segment_count)

    if (abs(xf - x_f_rotated) > 0.01) || (abs(yf - y_f_rotated) > 0.01) || (abs(tf - t_f) > 0.01) || (abs(ti - path[1, 3]) > 0.01)
      feasible = false
    end

    new(xf, yf, wrapTo2Pi(ti), wrapTo2Pi(tf), ki, kf, coefficients, path, curvature_vals, line_segment_count, feasible, s_f, bending_energy)

  end

end

function optimizeSpiral(xf::Float64, yf::Float64, ti::Float64, tf::Float64, ki::Float64, kf::Float64, kmax::Float64=0.5)
  
  m = Model(solver=IpoptSolver(print_level=0, max_iter=3000))
  JuMP.register(m, :objective, 8, objective, objectiveGrad)

  # The optimization functions assume that the initial angle is zero.
  # Therefore, we need to rotate the boundary conditions such that this
  # assumption is true.
  xf_rotated::Float64 = xf*cos(-ti) - yf*sin(-ti)
  yf_rotated::Float64 = xf*sin(-ti) + yf*cos(-ti)
  tf_rotated::Float64 = tf - ti

  straight_line_norm = norm([xf, yf])

  # These variables correspond to the curvature
  # 1/3rd of the way through the curve, 2/3rds
  # of the way through the curve, and the final
  # arc length of the curve, respectively.
  @variable(m, -kmax <= p1 <= kmax)
  @variable(m, -kmax <= p2 <= kmax)
  @variable(m, straight_line_norm <= p4 <= 20.0)
  @variable(m, x == xf_rotated)
  @variable(m, y == yf_rotated)
  @variable(m, t == tf_rotated)
  @variable(m, p0 == ki)
  @variable(m, p3 == kf)
  @NLobjective(m, Min, objective(p0, p1, p2, p3, p4, x, y, t))

  status = solve(m)
  params::Array{Float64} = [getValue(p0), getvalue(p1), getvalue(p2), getValue(p3), getvalue(p4)]

  return (params, status)

end

function objective(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64, xf::Float64, yf::Float64, tf::Float64)

  cost = 25.0*fxf(p0, p1, p2, p3, p4, xf) + 25.0*fyf(p0, p1, p2, p3, p4, yf) + 30.0*ftf(p0, p1, p2, p3, p4, tf) + fbe(p0, p1, p2, p3, p4)

  return cost

end

# The square error in the final x position of the spiral.
# The x position is approximated using Simpson's rule with
# n = 8 intervals.
function fxf(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64, xf::Float64)::Float64
  t2::Float64 = p0*(1.1E1/2.0)
  t3::Float64 = p1*9.0
  t4::Float64 = p2*(9.0/2.0)
  t5::Float64 = p0*(9.0/2.0)
  t6::Float64 = p1*(2.7E1/2.0)
  t7::Float64 = p2*(2.7E1/2.0)
  t8::Float64 = p3*(9.0/2.0)
  t9::Float64 = t5-t6+t7-t8
  t10::Float64 = p0*9.0
  t11::Float64 = p1*(4.5E1/2.0)
  t12::Float64 = p2*1.8E1
  t13::Float64 = t8-t10+t11-t12
  t14::Float64 = p3-t2+t3-t4
  t15::Float64 = xf-p4*(cos(p0*p4-p4*t9*(1.0/4.0)-p4*t13*(1.0/3.0)+p4*t14*(1.0/2.0))+cos(p0*p4*(1.0/2.0)-p4*t9*(1.0/6.4E1)-p4*t13*(1.0/2.4E1)+p4*t14*(1.0/8.0))*2.0+cos(p0*p4*(3.0/4.0)-p4*t9*7.91015625E-2-p4*t13*(9.0/6.4E1)+p4*t14*(9.0/3.2E1))*2.0+cos(p0*p4*(1.0/4.0)-p4*t9*9.765625E-4-p4*t13*(1.0/1.92E2)+p4*t14*(1.0/3.2E1))*2.0+cos(p0*p4*(3.0/8.0)-p4*t9*4.94384765625E-3-p4*t13*(9.0/5.12E2)+p4*t14*(9.0/1.28E2))*4.0+cos(p0*p4*(1.0/8.0)-p4*t9*6.103515625E-5-p4*t13*6.510416666666667E-4+p4*t14*(1.0/1.28E2))*4.0+cos(p0*p4*(5.0/8.0)-p4*t9*3.814697265625E-2-p4*t13*8.138020833333333E-2+p4*t14*(2.5E1/1.28E2))*4.0+cos(p0*p4*(7.0/8.0)-p4*t9*1.4654541015625E-1-p4*t13*2.233072916666667E-1+p4*t14*(4.9E1/1.28E2))*4.0+1.0)*(1.0/2.4E1)
  t0::Float64 = t15*t15

  return t0

end

# The square error in the final y position of the spiral.
# The y position is approximated using Simpson's rule with
# n = 8 intervals.
function fyf(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64, yf::Float64)::Float64
  t2::Float64 = p0*(1.1E1/2.0)
  t3::Float64 = p1*9.0
  t4::Float64 = p2*(9.0/2.0)
  t5::Float64 = p0*(9.0/2.0)
  t6::Float64 = p1*(2.7E1/2.0)
  t7::Float64 = p2*(2.7E1/2.0)
  t8::Float64 = p3*(9.0/2.0)
  t9::Float64 = t5-t6+t7-t8
  t10::Float64 = p0*9.0
  t11::Float64 = p1*(4.5E1/2.0)
  t12::Float64 = p2*1.8E1
  t13::Float64 = t8-t10+t11-t12
  t14::Float64 = p3-t2+t3-t4
  t15::Float64 = yf-p4*(sin(p0*p4-p4*t9*(1.0/4.0)-p4*t13*(1.0/3.0)+p4*t14*(1.0/2.0))+sin(p0*p4*(1.0/2.0)-p4*t9*(1.0/6.4E1)-p4*t13*(1.0/2.4E1)+p4*t14*(1.0/8.0))*2.0+sin(p0*p4*(3.0/4.0)-p4*t9*7.91015625E-2-p4*t13*(9.0/6.4E1)+p4*t14*(9.0/3.2E1))*2.0+sin(p0*p4*(1.0/4.0)-p4*t9*9.765625E-4-p4*t13*(1.0/1.92E2)+p4*t14*(1.0/3.2E1))*2.0+sin(p0*p4*(3.0/8.0)-p4*t9*4.94384765625E-3-p4*t13*(9.0/5.12E2)+p4*t14*(9.0/1.28E2))*4.0+sin(p0*p4*(1.0/8.0)-p4*t9*6.103515625E-5-p4*t13*6.510416666666667E-4+p4*t14*(1.0/1.28E2))*4.0+sin(p0*p4*(5.0/8.0)-p4*t9*3.814697265625E-2-p4*t13*8.138020833333333E-2+p4*t14*(2.5E1/1.28E2))*4.0+sin(p0*p4*(7.0/8.0)-p4*t9*1.4654541015625E-1-p4*t13*2.233072916666667E-1+p4*t14*(4.9E1/1.28E2))*4.0)*(1.0/2.4E1)
  t0::Float64 = t15*t15

  return t0

end

# The square error in the final theta of the spiral.
# Evaluated analytically, since the curvature has a 
# closed form solution.
function ftf(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64, tf::Float64)::Float64
  t2::Float64 = tf-p0*p4+p4*(p0*(1.1E1/2.0)-p1*9.0+p2*(9.0/2.0)-p3)*(1.0/2.0)+p4*(p0*(9.0/2.0)-p1*(2.7E1/2.0)+p2*(2.7E1/2.0)-p3*(9.0/2.0))*(1.0/4.0)-p4*(p0*9.0-p1*(4.5E1/2.0)+p2*1.8E1-p3*(9.0/2.0))*(1.0/3.0)
  t0::Float64 = t2*t2

  return t0

end
 
# The bending energy of the spiral. 
# Evaluated analytically, since the curvature has a
# closed form solution.
function fbe(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64)::Float64
  t0::Float64 = p4*(p0*p1*9.9E1-p0*p2*3.6E1+p0*p3*1.9E1-p1*p2*8.1E1-p1*p3*3.6E1+p2*p3*9.9E1+(p0*p0)*6.4E1+(p1*p1)*3.24E2+(p2*p2)*3.24E2+(p3*p3)*6.4E1)*(1.0/8.4E2)

  return t0

end

# Gives the Simpson's rule approximation to the final x position
# of the spiral.
function xSimpson(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64)::Float64
  t2::Float64 = p0*(1.1E1/2.0);
  t3::Float64 = p1*9.0;
  t4::Float64 = p2*(9.0/2.0);
  t5::Float64 = p0*(9.0/2.0);
  t6::Float64 = p1*(2.7E1/2.0);
  t7::Float64 = p2*(2.7E1/2.0);
  t8::Float64 = p3*(9.0/2.0);
  t9::Float64 = t5-t6+t7-t8;
  t10::Float64 = p0*9.0;
  t11::Float64 = p1*(4.5E1/2.0);
  t12::Float64 = p2*1.8E1;
  t13::Float64 = t8-t10+t11-t12;
  t0::Float64 = p4*(cos(p4*(p3*(9.0/2.0)-t10+t11-t12)*(1.0/3.0)-p0*p4+p4*t9*(1.0/4.0)-p4*(p3-t2+t3-t4)*(1.0/2.0))+cos(p0*p4*(1.0/2.0)-p4*t9*(1.0/6.4E1)-p4*t13*(1.0/2.4E1)+p4*(p3-t2+t3-t4)*(1.0/8.0))*2.0+cos(p0*p4*(3.0/4.0)-p4*t9*7.91015625E-2-p4*t13*(9.0/6.4E1)+p4*(p3-t2+t3-t4)*(9.0/3.2E1))*2.0+cos(p0*p4*(1.0/4.0)-p4*t9*9.765625E-4-p4*t13*(1.0/1.92E2)+p4*(p3-t2+t3-t4)*(1.0/3.2E1))*2.0+cos(p0*p4*(3.0/8.0)-p4*t9*4.94384765625E-3-p4*t13*(9.0/5.12E2)+p4*(p3-t2+t3-t4)*(9.0/1.28E2))*4.0+cos(p0*p4*(1.0/8.0)-p4*t9*6.103515625E-5-p4*t13*6.510416666666667E-4+p4*(p3-t2+t3-t4)*(1.0/1.28E2))*4.0+cos(p0*p4*(5.0/8.0)-p4*t9*3.814697265625E-2-p4*t13*8.138020833333333E-2+p4*(p3-t2+t3-t4)*(2.5E1/1.28E2))*4.0+cos(p0*p4*(7.0/8.0)-p4*t9*1.4654541015625E-1-p4*t13*2.233072916666667E-1+p4*(p3-t2+t3-t4)*(4.9E1/1.28E2))*4.0+1.0)*(1.0/2.4E1);

  return t0

end

function ySimpson(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64)::Float64
  t2::Float64 = p0*(1.1E1/2.0);
  t3::Float64 = p1*9.0;
  t4::Float64 = p2*(9.0/2.0);
  t5::Float64 = p0*(9.0/2.0);
  t6::Float64 = p1*(2.7E1/2.0);
  t7::Float64 = p2*(2.7E1/2.0);
  t8::Float64 = p3*(9.0/2.0);
  t9::Float64 = t5-t6+t7-t8;
  t10::Float64 = p0*9.0;
  t11::Float64 = p1*(4.5E1/2.0);
  t12::Float64 = p2*1.8E1;
  t13::Float64 = t8-t10+t11-t12;
  t0::Float64 = p4*(-sin(p4*(p3*(9.0/2.0)-t10+t11-t12)*(1.0/3.0)-p0*p4+p4*t9*(1.0/4.0)-p4*(p3-t2+t3-t4)*(1.0/2.0))+sin(p0*p4*(1.0/2.0)-p4*t9*(1.0/6.4E1)-p4*t13*(1.0/2.4E1)+p4*(p3-t2+t3-t4)*(1.0/8.0))*2.0+sin(p0*p4*(3.0/4.0)-p4*t9*7.91015625E-2-p4*t13*(9.0/6.4E1)+p4*(p3-t2+t3-t4)*(9.0/3.2E1))*2.0+sin(p0*p4*(1.0/4.0)-p4*t9*9.765625E-4-p4*t13*(1.0/1.92E2)+p4*(p3-t2+t3-t4)*(1.0/3.2E1))*2.0+sin(p0*p4*(3.0/8.0)-p4*t9*4.94384765625E-3-p4*t13*(9.0/5.12E2)+p4*(p3-t2+t3-t4)*(9.0/1.28E2))*4.0+sin(p0*p4*(1.0/8.0)-p4*t9*6.103515625E-5-p4*t13*6.510416666666667E-4+p4*(p3-t2+t3-t4)*(1.0/1.28E2))*4.0+sin(p0*p4*(5.0/8.0)-p4*t9*3.814697265625E-2-p4*t13*8.138020833333333E-2+p4*(p3-t2+t3-t4)*(2.5E1/1.28E2))*4.0+sin(p0*p4*(7.0/8.0)-p4*t9*1.4654541015625E-1-p4*t13*2.233072916666667E-1+p4*(p3-t2+t3-t4)*(4.9E1/1.28E2))*4.0)*(1.0/2.4E1);

  return t0

end

# Calculates the gradient of the objective for the optimizer.
function objectiveGrad(g, p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64, xf::Float64, yf::Float64, tf::Float64)
  # p0, the initial curvature, is constant.
  g[1] = 0.0
  g[2] = 25.0*dxdp1(p0, p1, p2, p3, p4, xf) + 25.0*dydp1(p0, p1, p2, p3, p4, yf) + 30.0*dtdp1(p0, p1, p2, p3, p4, tf) + dbedp1(p0, p1, p2, p3, p4)
  g[3] = 25.0*dxdp2(p0, p1, p2, p3, p4, xf) + 25.0*dydp2(p0, p1, p2, p3, p4, yf) + 30.0*dtdp2(p0, p1, p2, p3, p4, tf) + dbedp2(p0, p1, p2, p3, p4)
  # p3, the final curvature, is constant.
  g[4] = 0.0
  g[5] = 25.0*dxdp4(p0, p1, p2, p3, p4, xf) + 25.0*dydp4(p0, p1, p2, p3, p4, yf) + 30.0*dtdp4(p0, p1, p2, p3, p4, tf) + dbedp4(p0, p1, p2, p3, p4)
  # The final 3 terms are constants (from the boundary conditions).
  g[6] = 0.0
  g[7] = 0.0
  g[8] = 0.0

end

# Gradient of fxf with respect to p1.
function dxdp1(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64, xf::Float64)::Float64
  t2::Float64 = p0*(1.1E1/2.0);
  t3::Float64 = p1*9.0;
  t4::Float64 = p2*(9.0/2.0);
  t5::Float64 = p0*(9.0/2.0);
  t6::Float64 = p1*(2.7E1/2.0);
  t7::Float64 = p2*(2.7E1/2.0);
  t8::Float64 = p3*(9.0/2.0);
  t9::Float64 = t5-t6+t7-t8;
  t10::Float64 = p0*9.0;
  t11::Float64 = p1*(4.5E1/2.0);
  t12::Float64 = p2*1.8E1;
  t13::Float64 = t8-t10+t11-t12;
  t14::Float64 = p3-t2+t3-t4;
  t15::Float64 = p0*p4;
  t16::Float64 = p0*p4*(1.0/2.0);
  t17::Float64 = p0*p4*(3.0/4.0);
  t18::Float64 = p0*p4*(1.0/4.0);
  t19::Float64 = p0*p4*(3.0/8.0);
  t20::Float64 = p0*p4*(1.0/8.0);
  t21::Float64 = p0*p4*(5.0/8.0);
  t22::Float64 = p0*p4*(7.0/8.0);
  t0::Float64 = p4*(xf-p4*(cos(t15-p4*t9*(1.0/4.0)-p4*t13*(1.0/3.0)+p4*t14*(1.0/2.0))+cos(t16-p4*t9*(1.0/6.4E1)-p4*t13*(1.0/2.4E1)+p4*t14*(1.0/8.0))*2.0+cos(t17-p4*t9*7.91015625E-2-p4*t13*(9.0/6.4E1)+p4*t14*(9.0/3.2E1))*2.0+cos(t18-p4*t9*9.765625E-4-p4*t13*(1.0/1.92E2)+p4*t14*(1.0/3.2E1))*2.0+cos(t19-p4*t9*4.94384765625E-3-p4*t13*(9.0/5.12E2)+p4*t14*(9.0/1.28E2))*4.0+cos(t20-p4*t9*6.103515625E-5-p4*t13*6.510416666666667E-4+p4*t14*(1.0/1.28E2))*4.0+cos(t21-p4*t9*3.814697265625E-2-p4*t13*8.138020833333333E-2+p4*t14*(2.5E1/1.28E2))*4.0+cos(t22-p4*t9*1.4654541015625E-1-p4*t13*2.233072916666667E-1+p4*t14*(4.9E1/1.28E2))*4.0+1.0)*(1.0/2.4E1))*(p4*sin(t15-p4*t9*(1.0/4.0)-p4*t13*(1.0/3.0)+p4*(p3-t2+t3-t4)*(1.0/2.0))*(3.0/8.0)+p4*sin(t16-p4*t9*(1.0/6.4E1)-p4*t13*(1.0/2.4E1)+p4*(p3-t2+t3-t4)*(1.0/8.0))*(5.1E1/6.4E1)+p4*sin(t17-p4*t9*7.91015625E-2-p4*t13*(9.0/6.4E1)+p4*(p3-t2+t3-t4)*(9.0/3.2E1))*8.701171875E-1+p4*sin(t18-p4*t9*9.765625E-4-p4*t13*(1.0/1.92E2)+p4*(p3-t2+t3-t4)*(1.0/3.2E1))*3.544921875E-1+p4*sin(t19-p4*t9*4.94384765625E-3-p4*t13*(9.0/5.12E2)+p4*(p3-t2+t3-t4)*(9.0/1.28E2))*1.2161865234375+p4*sin(t20-p4*t9*6.103515625E-5-p4*t13*6.510416666666667E-4+p4*(p3-t2+t3-t4)*(1.0/1.28E2))*2.259521484375E-1+p4*sin(t21-p4*t9*3.814697265625E-2-p4*t13*8.138020833333333E-2+p4*(p3-t2+t3-t4)*(2.5E1/1.28E2))*1.7669677734375+p4*sin(t22-p4*t9*1.4654541015625E-1-p4*t13*2.233072916666667E-1+p4*(p3-t2+t3-t4)*(4.9E1/1.28E2))*1.5970458984375)*(1.0/1.2E1);

  return t0

end

# Gradient of fxf with respect to p2.
function dxdp2(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64, xf::Float64)::Float64
  t2::Float64 = p0*(1.1E1/2.0);
  t3::Float64 = p1*9.0;
  t4::Float64 = p2*(9.0/2.0);
  t5::Float64 = p0*(9.0/2.0);
  t6::Float64 = p1*(2.7E1/2.0);
  t7::Float64 = p2*(2.7E1/2.0);
  t8::Float64 = p3*(9.0/2.0);
  t9::Float64 = t5-t6+t7-t8;
  t10::Float64 = p0*9.0;
  t11::Float64 = p1*(4.5E1/2.0);
  t12::Float64 = p2*1.8E1;
  t13::Float64 = t8-t10+t11-t12;
  t14::Float64 = p3-t2+t3-t4;
  t15::Float64 = p0*p4;
  t16::Float64 = p0*p4*(1.0/2.0);
  t17::Float64 = p4*t14*(1.0/8.0);
  t18::Float64 = t16+t17-p4*t9*(1.0/6.4E1)-p4*t13*(1.0/2.4E1);
  t19::Float64 = p0*p4*(3.0/4.0);
  t20::Float64 = p0*p4*(1.0/4.0);
  t21::Float64 = p4*t14*(1.0/3.2E1);
  t22::Float64 = t20+t21-p4*t9*9.765625E-4-p4*t13*(1.0/1.92E2);
  t23::Float64 = p0*p4*(3.0/8.0);
  t24::Float64 = p4*t14*(9.0/1.28E2);
  t25::Float64 = t23+t24-p4*t9*4.94384765625E-3-p4*t13*(9.0/5.12E2);
  t26::Float64 = p0*p4*(1.0/8.0);
  t27::Float64 = p4*t14*(1.0/1.28E2);
  t28::Float64 = t26+t27-p4*t9*6.103515625E-5-p4*t13*6.510416666666667E-4;
  t29::Float64 = p0*p4*(5.0/8.0);
  t30::Float64 = p0*p4*(7.0/8.0);
  t0::Float64 = p4*(xf-p4*(cos(t15-p4*t9*(1.0/4.0)-p4*t13*(1.0/3.0)+p4*t14*(1.0/2.0))+cos(t19-p4*t9*7.91015625E-2-p4*t13*(9.0/6.4E1)+p4*t14*(9.0/3.2E1))*2.0+cos(t29-p4*t9*3.814697265625E-2-p4*t13*8.138020833333333E-2+p4*t14*(2.5E1/1.28E2))*4.0+cos(t30-p4*t9*1.4654541015625E-1-p4*t13*2.233072916666667E-1+p4*t14*(4.9E1/1.28E2))*4.0+cos(t18)*2.0+cos(t22)*2.0+cos(t25)*4.0+cos(t28)*4.0+1.0)*(1.0/2.4E1))*(p4*sin(t15-p4*t9*(1.0/4.0)-p4*t13*(1.0/3.0)+p4*(p3-t2+t3-t4)*(1.0/2.0))*(3.0/8.0)+p4*sin(t19-p4*t9*7.91015625E-2-p4*t13*(9.0/6.4E1)+p4*(p3-t2+t3-t4)*(9.0/3.2E1))*3.955078125E-1+p4*sin(t29-p4*t9*3.814697265625E-2-p4*t13*8.138020833333333E-2+p4*(p3-t2+t3-t4)*(2.5E1/1.28E2))*2.838134765625E-1+p4*sin(t30-p4*t9*1.4654541015625E-1-p4*t13*2.233072916666667E-1+p4*(p3-t2+t3-t4)*(4.9E1/1.28E2))*1.2740478515625-p4*sin(t18)*(3.0/6.4E1)-p4*sin(t22)*1.201171875E-1-p4*sin(t25)*2.669677734375E-1-p4*sin(t28)*9.70458984375E-2)*(1.0/1.2E1);

  return t0

end

# Gradient of fxf with respect to p4.
function dxdp4(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64, xf::Float64)::Float64
  t2::Float64 = p0*(1.1E1/2.0);
  t3::Float64 = p1*9.0;
  t4::Float64 = p2*(9.0/2.0);
  t5::Float64 = p0*(9.0/2.0);
  t6::Float64 = p1*(2.7E1/2.0);
  t7::Float64 = p2*(2.7E1/2.0);
  t8::Float64 = p3*(9.0/2.0);
  t9::Float64 = t5-t6+t7-t8;
  t10::Float64 = p0*9.0;
  t11::Float64 = p1*(4.5E1/2.0);
  t12::Float64 = p2*1.8E1;
  t13::Float64 = t8-t10+t11-t12;
  t14::Float64 = p3-t2+t3-t4;
  t15::Float64 = p0*p4;
  t16::Float64 = p0*p4*(1.0/2.0);
  t17::Float64 = p0*p4*(3.0/4.0);
  t18::Float64 = p0*p4*(1.0/4.0);
  t19::Float64 = p0*p4*(3.0/8.0);
  t20::Float64 = p0*p4*(1.0/8.0);
  t21::Float64 = p0*p4*(5.0/8.0);
  t22::Float64 = p0*p4*(7.0/8.0);
  t23::Float64 = p4*(p3-t2+t3-t4)*(1.0/2.0);
  t39::Float64 = p4*t9*(1.0/4.0);
  t40::Float64 = p4*t13*(1.0/3.0);
  t24::Float64 = t15+t23-t39-t40;
  t25::Float64 = p4*(p3-t2+t3-t4)*(1.0/8.0);
  t41::Float64 = p4*t9*(1.0/6.4E1);
  t42::Float64 = p4*t13*(1.0/2.4E1);
  t26::Float64 = t16+t25-t41-t42;
  t27::Float64 = p4*(p3-t2+t3-t4)*(1.0/3.2E1);
  t45::Float64 = p4*t9*9.765625E-4;
  t46::Float64 = p4*t13*(1.0/1.92E2);
  t28::Float64 = t18+t27-t45-t46;
  t29::Float64 = p4*(p3-t2+t3-t4)*(9.0/3.2E1);
  t43::Float64 = p4*t9*7.91015625E-2;
  t44::Float64 = p4*t13*(9.0/6.4E1);
  t30::Float64 = t17+t29-t43-t44;
  t31::Float64 = p4*(p3-t2+t3-t4)*(1.0/1.28E2);
  t49::Float64 = p4*t9*6.103515625E-5;
  t50::Float64 = p4*t13*6.510416666666667E-4;
  t32::Float64 = t20+t31-t49-t50;
  t33::Float64 = p4*(p3-t2+t3-t4)*(9.0/1.28E2);
  t47::Float64 = p4*t9*4.94384765625E-3;
  t48::Float64 = p4*t13*(9.0/5.12E2);
  t34::Float64 = t19+t33-t47-t48;
  t35::Float64 = p4*(p3-t2+t3-t4)*(2.5E1/1.28E2);
  t51::Float64 = p4*t9*3.814697265625E-2;
  t52::Float64 = p4*t13*8.138020833333333E-2;
  t36::Float64 = t21+t35-t51-t52;
  t37::Float64 = p4*(p3-t2+t3-t4)*(4.9E1/1.28E2);
  t53::Float64 = p4*t9*1.4654541015625E-1;
  t54::Float64 = p4*t13*2.233072916666667E-1;
  t38::Float64 = t22+t37-t53-t54;
  t0::Float64 = (xf-p4*(cos(t15-t39-t40+p4*t14*(1.0/2.0))+cos(t16-t41-t42+p4*t14*(1.0/8.0))*2.0+cos(t18-t45-t46+p4*t14*(1.0/3.2E1))*2.0+cos(t17-t43-t44+p4*t14*(9.0/3.2E1))*2.0+cos(t20-t49-t50+p4*t14*(1.0/1.28E2))*4.0+cos(t19-t47-t48+p4*t14*(9.0/1.28E2))*4.0+cos(t21-t51-t52+p4*t14*(2.5E1/1.28E2))*4.0+cos(t22-t53-t54+p4*t14*(4.9E1/1.28E2))*4.0+1.0)*(1.0/2.4E1))*(cos(t24)*(1.0/2.4E1)+cos(t26)*(1.0/1.2E1)+cos(t28)*(1.0/1.2E1)+cos(t30)*(1.0/1.2E1)+cos(t32)*(1.0/6.0)+cos(t34)*(1.0/6.0)+cos(t36)*(1.0/6.0)+cos(t38)*(1.0/6.0)-p4*(sin(t24)*(p0*(1.0/8.0)+p1*(3.0/8.0)+p2*(3.0/8.0)+p3*(1.0/8.0))+sin(t26)*(p0*(1.5E1/1.28E2)+p1*(5.1E1/1.28E2)-p2*(3.0/1.28E2)+p3*(1.0/1.28E2))*2.0+sin(t28)*(p0*1.2060546875E-1+p1*1.7724609375E-1-p2*6.005859375E-2+p3*1.220703125E-2)*2.0+sin(t30)*(p0*1.1279296875E-1+p1*4.3505859375E-1+p2*1.9775390625E-1+p3*4.39453125E-3)*2.0+sin(t32)*(p0*8.7615966796875E-2+p1*5.6488037109375E-2-p2*2.4261474609375E-2+p3*5.157470703125E-3)*4.0+sin(t34)*(p0*1.24237060546875E-1+p1*3.04046630859375E-1-p2*6.6741943359375E-2+p3*1.3458251953125E-2)*4.0+sin(t36)*(p0*1.11541748046875E-1+p1*4.41741943359375E-1+p2*7.0953369140625E-2+p3*7.62939453125E-4)*4.0+sin(t38)*(p0*1.19842529296875E-1+p1*3.99261474609375E-1+p2*3.18511962890625E-1+p3*3.7384033203125E-2)*4.0)*(1.0/2.4E1)+1.0/2.4E1)*-2.0;

  return t0

end

# Gradient of fyf with respect to p1.
function dydp1(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64, yf::Float64)::Float64
  t2::Float64= p0*(1.1E1/2.0);
  t3::Float64= p1*9.0;
  t4::Float64= p2*(9.0/2.0);
  t5::Float64= p0*(9.0/2.0);
  t6::Float64= p1*(2.7E1/2.0);
  t7::Float64= p2*(2.7E1/2.0);
  t8::Float64= p3*(9.0/2.0);
  t9::Float64= t5-t6+t7-t8;
  t10::Float64 = p0*9.0;
  t11::Float64 = p1*(4.5E1/2.0);
  t12::Float64 = p2*1.8E1;
  t13::Float64 = t8-t10+t11-t12;
  t14::Float64 = p3-t2+t3-t4;
  t15::Float64 = p0*p4;
  t16::Float64 = p0*p4*(1.0/2.0);
  t17::Float64 = p0*p4*(3.0/4.0);
  t18::Float64 = p0*p4*(1.0/4.0);
  t19::Float64 = p0*p4*(3.0/8.0);
  t20::Float64 = p0*p4*(1.0/8.0);
  t21::Float64 = p0*p4*(5.0/8.0);
  t22::Float64 = p0*p4*(7.0/8.0);
  t23::Float64 = p4*t14*(1.0/2.0);
  t24::Float64 = t15+t23-p4*t9*(1.0/4.0)-p4*t13*(1.0/3.0);
  t25::Float64 = p4*t14*(1.0/8.0);
  t26::Float64 = t16+t25-p4*t9*(1.0/6.4E1)-p4*t13*(1.0/2.4E1);
  t27::Float64 = p4*t14*(9.0/3.2E1);
  t28::Float64 = t17+t27-p4*t9*7.91015625E-2-p4*t13*(9.0/6.4E1);
  t29::Float64 = p4*t14*(1.0/3.2E1);
  t30::Float64 = t18+t29-p4*t9*9.765625E-4-p4*t13*(1.0/1.92E2);
  t31::Float64 = p4*t14*(9.0/1.28E2);
  t32::Float64 = t19+t31-p4*t9*4.94384765625E-3-p4*t13*(9.0/5.12E2);
  t33::Float64 = p4*t14*(1.0/1.28E2);
  t34::Float64 = t20+t33-p4*t9*6.103515625E-5-p4*t13*6.510416666666667E-4;
  t35::Float64 = p4*t14*(2.5E1/1.28E2);
  t36::Float64 = t21+t35-p4*t9*3.814697265625E-2-p4*t13*8.138020833333333E-2;
  t37::Float64 = p4*t14*(4.9E1/1.28E2);
  t38::Float64 = t22+t37-p4*t9*1.4654541015625E-1-p4*t13*2.233072916666667E-1;
  t0::Float64 = p4*(yf-p4*(sin(t24)+sin(t26)*2.0+sin(t28)*2.0+sin(t30)*2.0+sin(t32)*4.0+sin(t34)*4.0+sin(t36)*4.0+sin(t38)*4.0)*(1.0/2.4E1))*(p4*cos(t24)*(3.0/8.0)+p4*cos(t26)*(5.1E1/6.4E1)+p4*cos(t28)*8.701171875E-1+p4*cos(t30)*3.544921875E-1+p4*cos(t32)*1.2161865234375+p4*cos(t34)*2.259521484375E-1+p4*cos(t36)*1.7669677734375+p4*cos(t38)*1.5970458984375)*(-1.0/1.2E1);

  return t0

end

# Gradient of fyf with respect to p2.
function dydp2(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64, yf::Float64)::Float64
  t2::Float64 = p0*(1.1E1/2.0);
  t3::Float64 = p1*9.0;
  t4::Float64 = p2*(9.0/2.0);
  t5::Float64 = p0*(9.0/2.0);
  t6::Float64 = p1*(2.7E1/2.0);
  t7::Float64 = p2*(2.7E1/2.0);
  t8::Float64 = p3*(9.0/2.0);
  t9::Float64 = t5-t6+t7-t8;
  t10::Float64 = p0*9.0;
  t11::Float64 = p1*(4.5E1/2.0);
  t12::Float64 = p2*1.8E1;
  t13::Float64 = t8-t10+t11-t12;
  t14::Float64 = p3-t2+t3-t4;
  t15::Float64 = p0*p4;
  t16::Float64 = p0*p4*(1.0/2.0);
  t17::Float64 = p4*t14*(1.0/8.0);
  t18::Float64 = t16+t17-p4*t9*(1.0/6.4E1)-p4*t13*(1.0/2.4E1);
  t19::Float64 = p0*p4*(3.0/4.0);
  t20::Float64 = p0*p4*(1.0/4.0);
  t21::Float64 = p4*t14*(1.0/3.2E1);
  t22::Float64 = t20+t21-p4*t9*9.765625E-4-p4*t13*(1.0/1.92E2);
  t23::Float64 = p0*p4*(3.0/8.0);
  t24::Float64 = p4*t14*(9.0/1.28E2);
  t25::Float64 = t23+t24-p4*t9*4.94384765625E-3-p4*t13*(9.0/5.12E2);
  t26::Float64 = p0*p4*(1.0/8.0);
  t27::Float64 = p4*t14*(1.0/1.28E2);
  t28::Float64 = t26+t27-p4*t9*6.103515625E-5-p4*t13*6.510416666666667E-4;
  t29::Float64 = p0*p4*(5.0/8.0);
  t30::Float64 = p0*p4*(7.0/8.0);
  t31::Float64 = p4*t14*(1.0/2.0);
  t32::Float64 = t15+t31-p4*t9*(1.0/4.0)-p4*t13*(1.0/3.0);
  t33::Float64 = p4*t14*(9.0/3.2E1);
  t34::Float64 = t19+t33-p4*t9*7.91015625E-2-p4*t13*(9.0/6.4E1);
  t35::Float64 = p4*t14*(2.5E1/1.28E2);
  t36::Float64 = t29+t35-p4*t9*3.814697265625E-2-p4*t13*8.138020833333333E-2;
  t37::Float64 = p4*t14*(4.9E1/1.28E2);
  t38::Float64 = t30+t37-p4*t9*1.4654541015625E-1-p4*t13*2.233072916666667E-1;
  t0::Float64 = p4*(yf-p4*(sin(t18)*2.0+sin(t22)*2.0+sin(t25)*4.0+sin(t28)*4.0+sin(t32)+sin(t34)*2.0+sin(t36)*4.0+sin(t38)*4.0)*(1.0/2.4E1))*(p4*cos(t18)*(3.0/6.4E1)+p4*cos(t22)*1.201171875E-1+p4*cos(t25)*2.669677734375E-1+p4*cos(t28)*9.70458984375E-2-p4*cos(t32)*(3.0/8.0)-p4*cos(t34)*3.955078125E-1-p4*cos(t36)*2.838134765625E-1-p4*cos(t38)*1.2740478515625)*(1.0/1.2E1);

  return t0

end

# Gradient of fyf with respect to p4.
function dydp4(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64, yf::Float64)::Float64
  t2 ::Float64 = p0*(1.1E1/2.0);
  t3 ::Float64 = p1*9.0;
  t4 ::Float64 = p2*(9.0/2.0);
  t5 ::Float64 = p0*(9.0/2.0);
  t6 ::Float64 = p1*(2.7E1/2.0);
  t7 ::Float64 = p2*(2.7E1/2.0);
  t8 ::Float64 = p3*(9.0/2.0);
  t9 ::Float64 = t5-t6+t7-t8;
  t10::Float64 = p0*9.0;
  t11::Float64 = p1*(4.5E1/2.0);
  t12::Float64 = p2*1.8E1;
  t13::Float64 = t8-t10+t11-t12;
  t14::Float64 = p3-t2+t3-t4;
  t15::Float64 = p0*p4;
  t16::Float64 = p0*p4*(1.0/2.0);
  t17::Float64 = p0*p4*(3.0/4.0);
  t18::Float64 = p0*p4*(1.0/4.0);
  t19::Float64 = p0*p4*(3.0/8.0);
  t20::Float64 = p0*p4*(1.0/8.0);
  t21::Float64 = p0*p4*(5.0/8.0);
  t22::Float64 = p0*p4*(7.0/8.0);
  t23::Float64 = p4*(p3-t2+t3-t4)*(1.0/2.0);
  t39::Float64 = p4*t9*(1.0/4.0);
  t40::Float64 = p4*t13*(1.0/3.0);
  t24::Float64 = t15+t23-t39-t40;
  t25::Float64 = p4*(p3-t2+t3-t4)*(1.0/8.0);
  t41::Float64 = p4*t9*(1.0/6.4E1);
  t42::Float64 = p4*t13*(1.0/2.4E1);
  t26::Float64 = t16+t25-t41-t42;
  t27::Float64 = p4*(p3-t2+t3-t4)*(1.0/3.2E1);
  t45::Float64 = p4*t9*9.765625E-4;
  t46::Float64 = p4*t13*(1.0/1.92E2);
  t28::Float64 = t18+t27-t45-t46;
  t29::Float64 = p4*(p3-t2+t3-t4)*(9.0/3.2E1);
  t43::Float64 = p4*t9*7.91015625E-2;
  t44::Float64 = p4*t13*(9.0/6.4E1);
  t30::Float64 = t17+t29-t43-t44;
  t31::Float64 = p4*(p3-t2+t3-t4)*(1.0/1.28E2);
  t49::Float64 = p4*t9*6.103515625E-5;
  t50::Float64 = p4*t13*6.510416666666667E-4;
  t32::Float64 = t20+t31-t49-t50;
  t33::Float64 = p4*(p3-t2+t3-t4)*(9.0/1.28E2);
  t47::Float64 = p4*t9*4.94384765625E-3;
  t48::Float64 = p4*t13*(9.0/5.12E2);
  t34::Float64 = t19+t33-t47-t48;
  t35::Float64 = p4*(p3-t2+t3-t4)*(2.5E1/1.28E2);
  t51::Float64 = p4*t9*3.814697265625E-2;
  t52::Float64 = p4*t13*8.138020833333333E-2;
  t36::Float64 = t21+t35-t51-t52;
  t37::Float64 = p4*(p3-t2+t3-t4)*(4.9E1/1.28E2);
  t53::Float64 = p4*t9*1.4654541015625E-1;
  t54::Float64 = p4*t13*2.233072916666667E-1;
  t38::Float64 = t22+t37-t53-t54;
  t0::Float64 = (yf-p4*(sin(t15-t39-t40+p4*t14*(1.0/2.0))+sin(t16-t41-t42+p4*t14*(1.0/8.0))*2.0+sin(t18-t45-t46+p4*t14*(1.0/3.2E1))*2.0+sin(t17-t43-t44+p4*t14*(9.0/3.2E1))*2.0+sin(t20-t49-t50+p4*t14*(1.0/1.28E2))*4.0+sin(t19-t47-t48+p4*t14*(9.0/1.28E2))*4.0+sin(t21-t51-t52+p4*t14*(2.5E1/1.28E2))*4.0+sin(t22-t53-t54+p4*t14*(4.9E1/1.28E2))*4.0)*(1.0/2.4E1))*(sin(t24)*(1.0/2.4E1)+sin(t26)*(1.0/1.2E1)+sin(t28)*(1.0/1.2E1)+sin(t30)*(1.0/1.2E1)+sin(t32)*(1.0/6.0)+sin(t34)*(1.0/6.0)+sin(t36)*(1.0/6.0)+sin(t38)*(1.0/6.0)+p4*(cos(t24)*(p0*(1.0/8.0)+p1*(3.0/8.0)+p2*(3.0/8.0)+p3*(1.0/8.0))+cos(t26)*(p0*(1.5E1/1.28E2)+p1*(5.1E1/1.28E2)-p2*(3.0/1.28E2)+p3*(1.0/1.28E2))*2.0+cos(t28)*(p0*1.2060546875E-1+p1*1.7724609375E-1-p2*6.005859375E-2+p3*1.220703125E-2)*2.0+cos(t30)*(p0*1.1279296875E-1+p1*4.3505859375E-1+p2*1.9775390625E-1+p3*4.39453125E-3)*2.0+cos(t32)*(p0*8.7615966796875E-2+p1*5.6488037109375E-2-p2*2.4261474609375E-2+p3*5.157470703125E-3)*4.0+cos(t34)*(p0*1.24237060546875E-1+p1*3.04046630859375E-1-p2*6.6741943359375E-2+p3*1.3458251953125E-2)*4.0+cos(t36)*(p0*1.11541748046875E-1+p1*4.41741943359375E-1+p2*7.0953369140625E-2+p3*7.62939453125E-4)*4.0+cos(t38)*(p0*1.19842529296875E-1+p1*3.99261474609375E-1+p2*3.18511962890625E-1+p3*3.7384033203125E-2)*4.0)*(1.0/2.4E1))*-2.0;

  return t0

end

# Gradient of ftf with respect to p1.
function dtdp1(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64, tf::Float64)::Float64
  t0::Float64 = p4*(tf-p0*p4+p4*(p0*(1.1E1/2.0)-p1*9.0+p2*(9.0/2.0)-p3)*(1.0/2.0)+p4*(p0*(9.0/2.0)-p1*(2.7E1/2.0)+p2*(2.7E1/2.0)-p3*(9.0/2.0))*(1.0/4.0)-p4*(p0*9.0-p1*(4.5E1/2.0)+p2*1.8E1-p3*(9.0/2.0))*(1.0/3.0))*(-3.0/4.0);

  return t0

end

# Gradient of ftf with respect to p2.
function dtdp2(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64, tf::Float64)::Float64 
  t0::Float64 = p4*(tf-p0*p4+p4*(p0*(1.1E1/2.0)-p1*9.0+p2*(9.0/2.0)-p3)*(1.0/2.0)+p4*(p0*(9.0/2.0)-p1*(2.7E1/2.0)+p2*(2.7E1/2.0)-p3*(9.0/2.0))*(1.0/4.0)-p4*(p0*9.0-p1*(4.5E1/2.0)+p2*1.8E1-p3*(9.0/2.0))*(1.0/3.0))*(-3.0/4.0);

  return t0

end

# Gradient of ftf with respect to p4.
function dtdp4(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64, tf::Float64)::Float64
  t0::Float64 = (p0*(1.0/8.0)+p1*(3.0/8.0)+p2*(3.0/8.0)+p3*(1.0/8.0))*(tf-p0*p4+p4*(p0*(1.1E1/2.0)-p1*9.0+p2*(9.0/2.0)-p3)*(1.0/2.0)+p4*(p0*(9.0/2.0)-p1*(2.7E1/2.0)+p2*(2.7E1/2.0)-p3*(9.0/2.0))*(1.0/4.0)-p4*(p0*9.0-p1*(4.5E1/2.0)+p2*1.8E1-p3*(9.0/2.0))*(1.0/3.0))*-2.0;

  return t0

end

# Gradient of fbe with respect to p1.
function dbedp1(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64)::Float64
  t0::Float64 = p4*(p0*9.9E1+p1*6.48E2-p2*8.1E1-p3*3.6E1)*(1.0/8.4E2);

  return t0

end

# Gradient of fbe with respect to p2.
function dbedp2(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64)::Float64
  t0::Float64 = p4*(p0*3.6E1+p1*8.1E1-p2*6.48E2-p3*9.9E1)*(-1.0/8.4E2);

  return t0

end

# Gradient of fbe with respect to p4.
function dbedp4(p0::Float64, p1::Float64, p2::Float64, p3::Float64, p4::Float64)::Float64
  t0::Float64 = p0*p1*(3.3E1/2.8E2)-p0*p2*(3.0/7.0E1)+p0*p3*(1.9E1/8.4E2)-p1*p2*(2.7E1/2.8E2)-p1*p3*(3.0/7.0E1)+p2*p3*(3.3E1/2.8E2)+(p0*p0)*(8.0/1.05E2)+(p1*p1)*(2.7E1/7.0E1)+(p2*p2)*(2.7E1/7.0E1)+(p3*p3)*(8.0/1.05E2);

  return t0

end

function getSpiralCoefficients(params::Array{Float64})::Array{Float64}
  coeff::Array{Float64} = Array{Float64}(5)

  coeff[1] = params[1]
  coeff[2] = (18 * params[2] + 2 * params[4] - 11 * params[1] - 9 * params[3]) / (2.0 * params[5])
  coeff[3] = (18 * params[1] - 45 * params[2] + 36 * params[3] - 9 * params[4]) / (2.0 * params[5]^2)
  coeff[4] = (27 * params[2] - 27 * params[3] + 9 * params[4] - 9 * params[1]) / (2.0 * params[5]^3)
  coeff[5] = params[5]

  return coeff

end

# Extracts an ID number from the given control action based on 
# the control action starting and ending geometric parameters.
function getControlId(control_action::SpiralControlAction)::UInt128
  control_id::UInt128 = 0

  index::UInt128 = round((control_action.xf + X_LENGTH) / RESOLUTION)
  control_id |= (index & UInt128(0x3FF))

  index = round((control_action.yf + Y_LENGTH) / RESOLUTION)
  control_id |= ((index & UInt128(0x3FF)) << 10)

  index = getClosestIndex(control_action.ti, TI_RANGE)
  control_id |= ((index & UInt128(0x3FF)) << 20)

  index = getClosestIndex(control_action.tf, TF_RANGE)
  control_id |= ((index & UInt128(0x3FF)) << 30) 

  index = round(control_action.ki / CURVATURE_RESOLUTION)
  control_id |= ((index & UInt128(0x3FF)) << 40)

  index = round(control_action.kf / CURVATURE_RESOLUTION)
  control_id |= ((index & UInt128(0x3FF)) << 50)

  index = control_action.line_segment_count
  control_id |= ((index & UInt128(0x3FF)) << 60)

  return control_id

end

# Return the control action's ending state, after transforming it to the new
# angle and (x, y) offset.
# This will be useful for decomposition.
function transformControlActionEnd(control_action::SpiralControlAction, angle::Float64, 
  x_offset::Float64, y_offset::Float64, step_offset::Int64)::State
  state::State = State(x_offset + cos(angle)*control_action.xf - sin(angle)*control_action.yf,
    y_offset + sin(angle)*control_action.xf + cos(angle)*control_action.yf,
    control_action.tf + angle,
    control_action.kf,
    control_action.line_segment_count + step_offset)
  return state

end

# Returns the final state of a control action when starting from a
# given state.
function transformControlActionFromState(control_action::SpiralControlAction, 
  state::State)::State
  # Make sure the control action agrees with the state.
  ti_mod = state.theta % (pi / 2.0)
  @assert(abs(TI_RANGE[getClosestIndex(ti_mod, TI_RANGE)] - control_action.ti) < 0.01)
  # Get the rotation of the state relative to the control action.
  delta_theta = state.theta - control_action.ti
  @assert((round(delta_theta / (pi / 2.0)) - delta_theta / (pi / 2.0)) < 0.01)

  state_f::State = State(
    state.x + cos(delta_theta)*control_action.xf - sin(delta_theta)*control_action.yf,
    state.y + sin(delta_theta)*control_action.xf + cos(delta_theta)*control_action.yf,
    control_action.tf + delta_theta,
    control_action.kf,
    control_action.line_segment_count + state.steps)

  return state_f

end

# Return a point on the path formed by a control action, assuming the control
# action started at state. Step gives the location on the control action's path
# that is relevant.
function transformControlActionPoint(control_action::SpiralControlAction, start_state::State, 
  step::Int64)::Array{Float64}
  path_point::Array{Float64, 1} = control_action.path[step, :]

  # Intermediate curvatures have not been calculated and are not important.
  # Approximate it by using the ending curvature.
  delta_theta::Float64 = start_state.theta - control_action.ti
  @assert(round(delta_theta / (pi / 2.0)) - delta_theta / (pi / 2.0) < 0.01)
  transformed_point::Array{Float64, 1} = [start_state.x + cos(delta_theta) * path_point[1] - sin(delta_theta) * path_point[2], start_state.y + sin(delta_theta) * path_point[1] + cos(delta_theta) * path_point[2]]

  return transformed_point

end

# Get the predecessor state of the given state, assuming the current control
# action was applied.
function getPredecessorState(control_action::SpiralControlAction, end_state::State)::State
  # Curvature is preserved upon rotating a control action.
  ki::Float64 = control_action.ki
  # The initial angle is the final angle less the change in angle of the
  # control action.
  ti::Float64 = wrapTo2Pi(end_state.theta - (control_action.tf - control_action.ti))
  delta_theta::Float64 = ti - control_action.ti
  # The difference between the the initial control action angle and the
  # predecessor angle should be a multiple of pi/2.
  multiple::Float64 = delta_theta / (pi / 2.0)

  try
    @assert((round(multiple) - multiple) < 0.01)
  catch
    @printf("getPredecessorState: multiples didn't line up.")
  end

  # The predecessor x and y is then the inverse transformation applied for
  # rotating the control action.
  xi::Float64 = end_state.x - cos(delta_theta)*control_action.xf + sin(delta_theta)*control_action.yf
  yi::Float64 = end_state.y - sin(delta_theta)*control_action.xf - cos(delta_theta)*control_action.yf

  prev_steps = end_state.steps - control_action.line_segment_count

  return State(xi, yi, ti, ki, prev_steps)

end

end # module

