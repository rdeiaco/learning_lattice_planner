module Geometry

using PyPlot, Constants, LatticeState, PolynomialSpiral

export getMaxLineSegmentDistance, sampleSpiral, sampleStraightAway, fTheta, fKappa, mengerCurvature

# Calculates the maximum distance between two
# line segements while moving along both at the
# same rate.
function getMaxLineSegmentDistance(u1::Array{Float64}, u2::Array{Float64}, 
  v1::Array{Float64}, v2::Array{Float64}, debug::Bool=false)::Float64
  d1 = norm(v1-u1)
  d2 = norm(v2-u2)

  if debug
    @printf("u1: %f, %f\n", u1[1], u1[2])
    @printf("v1: %f, %f\n", v1[1], v1[2])
    @printf("u2: %f, %f\n", u2[1], u2[2])
    @printf("v2: %f, %f\n", v2[1], v2[2])
  end

  return max(d1, d2)

end
 
# Samples the spiral encoded in the spiral coefficients by 
# using the Trapezoidal rule. 
function sampleSpiral(coefficients::Array{Float64}, ti::Float64, xi::Float64=0.0, yi::Float64=0.0)
  path::Array{Float64, 1} = Array{Float64, 1}()
  curvature_vals::Array{Float64, 1} = Array{Float64, 1}()

  steps::Int64 = trunc(ceil(1.0 / ARC_LENGTH_RESOLUTION))

  s_start::Float64 = 0.0
  theta_start::Float64 = fTheta(coefficients, s_start, ti)
  line_segment_count::Int64 = 0

  # Append the start point of the path.
  append!(path, [xi, yi, theta_start])
  push!(curvature_vals, fKappa(coefficients, 0.0))

  deltax1::Float64 = 0.0
  deltay1::Float64 = 0.0
  deltax2::Float64 = 0.0
  deltay2::Float64 = 0.0

  x2::Float64 = xi
  y2::Float64 = yi

  for i = 1:steps
    s1::Float64 = s_start
    s2::Float64 = convert(Float64, i) / convert(Float64, steps) * coefficients[5]
    theta1::Float64 = theta_start
    theta2::Float64 = fTheta(coefficients, s2, ti)

    deltax2 = (cos(theta2) + cos(theta1)) / (2 * convert(Float64, i)) + deltax1 * (i - 1) / convert(Float64, i)
    deltay2 = (sin(theta2) + sin(theta1)) / (2 * convert(Float64, i)) + deltay1 * (i - 1) / convert(Float64, i)

    x2 = deltax2 * s2 + xi
    y2 = deltay2 * s2 + yi

    deltax1 = deltax2
    deltay1 = deltay2
    
    if (s2 - s1) > PATH_RESOLUTION
      append!(path, [x2, y2, theta2])
      push!(curvature_vals, fKappa(coefficients, s2))
      s_start = s2
      theta_start = theta2
      line_segment_count += 1
    elseif i == steps
      if (s2 - s1) > (PATH_RESOLUTION / 2.0)
        append!(path, [x2, y2, theta2])
        push!(curvature_vals, fKappa(coefficients, s2))
        line_segment_count += 1
      end
    end

  end

  return (permutedims(reshape(path, 3, :), [2, 1]), line_segment_count, curvature_vals)

end

# Samples a straight line with the given position and heading.
function sampleStraightAway(arc_length::Float64, ti::Float64, xi::Float64=0.0, yi::Float64=0.0)
  steps::Int64 = trunc(ceil(arc_length / PATH_RESOLUTION))
  path::Array{Float64, 1} = Array{Float64, 1}()

  append!(path, [xi, yi, ti])
  
  for i = 1:steps
    append!(path, [xi + i * cos(ti) * PATH_RESOLUTION, yi + i * sin(ti) * PATH_RESOLUTION, ti])
  end

  return permutedims(reshape(path, 3, :), [2, 1])

end

# Evaluates theta at a given arc length along the spiral.
function fTheta(coefficients::Array{Float64}, s::Float64, ti::Float64)::Float64
  return ti + coefficients[1] * s + coefficients[2] * s^2 / 2.0 + coefficients[3] * s^3 / 3.0 + coefficients[4] * s^4 / 4.0

end

# Evaluates curvature at a given arc length along the spiral.
function fKappa(coefficients::Array{Float64}, s::Float64)::Float64
  return coefficients[1] + coefficients[2] * s + coefficients[3] * s^2 + coefficients[4] * s^3

end

# Does a 3-point estimate of menger curvature.
function mengerCurvature(p1, p2, p3)
  delta_x = p2[1] - p1[1]
  delta_y = p2[2] - p1[2]
  
  a = sqrt(delta_x*delta_x + delta_y*delta_y)

  delta_x = p3[1] - p2[1]
  delta_y = p3[2] - p2[2]
  b = sqrt(delta_x*delta_x + delta_y*delta_y)

  delta_x = p1[1] - p3[1]
  delta_y = p1[2] - p3[2]
  c = sqrt(delta_x*delta_x + delta_y*delta_y)

  if a < 1e-5 || b < 1e-5 || c < 1e-5
    return 0.0
  end

  s = (a + b + c) / 2.0 

  K = sqrt(abs(s * (s - a) * (s - b) * (s - c)))

  curvature = 4 * K / (a * b * c)

  rotate_direction = (p2[1] - p1[1]) * (p3[2] - p2[2]) - (p2[2] - p1[2]) * (p3[1] - p2[1])

  if rotate_direction < 0.0
    curvature = -curvature
  end

  return curvature

end

end # module
