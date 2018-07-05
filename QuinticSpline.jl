module QuinticSpline

using Constants, LatticeState, Utils

export SplineControlAction, calculateQuinticSpline,
  sampleQuinticSpline, getControlId, transformControlActionEnd,
  transformControlActionPoint, transformControlActionFromState,
  getPredecessorState

const ETA = [0.2, 0.2, 0.0, 0.0]

struct SplineControlAction
  xf::Float64
  yf::Float64
  ti::Float64
  tf::Float64
  ki::Float64
  kf::Float64
  x_params::Array{Float64, 1}
  y_params::Array{Float64, 1}
  path::Array{Float64, 2}
  line_segment_count::Int64
  feasible::Bool
  arc_length::Float64

  function SplineControlAction(xf::Float64, yf::Float64, ti::Float64, tf::Float64,
    ki::Float64, kf::Float64)

    ti = wrapTo2Pi(ti)
    tf = wrapTo2Pi(tf)
    spline_params = calculateQuinticSpline(xf, yf, ti, tf, ki, kf)
    x_params = spline_params[1]
    y_params = spline_params[2]

    path_vals = sampleQuinticSpline(x_params, y_params, ti)
    path = path_vals[1]
    line_segment_count = path_vals[2]
    arc_length = path_vals[2]

    @assert(size(path, 1) - 1 == line_segment_count)
    new(xf, yf, wrapTo2Pi(ti), wrapTo2Pi(tf), ki, kf, x_params, y_params, path, 
      line_segment_count, true, arc_length)
  end

end

# Given a start and final state, calculate the
# spline parameters of the associated quintic spline.
function calculateQuinticSpline(xf::Float64, yf::Float64, ti::Float64, tf::Float64,
  ki::Float64, kf::Float64)
  xi = 0.0
  yi = 0.0
  n1 = ETA[1]
  n2 = ETA[2]
  n3 = ETA[3]
  n4 = ETA[4]

  x_params::Array{Float64} = zeros(Float64, 6)
  y_params::Array{Float64} = zeros(Float64, 6)

  x_params[1] = xi
  x_params[2] = n1*cos(ti)
  x_params[3] = 0.5*(n3*cos(ti) - n1^2*ki*sin(ti))
  x_params[4] = 10.0*(xf-xi) - (6.0*n1+1.5*n3)*cos(ti) - (4.0*n2-0.5*n4)*cos(tf) + 1.5*n1^2*ki*sin(ti) - 0.5*n2^2*kf*sin(tf)
  x_params[5] = -15.0*(xf-xi) + (8.0*n1+1.5*n3)*cos(ti) + (7.0*n2-n4)*cos(tf) - 1.5*n1^2*ki*sin(ti) + n2^2*kf*sin(tf)
  x_params[6] = 6.0*(xf-xi) - (3.0*n1+0.5*n3)*cos(ti) - (3.0*n2-0.5*n4)*cos(tf) + 0.5*n1^2*ki*sin(ti) - 0.5*n2^2*kf*sin(tf)

  y_params[1] = yi
  y_params[2] = n1*sin(ti)
  y_params[3] = 0.5*(n3*sin(ti) + n1^2*ki*cos(ti))
  y_params[4] = 10.0*(yf-yi) - (6.0*n1+1.5*n3)*sin(ti) - (4.0*n2-0.5*n4)*sin(tf) - 1.5*n1^2*ki*cos(ti) + 0.5*n2^2*kf*cos(tf)
  y_params[5] = -15.0*(yf-yi) + (8.0*n1+1.5*n3)*sin(ti) + (7.0*n2-n4)*sin(tf) + 1.5*n1^2*ki*cos(ti) - n2^2*kf*cos(tf)
  y_params[6] = 6.0*(yf-yi) - (3.0*n1+0.5*n3)*sin(ti) - (3.0*n2-0.5*n4)*sin(tf) - 0.5*n1^2*ki*cos(ti) + 0.5*n2^2*kf*cos(tf)

  return (x_params, y_params)
  
end

# Given the parameters of the spline, sample the path to approximate
# the spline using line segments, with each line segment having equal
# length. Also saves the number of line segments used to approximate
# the spline.
function sampleQuinticSpline(x_params::Array{Float64}, y_params::Array{Float64}, ti::Float64)
  n1 = ETA[1]
  n2 = ETA[2]
  n3 = ETA[3]
  n4 = ETA[4]

  arc_length = 0.0

  # Store the path as a 1-D array to start, then reshape at
  # the end.
  path::Array{Float64, 1} = Array{Float64, 1}()
  # Initial state is (0, 0, ti).
  append!(path, [0.0, 0.0, ti])

  # This calculates the search resolution along the spline
  # for each line segment.
  steps::Int64 = trunc(ceil(1.0 / ARC_LENGTH_RESOLUTION))

  u_start::Float64 = 0.0
  line_segment_count::Int64 = 0

  for i = 1:steps
    u1::Float64 = u_start
    u2::Float64 = convert(Float64, i) / convert(Float64, steps)

    x1::Float64 = x_params[1] + x_params[2]*u1 + x_params[3]*u1^2 + x_params[4]*u1^3 + x_params[5]*u1^4 + x_params[6]*u1^5
    x2::Float64 = x_params[1] + x_params[2]*u2 + x_params[3]*u2^2 + x_params[4]*u2^3 + x_params[5]*u2^4 + x_params[6]*u2^5

    y1::Float64 = y_params[1] + y_params[2]*u1 + y_params[3]*u1^2 + y_params[4]*u1^3 + y_params[5]*u1^4 + y_params[6]*u1^5
    y2::Float64 = y_params[1] + y_params[2]*u2 + y_params[3]*u2^2 + y_params[4]*u2^3 + y_params[5]*u2^4 + y_params[6]*u2^5

    theta::Float64 = atan2(y2-y1, x2-x1)
    v1::Array{Float64} = [x1, y1]
    v2::Array{Float64} = [x2, y2]

    if norm(v2-v1) > PATH_RESOLUTION
      append!(path, [x2, y2, theta])
      u_start = u2
      arc_length += norm(v2-v1)
      line_segment_count += 1
    end
  end

  # If the last point adds more than half of the arc length resolution to the path,
  # count it as a full line segment.
  v1 = [path[end-2], path[end-1]]
  xf::Float64 = x_params[1] + x_params[2]*1.0 + x_params[3]*1.0^2 + x_params[4]*1.0^3 + x_params[5]*1.0^4 + x_params[6]*1.0^5
  yf::Float64 = y_params[1] + y_params[2]*1.0 + y_params[3]*1.0^2 + y_params[4]*1.0^3 + y_params[5]*1.0^4 + y_params[6]*1.0^5
  v2 = [xf, yf]
  tf = atan2(v2[2]-v1[2], v2[1]-v1[1])
  if norm(v2-v1) > (PATH_RESOLUTION / 2)
    append!(path, [xf, yf, tf])
    line_segment_count += 1
  end
  # Increment the arc length regardless.
  arc_length += norm(v2-v1)

  # Reshape the path array to be 3 x N, where N is the number of steps, 
  # and 3 is the dimension in SE(2). Then, take the transpose. This is
  # done because the arrays are column major.
  return (permutedims(reshape(path, 3, round(Int64, length(path)/3)), [2, 1]), line_segment_count, arc_length)

end

# Extracts an ID number from the given control action based on 
# the control action starting and ending geometric parameters.
function getControlId(control_action::SplineControlAction)::UInt128
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
function transformControlActionEnd(control_action::SplineControlAction, angle::Float64, 
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
function transformControlActionFromState(control_action::SplineControlAction,
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
function transformControlActionPoint(control_action::SplineControlAction, start_state::State, 
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
function getPredecessorState(control_action::SplineControlAction, end_state::State)::State
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
