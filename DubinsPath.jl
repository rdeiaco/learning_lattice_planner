module DubinsPath

using Constants, LatticeState, Utils, Geometry, PyCall
@pyimport dubins

export DubinsAction, getControlId, transformControlActionEnd, transformControlActionFromState, transformControlActionPoint, getPredecessorState

struct DubinsAction
  xf::Float64
  yf::Float64
  ti::Float64
  tf::Float64
  ki::Float64
  kf::Float64
  kmax::Float64
  path::Array{Float64, 2}
  curvature_vals::Array{Float64}
  line_segment_count::Int64
  arc_length::Float64
  feasible::Bool

  function DubinsAction(xf::Float64, yf::Float64, ti::Float64, tf::Float64,
    ki::Float64, kf::Float64, kmax::Float64=0.5)

    ti = wrapTo2Pi(ti)
    tf = wrapTo2Pi(tf)

    path_vals = sampleDubinsPath(xf, yf, ti, tf, kmax)
    path = path_vals[1]
    line_segment_count = path_vals[2]
    curvature_vals = path_vals[3]
    arc_length = path_vals[4]
    if arc_length > 2*norm([xf, yf])
      feasible = false
    else
      feasible = true
    end


    new(xf, yf, ti, tf, ki, kf, kmax, path, curvature_vals, line_segment_count, arc_length, feasible)

  end

end

function sampleDubinsPath(xf::Float64, yf::Float64, ti::Float64, tf::Float64, kmax::Float64)
  start_config = (0.0, 0.0, ti)
  end_config = (xf, yf, tf)

  # Sanitize inputs.
  if kmax < 1e-7
    kmax = 1e-7
  end

  turning_radius = 1.0 / kmax
  path_struct = dubins.shortest_path(start_config, end_config, turning_radius)
  sample_result = path_struct[:sample_many](PATH_RESOLUTION)
  sampled_path = sample_result[1]

  # Check to see if there is at least half of a step size at the end of the path.
  # If there is, then we can add the final point to the end of the path.
  if norm([sampled_path[end][1], sampled_path[end][2]] - [xf, yf]) > 0.5*PATH_RESOLUTION
    push!(sampled_path, (xf, yf, tf))
  end

  # Format the sample path into a 2D array for compatibility.
  path::Array{Float64, 2} = Array{Float64, 2}(size(sampled_path, 1), 3)
  for i = 1:size(sampled_path, 1)
    path[i, 1] = sampled_path[i][1] 
    path[i, 2] = sampled_path[i][2] 
    path[i, 3] = sampled_path[i][3] 
  end
  
  # The line segment count is one less than the number of endpoints.
  line_segment_count = size(path, 1) - 1

  # Sample the menger curvature.
  curvature_vals::Array{Float64} = Array{Float64}(size(path, 1))
  STEP_SIZE = 5
  STEP_RADIUS = 2

  for i = STEP_RADIUS+1:size(curvature_vals, 1)-STEP_RADIUS
    p1 = [path[i-STEP_RADIUS, 1], path[i-STEP_RADIUS, 2]]
    p2 = [path[i, 1], path[i, 2]]
    p3 = [path[i+STEP_RADIUS, 1], path[i+STEP_RADIUS, 2]]
    curvature_vals[i] = mengerCurvature(p1, p2, p3)

    for i = 1:STEP_RADIUS
      curvature_vals[i] = curvature_vals[STEP_RADIUS+1]
    end

    for i = (size(curvature_vals, 1)-STEP_RADIUS+1):size(curvature_vals, 1)
      curvature_vals[i] = curvature_vals[size(curvature_vals, 1)-STEP_RADIUS]
    end

  end

  # Calculate the arc length of the primitive.
  arc_length::Float64 = 0.0
  for i = 1:size(path, 1)-1
    arc_length += norm(path[i+1, 1:2] - path[i, 1:2])
  end

  return (path, line_segment_count, curvature_vals, arc_length)

end

# Extracts an ID number from the given control action based on 
# the control action starting and ending geometric parameters.
function getControlId(control_action::DubinsAction)::UInt128
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

  index = round(control_action.kmax / CURVATURE_RESOLUTION)
  control_id |= ((index & UInt128(0x3FF)) << 70)

  return control_id

end

# Return the control action's ending state, after transforming it to the new
# angle and (x, y) offset.
# This will be useful for decomposition.
function transformControlActionEnd(control_action::DubinsAction, angle::Float64, 
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
function transformControlActionFromState(control_action::DubinsAction, 
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
function transformControlActionPoint(control_action::DubinsAction, start_state::State, 
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
function getPredecessorState(control_action::DubinsAction, end_state::State)::State
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
