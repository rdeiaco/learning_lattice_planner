module LatticeState

using Constants, Utils

export State, getStateId, getSpatialStateId, getStateFromID, getDeltaState

mutable struct State
  x::Float64
  y::Float64
  theta::Float64
  k::Float64
  steps::Int64

  function State(x::Float64, y::Float64, theta::Float64, k::Float64, steps::Int64)
    # Round to the nearest grid point.
    x = round(x / RESOLUTION) * RESOLUTION
    y = round(y / RESOLUTION) * RESOLUTION
    # Round the input angle to the nearest value of the
    # valid final angles, after shifting it to be within
    # [0, 2*pi).
    theta = wrapTo2Pi(theta)
    t_index = getClosestIndex(theta, ANGLE_RANGE)
    # Handle the case when the angle is close to 2*pi, so we
    # need to roll over back to 0.
    if abs(theta - 2*pi) < abs(theta - ANGLE_RANGE[t_index])
      theta = 0.0
    else
      theta = ANGLE_RANGE[t_index]
    end

    new(x, y, theta, k, steps)
  end
end

# To handle proper storage in a hash table, we need to ensure the same
# State maps to the same location in the hash table. With floating point
# arithemtic, this is problematic, since small numerical errors can result
# in a different key, even though the intended State is the same. Since the
# obstacle occupancy grid is discretized to a resolution of 0.1 m, we can
# map each State's position to an integer representation.
# Curvature and heading can only take on so many possible values, so they
# can be enumerated.
function getStateId(state::State)::UInt128
  state_id::UInt128 = 0
  
  delta_k::Float64 = 0
  if size(KF_RANGE, 1) > 1
    delta_k = KF_RANGE[2] - KF_RANGE[1]
  end

  # Since x values can be negative, add the maximum positive value
  # along the x-axis to ensure we get only positive integers.
  x_val::UInt128 = round(UInt128, (state.x + X_LENGTH) / RESOLUTION)

  # Since y values can be negative, add the maximum positive value
  # along the y-axis to ensure we get only positive integers.
  y_val::UInt128 = round(UInt128, (state.y + Y_LENGTH) / RESOLUTION)

  # Theta values can only take on so many values, so find the closest
  # one available in the list.
  t_val = getClosestIndex(state.theta, ANGLE_RANGE)
  @assert(abs(ANGLE_RANGE[t_val] - state.theta) < 0.01)

  # Curvature values can only take on so many values, so enumerate them.
  # Offset the curvature to ensure it is positive.

  k_val::UInt128 = 0
  if delta_k > 0
    k_val = round(UInt128, (state.k + KF_RANGE[end]) / delta_k) 
  end

  # 10 bits gives 1024 denominations of each type, which should be enough
  # granularity for now. Throw an error otherwise.
  @assert((round(X_LENGTH / RESOLUTION) <= 1024) && (round(2*Y_LENGTH / RESOLUTION) <= 1024) && (size(ANGLE_RANGE, 1) <= 1024) && (size(KF_RANGE, 1) <= 1024))

  # 10 bits gives 1024 as the upper bound for number of line segment steps,
  # which should be enough.
  s_val::UInt128 = state.steps

  state_id |= (UInt128(0x3FF) & x_val)
  state_id |= ((UInt128(0x3FF) & y_val) << 10)
  state_id |= ((UInt128(0x3FF) & t_val) << 20)
  state_id |= ((UInt128(0x3FF) & k_val) << 30)
  state_id |= ((UInt128(0x3FF) & s_val) << 40)

  return state_id

end


# This is an alternative ID that doesn't take into consideration
# the steps required to reach a state.
function getSpatialStateId(state::State)::UInt128
  state_id::UInt128 = 0
  
  delta_k::Float64 = 0
  if size(KF_RANGE, 1) > 1
    delta_k = KF_RANGE[2] - KF_RANGE[1]
  end

  # Since x values can be negative, add the maximum positive value
  # along the x-axis to ensure we get only positive integers.
  x_val::UInt128 = round(UInt128, (state.x + X_LENGTH) / RESOLUTION)

  # Since y values can be negative, add the maximum positive value
  # along the y-axis to ensure we get only positive integers.
  y_val::UInt128 = round(UInt128, (state.y + Y_LENGTH) / RESOLUTION)

  # Theta values can only take on so many values, so find the closest
  # one available in the list.
  t_val = getClosestIndex(state.theta, ANGLE_RANGE)
  @assert(abs(ANGLE_RANGE[t_val] - state.theta) < 0.01)

  # Curvature values can only take on so many values, so enumerate them.
  # Offset the curvature to ensure it is positive.

  k_val::UInt128 = 0
  if delta_k > 0
    k_val = round(UInt128, (state.k + KF_RANGE[end]) / delta_k) 
  end

  # 10 bits gives 1024 denominations of each type, which should be enough
  # granularity for now. Throw an error otherwise.
  @assert((round(X_LENGTH / RESOLUTION) <= 1024) && (round(2*Y_LENGTH / RESOLUTION) <= 1024) && (size(ANGLE_RANGE, 1) <= 1024) && (size(KF_RANGE, 1) <= 1024))

  state_id |= (UInt128(0x3FF) & x_val)
  state_id |= ((UInt128(0x3FF) & y_val) << 10)
  state_id |= ((UInt128(0x3FF) & t_val) << 20)
  state_id |= ((UInt128(0x3FF) & k_val) << 30)

  return state_id

end
   
function getStateFromID(state_id::UInt128)::State
  delta_k::Float64 = 0.0
  if size(KF_RANGE, 1) > 1
    delta_k = KF_RANGE[2] - KF_RANGE[1]
  end

  x::Float64 = convert(Float64, (UInt128(0x3FF) & state_id)) * RESOLUTION - X_LENGTH
  # The y value is offset to account for negative values.
  y::Float64 = convert(Float64, ((UInt128(0xFFC00) & state_id) >> 10)) * RESOLUTION - Y_LENGTH
  t::Float64 = ANGLE_RANGE[(UInt128(0x3FF00000) & state_id) >> 20]
 
  k::Float64 = 0.0 
  if delta_k > 0.0
    k = convert(Float64, ((UInt128(0xFFC0000000) & state_id) >> 30)) * delta_k - KF_RANGE[end]
  end

  steps::Int64 = (UInt128(0x3FF0000000000) & state_id) >> 40

  return State(x, y, t, k, steps)

end

# Calculates the difference between two states such that
# the origin input state is transformed to its associated
# true origin.
function getDeltaState(origin::State, goal::State)::State
  ti_mod::Float64 = origin.theta % (pi / 2.0)
  if abs(ti_mod - (pi / 2.0)) < 0.01
    ti_mod = 0.0
  end

  ti_index::UInt64 = getClosestIndex(ti_mod, TI_RANGE)

  delta_theta::Float64 = origin.theta - ti_mod
  multiple::Float64 = delta_theta / (pi / 2.0)
  @assert((round(multiple) - multiple) < 0.01)

  x_goal_translated::Float64 = goal.x - origin.x
  y_goal_translated::Float64 = goal.y - origin.y
  goal_transformed::State = State(x_goal_translated * cos(delta_theta) + y_goal_translated * sin(delta_theta),
    -x_goal_translated * sin(delta_theta) + y_goal_translated * cos(delta_theta),
    goal.theta - delta_theta, 0.0, 0)

  return goal_transformed

end

end # module
