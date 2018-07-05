module PathConstructor

using Constants, LatticeState, PolynomialSpiral, LatticePathFinder, Geometry

export constructPath

function constructPath(control_action_lut::Array{Dict{UInt128, SpiralControlAction}},
  path::Array{Float64, 2})

  @time result = findClosestPath(control_action_lut, path)
  best_dl::Float64 = result[1]
  best_lattice_path::Array{UInt128} = result[2]
  best_control_action_sequence::Array{UInt128} = result[3]

  closest_path::Array{Float64, 1} = []
  action_points::Array{Float64, 1} = []

  control_start_position::Array{Float64} = [0.0, 0.0]
  control_start_heading::Float64 = 0.0
  control_start_steps::Int64 = 0

  start_state::State = State(0.0, 0.0, 0.0, 0.0, 0)

  # Iterate over the optimal set of control actions,
  # reconstructing the path from each translated and rotated
  # control action path.
  path_length::Int64 = 0
  for i = 1:size(best_control_action_sequence, 1)
    control_id = best_control_action_sequence[i]
    append!(action_points, control_start_position)
    ti_index::UInt64 = (control_id & (UInt128(0x3FF) << 20)) >> 20 

    action::SpiralControlAction = control_action_lut[ti_index][control_id]

    delta_theta::Float64 = control_start_heading - action.ti
    delta_theta_multiple::Float64 = delta_theta / (pi / 2.0)
    @assert(abs(round(delta_theta_multiple) - delta_theta_multiple) < 0.01)

    end_state::State = transformControlActionEnd(action, delta_theta, control_start_position[1], control_start_position[2], control_start_steps)

    # Transform the points of each control action path based on the
    # ending state of each previous control action. If the last point
    # in the control action path is too close to the endpoint of the
    # control action, we don't add it to the path to avoid double counting
    # (as it will be a point at the start of the next control action).
    # If it is the last control action in the optimal set, then we include
    # the final endpoint regardless, as there are no further control actions.
    for j = 1:size(action.path, 1)
      point = transformControlActionPoint(action, start_state, j)
      if (norm(point-[end_state.x, end_state.y]) < (PATH_RESOLUTION / 2) && (i < size(best_control_action_sequence, 1)))
        @assert(j == size(action.path, 1))
        break
      else
        append!(closest_path, point)
        path_length += 1
      end

      if path_length >= PATH_LENGTH
        @assert(i == size(best_control_action_sequence, 1))
        break
      end
    end

    control_start_position = [end_state.x, end_state.y]
    control_start_heading = end_state.theta
    control_start_steps = end_state.steps
    
    start_state = end_state

  end

  closest_path_final::Array{Float64, 2} = permutedims(reshape(closest_path, 2, round(Int64, length(closest_path) / 2)), [2, 1])
  action_points_final::Array{Float64, 2} = permutedims(reshape(action_points, 2, round(Int64, length(action_points) / 2)), [2, 1])

  return (closest_path_final, action_points_final)

end


end # module
