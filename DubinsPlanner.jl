module DubinsPlanner

using Constants, DubinsPath, Geometry, SwathGenerator, LatticeOccupancyGrid, LatticeVisualizer

export planDubinsPath

function planDubinsPath(center_line::Array{Float64, 2}, occupancy_grid::OccupancyGrid, max_curvature::Float64=0.5, lookahead::Float64=10.0)
  car_position = [0.0, 0.0, 0.0]
  car_curvature = 0.0
  # Add the initial state.
  output_path::Array{Float64} = [0.0, 0.0, 0.0]
  output_curvatures::Array{Float64} = [0.0]

  while true
    # First, find the goal state in the center line.
    goal_state = getGoalState(center_line, car_position, lookahead)

    # Transform the goal state such that it is in the
    # car's local frame.
    goal_state_local::Array{Float64} = globalToLocal(goal_state, car_position)

    # Calculate the array of goal states.
    goal_state_array_local::Array{Array{Float64}} = calculateGoalArray(goal_state_local)

    best_score::Float64 = Inf
    best_index::Int64 = GOAL_ARRAY_CENTER_INDEX
    best_path::Array{Float64, 2} = Array{Float64, 2}(0, 0)
    best_curvatures::Array{Float64} = Array{Float64}(0)
    best_swath::Set{Tuple{UInt64, UInt64}} = Set{Tuple{UInt64, UInt64}}()
    center_path::Array{Float64, 2} = Array{Float64, 2}(0, 0)
    center_swath::Set{Tuple{UInt64, UInt64}} = Set{Tuple{UInt64, UInt64}}()

    collision_scores::Array{Float64} = Array{Float64}(size(goal_state_array_local, 1))
    paths::Array{Array{Float64, 2}} = Array{Array{Float64, 2}}(size(goal_state_array_local, 1))
    curvature_vals_list::Array{Array{Float64}} = Array{Array{Float64}}(size(goal_state_array_local, 1))

    for i = 1:size(goal_state_array_local, 1)
      # Calculate the Dubins path. Assume final curvature is zero, want to end on a straight
      # line if possible.
      dubins_action = DubinsAction(goal_state_array_local[i][1], goal_state_array_local[i][2],
        0.0, goal_state_array_local[i][3], car_curvature, 0.0, max_curvature)
      if dubins_action.feasible == false
        @printf("Action was infeasible.")
        @printf("\n")
        return
      end
   
      # Get the path formed by the Dubins path in the local frame. 
      path_local::Array{Float64, 2} = dubins_action.path
      curvature_vals::Array{Float64} = dubins_action.curvature_vals

      # Convert the path to the global frame.
      path::Array{Float64, 2} = Array{Float64, 2}(size(path_local))
      for j = 1:size(path_local, 1)
        temp::Array{Float64} = localToGlobal(path_local[j, :], car_position)
        path[j, 1] = temp[1]
        path[j, 2] = temp[2]
        path[j, 3] = temp[3]
      end  

      swath::Set{Tuple{UInt64, UInt64}} = generateSwathFromPath(path)

      # Keep the center path, in case there are no feasible paths.
      if i == GOAL_ARRAY_CENTER_INDEX
        center_path = path
        center_swath = swath
      end
     
      # Biases towards the center of the goal array. 
      location_score::Float64 = (i - GOAL_ARRAY_CENTER_INDEX)^2

      # Calculate collision score from the swath formed by the path.
      collision_score::Float64 = getCostFromSwath(occupancy_grid, swath)

      collision_scores[i] = collision_score
      
      total_score::Float64 = location_score + collision_score
      paths[i] = path
      curvature_vals_list[i] = curvature_vals

      if total_score < best_score
        best_index = i
        best_score = total_score 
        best_path = path
        best_curvatures = curvature_vals
        best_swath = swath
      end

    end

    # If none of the paths were feasible, take the center path.
    if best_score == Inf
      error("Optimization planner path is infeasible.")
    end

    # Adjust indices to make sure we don't graze the object in the future.
    if (best_index > 1) && (best_index < size(goal_state_array_local, 1))
      if (collision_scores[best_index+1] == Inf) && (collision_scores[best_index-1] != Inf)
        best_index = best_index - 1
      elseif (collision_scores[best_index-1] == Inf) && (collision_scores[best_index+1] != Inf)
        best_index = best_index + 1
      end

      best_path = paths[best_index]
      best_curvatures = curvature_vals_list[best_index]

    end

    # If the goal state is at the end of the path,
    # we're done. Skip the first point to avoid duplicates.
    if sqrt((goal_state[1]-center_line[end, 1])^2 + (goal_state[2]-center_line[end, 2])^2) < 0.01
      for i = 2:size(best_path, 1)
        push!(output_path, best_path[i, 1])
        push!(output_path, best_path[i, 2])
        push!(output_path, best_path[i, 3])
        push!(output_curvatures, best_curvatures[i])
      end
      break
    end

    # Otherwise, move the car along the path to complete the iteration.
    moveCar(output_path, output_curvatures, best_path, best_curvatures, car_position)
    car_curvature = output_curvatures[end]
  
  end

  return (permutedims(reshape(output_path, 3, :), [2, 1]), output_curvatures)

end

# Finds the state in the center line that is within
# the lookahead distance of the car.
function getGoalState(center_line::Array{Float64, 2}, car_position::Array{Float64}, lookahead_distance::Float64)::Array{Float64}
  arc_length_acc::Float64 = 0.0

  # Find the point in the center line closest to the car's current
  # position.
  min_dist::Float64 = Inf
  closest_point_index::Int64 = 1
  for i = 1:size(center_line, 1)
    temp_dist::Float64 = (center_line[i, 1] - car_position[1])^2 + (center_line[i, 2] - car_position[2])^2
    if temp_dist < min_dist
      min_dist = temp_dist
      closest_point_index = i
    end
  end


  # If the closest point index is the last one in the centerline,
  # then return that as the goal.
  if closest_point_index == size(center_line, 1)
    return center_line[closest_point_index, :]
  end

  goal_index::Int64 = closest_point_index
  arc_length_accumulated::Float64 = 0.0
  # From the closest point to the car in the centerline, find
  # the point that is within the lookahead distance of the
  # point.
  for i = closest_point_index:size(center_line, 1)-1
    arc_length_accumulated += sqrt((center_line[i+1, 1] - center_line[i, 1])^2 + (center_line[i+1, 2] - center_line[i, 2])^2)
    goal_index = i + 1

    if arc_length_accumulated >= lookahead_distance 
      break
    end

  end

  return center_line[goal_index, :]

end

# Transforms to_transform to the frame of base_state.
function globalToLocal(to_transform::Array{Float64},
  base_state::Array{Float64})::Array{Float64}
  # Translate the state such that the base state is at the origin.
  transformed::Array{Float64} = [to_transform[1] - base_state[1], to_transform[2] - base_state[2], to_transform[3]]
  # Rotate the point such that the relative angle is preserved, but the
  # base frame has zero angle.
  transformed = [transformed[1]*cos(-base_state[3])-transformed[2]*sin(-base_state[3]), transformed[1]*sin(-base_state[3])+transformed[2]*cos(-base_state[3]), transformed[3]-base_state[3]]

  return transformed

end

# Transforms to_transform (that is in the frame
# of base state) into the global frame.
function localToGlobal(to_transform::Array{Float64},
  base_state::Array{Float64})::Array{Float64}
  # Rotate the point into the global frame, assuming it is currently
  # in the frame of the base state.
  transformed::Array{Float64} = [to_transform[1]*cos(base_state[3])-to_transform[2]*sin(base_state[3]), to_transform[1]*sin(base_state[3])+to_transform[2]*cos(base_state[3]), to_transform[3]+base_state[3]]

  # Translate the state, assuming the base state was the origin.
  transformed = [transformed[1] + base_state[1], transformed[2] + base_state[2], transformed[3]]

  return transformed

end

# Calculates the laterally offset goal array from the input goal
# state.
function calculateGoalArray(goal_state_local::Array{Float64})::Array{Array{Float64}}
  goal_array::Array{Array{Float64}} = []
  for i = 1:GOAL_ARRAY_SIZE
    offset::Float64 = GOAL_ARRAY_STEP * (i - GOAL_ARRAY_CENTER_INDEX)
    # Point is offset laterally from the goal, while preserving
    # the heading of the goal state.
    x = goal_state_local[1] + offset * cos(goal_state_local[3] + pi/2)
    y = goal_state_local[2] + offset * sin(goal_state_local[3] + pi/2)
    
    push!(goal_array, [x, y, goal_state_local[3]])

  end

  return goal_array
     
end

# Moves the car by a set distance along the given path.
# The output path is collapsed to 1D to allow for push
# operations.
function moveCar(output_path::Array{Float64}, output_curvatures::Array{Float64}, planned_path::Array{Float64, 2}, planned_curvatures::Array{Float64}, car_position::Array{Float64})
  arc_length_accumulated::Float64 = 0.0

  for i = 1:size(planned_path, 1)-1 
    arc_length_accumulated += sqrt((planned_path[i+1, 1] - planned_path[i, 1])^2 + (planned_path[i+1, 2] - planned_path[i, 2])^2)

    # Append the values at this step to the output.
    push!(output_path, planned_path[i+1, 1])
    push!(output_path, planned_path[i+1, 2])
    push!(output_path, planned_path[i+1, 3])
    push!(output_curvatures, planned_curvatures[i+1])

    # Make sure to do this by reference, if we do an assignment
    # then the original car position won't be changed.
    car_position[1] = planned_path[i+1, 1]
    car_position[2] = planned_path[i+1, 2]
    car_position[3] = planned_path[i+1, 3]

    if arc_length_accumulated > CAR_STEP
      break
    end
     
  end 

end


end #module
