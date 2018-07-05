module OptimizationPlanner

using Constants, PolynomialSpiral, Geometry, SwathGenerator, LatticeOccupancyGrid, LatticeVisualizer

export planOptimizerPath

function planOptimizerPath(center_line::Array{Float64, 2}, occupancy_grid::OccupancyGrid)::Array{Float64, 2}
  car_position = [0.0, 0.0, 0.0]
  output_path::Array{Float64} = []

  while true
    # First, find the goal state in the center line.
    goal_state = getGoalState(center_line, car_position)

    @printf("Goal state = ")
    show(goal_state)
    @printf("\n")

    @printf("Car position = ")
    show(car_position)
    @printf("\n")

    @printf("Center line end = ")
    show(center_line[end, :])
    @printf("\n")

    # Transform the goal state such that it is in the
    # car's local frame.
    goal_state_local::Array{Float64} = globalToLocal(goal_state, car_position)
    #@printf("Goal state local = ")
    #show(goal_state_local)
    #@printf("\n")

    # Calculate the array of goal states.
    goal_state_array_local::Array{Array{Float64}} = calculateGoalArray(goal_state_local)

    best_score::Float64 = Inf
    best_index::Int64 = GOAL_ARRAY_CENTER_INDEX
    best_path::Array{Float64, 2} = Array{Float64, 2}(0, 0)
    best_swath::Set{Tuple{UInt64, UInt64}} = Set{Tuple{UInt64, UInt64}}()
    center_path::Array{Float64, 2} = Array{Float64, 2}(0, 0)
    center_swath::Set{Tuple{UInt64, UInt64}} = Set{Tuple{UInt64, UInt64}}()

    for i = 1:size(goal_state_array_local, 1)
#      @printf("Goal state array = ")
#      show(goal_state_array_local)
#      @printf("\n")

      # Calculate the spiral path.
      opt_result = optimizeSpiral(goal_state_array_local[i][1], goal_state_array_local[i][2],
        0.0, goal_state_array_local[i][3], 0.0, 0.0)
      if (opt_result[2] == :Optimal) || (opt_result[2] == :UserLimit)
        feasible = true
      else
        feasible = false
        @printf("Opt result = ")
        show(opt_result)
        @printf("\n")
        return
      end
   
      # Get the path formed by the spiral in the local frame. 
      path_local::Array{Float64, 2} = sampleSpiral(getSpiralCoefficients(opt_result[1]), 0.0)[1]
    #  @printf("Goal local = (%f, %f, %f)\n", goal_state_array_local[i][1], goal_state_array_local[i][2], goal_state_array_local[i][3])
    #  @printf("Path end local = (%f, %f, %f)\n", path_local[end, 1], path_local[end, 2], path_local[end, 3]) 
    #  @printf("\n\n")

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
    
#      @printf("Location Score = %f\n", location_score)
#      @printf("Collision Score = %f\n", collision_score)
      
      total_score::Float64 = location_score + collision_score
      if total_score < best_score
        best_index = i
        best_score = total_score 
        best_path = path
        best_swath = swath
        #@printf("Best path = ")
        #show(best_path)
        #@printf("\n")
      end

    end

    # If none of the paths were feasible, take the center path.
    if best_score == Inf
      best_path = center_path
      best_swath = center_swath
    end

    #@printf("Best path goal = ")
    #show(best_path[end, :])
    #@printf("\n")

    #plotPathOnOccupancy(best_path, best_swath, occupancy_grid, center_line)

    # If the goal state is at the end of the path,
    # we're done.
    @printf("dist to centerline end = %f\n", sqrt((goal_state[1]-center_line[end, 1])^2 + (goal_state[2]-center_line[end, 1])^2))
    if sqrt((goal_state[1]-center_line[end, 1])^2 + (goal_state[2]-center_line[end, 2])^2) < 0.01
      for i = 1:size(best_path, 1)
        push!(output_path, best_path[i, 1])
        push!(output_path, best_path[i, 2])
        push!(output_path, best_path[i, 3])
      end
      break
    end

    # Otherwise, move the car along the path to complete the iteration.
    moveCar(output_path, best_path, car_position)

  end

  return permutedims(reshape(output_path, 3, :), [2, 1])

end

# Finds the state in the center line that is within
# the lookahead distance of the car.
function getGoalState(center_line::Array{Float64, 2}, car_position::Array{Float64})::Array{Float64}
  arc_length_acc::Float64 = 0.0

  #@printf("Centerline = ")
  #show(center_line)
  #@printf("\n\n\n")

  #@printf("Car position = ")
  #show(car_position)
  #@printf("\n\n")

  # Find the point in the center line closest to the car's current
  # position.
  min_dist::Float64 = Inf
  closest_point_index::Int64 = 1
  for i = 1:size(center_line, 1)
    temp_dist::Float64 = (center_line[i, 1] - car_position[1])^2 + (center_line[i, 2] - car_position[2])^2
    if temp_dist < min_dist
      min_dist = temp_dist
      closest_point_index = i
      #@printf("New closest point = ")
      #show(center_line[closest_point_index, :])
      #@printf("\n\n")
    end
  end

  #@printf("Closest point = ")
  #show(center_line[closest_point_index, :])
  #@printf("\n\n")

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

    #@printf("Current goal = ")
    #show(center_line[i+1, :])
    #@printf("\n\n")
    #@printf("Arc length accumulated = %f\n\n", arc_length_accumulated)

    if arc_length_accumulated >= LOOKAHEAD
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
function moveCar(output_path::Array{Float64}, planned_path::Array{Float64, 2}, car_position::Array{Float64})
  arc_length_accumulated::Float64 = 0.0

  #@printf("Car position before = ")
  #show(car_position)
  #@printf("\n")

  #@printf("Planned path = ")
  #show(planned_path)
  #@printf("\n")

  for i = 1:size(planned_path, 1)-1 
    arc_length_accumulated += sqrt((planned_path[i+1, 1] - planned_path[i, 1])^2 + (planned_path[i+1, 2] - planned_path[i, 2])^2)
    push!(output_path, planned_path[i+1, 1])
    push!(output_path, planned_path[i+1, 2])
    push!(output_path, planned_path[i+1, 3])

    #@printf("Arc length acc = %f\n", arc_length_accumulated)
    # Make sure to do this by reference, if we do an assignment
    # then the original car position won't be changed.
    car_position[1] = planned_path[i+1, 1]
    car_position[2] = planned_path[i+1, 2]
    car_position[3] = planned_path[i+1, 3]

    if arc_length_accumulated > CAR_STEP
      break
    end
     
  end 

  #@printf("Car position after = ")
  #show(car_position)
  #@printf("\n")

end

end # module
