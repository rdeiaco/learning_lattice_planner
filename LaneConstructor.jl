module LaneConstructor

using Constants, Geometry, LatticeState

export generateCenterLine, generateLaneBounds

# Constructs a lane centerline in terms of a clothoid and a straightaway, or vice versa.
function generateCenterLine(curve_rate::Float64, curve_length::Float64, 
  straight_length::Float64, curve_first::Bool=true)::Array{Float64, 2}

  lane_centerline::Array{Float64, 2} = Array{Float64, 2}(0, 0)
  lane_spiral::Array{Float64, 2} = Array{Float64, 2}(0, 0)
  lane_straightaway::Array{Float64, 2} = Array{Float64, 2}(0, 0)

  # The order can vary depending on the input variable.
  if curve_first
    lane_spiral = sampleSpiral([0.0, curve_rate, 0.0, 
      0.0, curve_length], 0.0)[1]
    lane_straightaway = sampleStraightAway(straight_length, 
      lane_spiral[end, 3], lane_spiral[end, 1], lane_spiral[end, 2])
    # Remove the duplicate point (the end of the first part
    # is the start of the next).
    lane_straightaway = lane_straightaway[2:end, :]
    lane_centerline = vcat(lane_spiral, lane_straightaway)
  else
    lane_straightaway = sampleStraightAway(straight_length, 
      0.0, 0.0, 0.0)
    lane_spiral = sampleSpiral([0.0, curve_rate, 0.0, 
      0.0, curve_length], lane_straightaway[end, 3], lane_straightaway[end, 1], 
      lane_straightaway[end, 2])[1]
    # Remove the duplicate point (the end of the first part
    # is the start of the next).
    lane_spiral = lane_spiral[2:end, :]
    lane_centerline = vcat(lane_straightaway, lane_spiral)
  end
  
  return lane_centerline

end

# Constructs the lane bounds from the centerline.
function generateLaneBounds(path_vals::Array{Float64, 2}, lane_change::Bool=false,
  left_lane::Bool=true, lane_change_index::Int64=10, return_to_lane::Bool=false, truncate_path::Bool=false)::Tuple{Array{Float64, 2}, 
  Array{Float64, 2}, Array{Float64, 2}, State}
  left_bound::Array{Float64, 1} = Array{Float64, 1}()
  right_bound::Array{Float64, 1} = Array{Float64, 1}()
  
  goal_index::UInt64 = 0

  # Truncate the path to be smaller if this flag is set.
  if truncate_path
    path_size = min(size(path_vals, 1), LANE_LENGTH/PATH_RESOLUTION)
  else
    path_size = size(path_vals, 1)
  end

  for i = 1:path_size
    point = path_vals[i, :]
 
    x_left::Float64 = 0.0
    y_left::Float64 = 0.0
    x_right::Float64 = 0.0
    y_right::Float64 = 0.0

    #if lane_change && left_lane && (i >= lane_change_index)
    if lane_change && left_lane
      x_left = point[1] + 1.5 * LANE_WIDTH * cos(point[3] + pi/2)
      y_left = point[2] + 1.5 * LANE_WIDTH * sin(point[3] + pi/2)
    else
      x_left = point[1] + 0.5 * LANE_WIDTH * cos(point[3] + pi/2)
      y_left = point[2] + 0.5 * LANE_WIDTH * sin(point[3] + pi/2)
    end

    #if lane_change && !left_lane && (i >= lane_change_index)
    if lane_change && !left_lane
      x_right = point[1] + 1.5 * LANE_WIDTH * cos(point[3] - pi/2) 
      y_right = point[2] + 1.5 * LANE_WIDTH * sin(point[3] - pi/2) 
    else
      x_right = point[1] + 0.5 * LANE_WIDTH * cos(point[3] - pi/2) 
      y_right = point[2] + 0.5 * LANE_WIDTH * sin(point[3] - pi/2) 
    end

    # If the lane bounds exceed the occupancy grid, then we take
    # the previous point as the end of the lane.
    if (round(x_left) >= GRID_X_END) || (round(x_left) <= GRID_X_START) || (round(y_left) >= GRID_Y_END) || (round(y_left) <= GRID_Y_START) || (round(x_right) >= GRID_X_END) || (round(x_right) <= GRID_X_START) || (round(y_right) >= GRID_Y_END) || (round(y_right) <= GRID_Y_START)
      break
    end

    push!(left_bound, x_left)
    push!(left_bound, y_left)

    push!(right_bound, x_right)
    push!(right_bound, y_right)

    # The goal index is the last index of the path that doesn't
    # exceed the bounds of the occupancy grid.
    goal_index = i

  end

  # Goal state should be in the original lane.
  if return_to_lane
    goal_state = State(path_vals[goal_index, 1],
      path_vals[goal_index, 2],
      path_vals[goal_index, 3],
      0.0, 0)
  # Goal state should be in the left lane.
  elseif lane_change && left_lane
    goal_state = State(path_vals[goal_index, 1] + LANE_WIDTH * cos(path_vals[goal_index, 3] + pi/2),
      path_vals[goal_index, 2] + LANE_WIDTH * sin(path_vals[goal_index, 3] + pi/2),
      path_vals[goal_index, 3],
      0.0, 0)
  # Goal state should be in the right lane.
  elseif lane_change && !left_lane
    goal_state = State(path_vals[goal_index, 1] + LANE_WIDTH * cos(path_vals[goal_index, 3] - pi/2),
      path_vals[goal_index, 2] + LANE_WIDTH * sin(path_vals[goal_index, 3] - pi/2),
      path_vals[goal_index, 3],
      0.0, 0)
  # Otherwise, goal state is the end of the centerline.
  else
    goal_state = State(path_vals[goal_index, 1],
      path_vals[goal_index, 2],
      path_vals[goal_index, 3],
      0.0, 0)
  end

  

  return (permutedims(reshape(left_bound, 2, :), [2 1]), permutedims(reshape(right_bound, 2, :), [2 1]), path_vals[1:goal_index, :], goal_state)

end 

end # module
