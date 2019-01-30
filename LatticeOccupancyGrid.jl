module LatticeOccupancyGrid

using Constants, Geometry, LatticeState, PolynomialSpiral, LaneConstructor, SwathGenerator

export OccupancyGrid, getGridIndices, getGridPoint, getCost, getCostFromSwath

struct OccupancyGrid
  grid::Array{Float64, 2}
  center_line::Array{Float64, 2}
  goal::State

  function OccupancyGrid(curvature_rate::Float64, curve_length::Float64, straight_length::Float64, curve_first::Bool=true, lane_change::Bool=false, left_lane::Bool=true, lane_change_index::Int64=10, add_obstacle::Bool=true)
    grid::Array{Float64, 2}, center_line::Array{Float64, 2}, goal::State = populateGrid(curvature_rate, curve_length, straight_length, curve_first, lane_change, left_lane, lane_change_index, add_obstacle)
    new(grid, center_line, goal)
  end

  function OccupancyGrid(path::Array{Float64, 2}, lane_change::Bool=false, left_lane::Bool=true, lane_change_index::Int64=10, truncate_path::Bool=false)
    path_with_angles::Array{Float64, 2} = Array{Float64, 2}(size(path, 1), 3)

    path_with_angles[1, :] = [path[1, 1], path[1, 2], 0.0]
    for i = 2:size(path_with_angles, 1)
      dx::Float64 = path[i, 1] - path[i-1, 1]
      dy::Float64 = path[i, 2] - path[i-1, 2]
      angle::Float64 = atan2(dy, dx)
      path_with_angles[i, :] = [path[i, 1], path[i, 2], angle]
    end

    grid::Array{Float64, 2}, center_line::Array{Float64, 2}, goal::State = populateGridFromPath(path_with_angles, lane_change, left_lane, lane_change_index, truncate_path)
    new(grid, center_line, goal)
  end

end

function populateGridFromPath(path::Array{Float64, 2}, lane_change::Bool=false, left_lane::Bool=true, lane_change_index::Int64=10, add_obstacle::Bool=true, truncate_path::Bool=false)::Tuple{Array{Float64, 2}, Array{Float64, 2}, State}
  left_bound, right_bound, center_line, goal = generateLaneBounds(path, lane_change, left_lane, lane_change_index, false, truncate_path)

  grid::Array{Float64, 2} = zeros((round(UInt64, GRID_LENGTH/GRID_RESOLUTION), round(UInt64, GRID_WIDTH/GRID_RESOLUTION)))

  for i in 1:size(left_bound, 1)
    @assert(left_bound[i, 1] > GRID_X_START)
    @assert(left_bound[i, 1] < GRID_X_END)
    @assert(left_bound[i, 2] > GRID_Y_START)
    @assert(left_bound[i, 2] < GRID_Y_END)

    @assert(right_bound[i, 1] > GRID_X_START)
    @assert(right_bound[i, 1] < GRID_X_END)
    @assert(right_bound[i, 2] > GRID_Y_START)
    @assert(right_bound[i, 2] < GRID_Y_END)

    x_left_index, y_left_index = getGridIndices(left_bound[i, :])
    x_right_index, y_right_index = getGridIndices(right_bound[i, :])
  
    grid[x_left_index, y_left_index] = Inf
    grid[x_right_index, y_right_index] = Inf

  end

  return (grid, center_line, goal)

end

# Populates the occupancy grid with boundaries for a
# lane specified by a clothoid with the given curvature
# rate.
function populateGrid(curve_rate::Float64, curve_length::Float64, straight_length::Float64, curve_first::Bool=true, lane_change::Bool=false, left_lane::Bool=true, lane_change_index::Int64=10, add_obstacle::Bool=true)::Tuple{Array{Float64, 2}, Array{Float64, 2}, State}
  grid::Array{Float64, 2} = zeros((round(UInt64, GRID_LENGTH/GRID_RESOLUTION), round(UInt64, GRID_WIDTH/GRID_RESOLUTION)))

  # Sample the clothoid, and add points laterally offset from it to
  # the occupancy grid.
  path_vals = generateCenterLine(curve_rate, curve_length, straight_length, curve_first)
  left_bound, right_bound, center_line, goal = generateLaneBounds(path_vals, lane_change, left_lane, lane_change_index, true)
  @assert(lane_change_index + LANE_CHANGE_OBSTACLE_INDEX_GAP < size(center_line, 1))
  obstacle_indices = generateObstacle(center_line, left_lane, add_obstacle, lane_change_index)

  for i in 1:size(left_bound, 1)
    x_left_index, y_left_index = getGridIndices(left_bound[i, :])
    x_right_index, y_right_index = getGridIndices(right_bound[i, :])
    
    grid[x_left_index, y_left_index] = Inf
    grid[x_right_index, y_right_index] = Inf

  end

  for indices in obstacle_indices
    grid[indices[1], indices[2]] = Inf
  end

  return (grid, center_line, goal)

end

# Generates the occupancy grid indices from placing the base link of a car at a
# point offset from the centerline at index given by the lane_change_index
# plus a small gap LANE_CHANGE_OBSTACLE_INDEX_GAP.
function generateObstacle(center_line::Array{Float64, 2}, left_lane::Bool, add_obstacle::Bool, lane_change_index::Int64)::Set{Tuple{UInt64, UInt64}}
  obstacle_start_index::Int64 = lane_change_index + LANE_CHANGE_OBSTACLE_INDEX_GAP
  obstacle_base_point::Array{Float64} = center_line[obstacle_start_index, :]
  obstacle_indices::Set{Tuple{UInt64, UInt64}} = Set{Tuple{UInt64, UInt64}}()

  # If it is a left lane change, put the obstacle parked to the right
  # of the centerline, and have the opposite for a right lane change.
  if left_lane
    obstacle_base_point[1] += cos(obstacle_base_point[3] - pi/2) 
    obstacle_base_point[2] += sin(obstacle_base_point[3] - pi/2)
  else
    obstacle_base_point[1] += cos(obstacle_base_point[3] + pi/2)
    obstacle_base_point[2] += sin(obstacle_base_point[3] + pi/2)
  end
    
  x_offset::Float64 = 0.0
  while x_offset < CAR_LENGTH
    y_offset::Float64 = -CAR_WIDTH / 2.0
    while y_offset < CAR_WIDTH / 2.0
      x_point::Float64 = obstacle_base_point[1] + cos(obstacle_base_point[3]) * x_offset - sin(obstacle_base_point[3]) * y_offset
      y_point::Float64 = obstacle_base_point[2] + sin(obstacle_base_point[3]) * x_offset + cos(obstacle_base_point[3]) * y_offset
      indices::Tuple{UInt64, UInt64} = getGridIndices([x_point, y_point])
      
      push!(obstacle_indices, indices)
      
      y_offset += GRID_RESOLUTION

    end

    x_offset += GRID_RESOLUTION

  end

  return obstacle_indices

end

# Converts a cartesian point to a grid index.
function getGridIndices(point::Array{Float64})::Tuple{UInt64, UInt64}
  # +1 since 1-indexing.
  x_index::UInt64 = round(UInt64, (point[1] - GRID_X_START) / GRID_RESOLUTION) + 1
  y_index::UInt64 = round(UInt64, (point[2] - GRID_Y_START) / GRID_RESOLUTION) + 1
  
  return (x_index, y_index)

end

# Converts a grid index to a cartesian point.
function getGridPoint(indices::Tuple{UInt64, UInt64})::Array{Float64}
  x_point::Float64 = (indices[1] - 1) * GRID_RESOLUTION + GRID_X_START
  y_point::Float64 = (indices[2] - 1) * GRID_RESOLUTION + GRID_Y_START
  
  return [x_point, y_point]

end

# Calculates the occupancy grid cost of a control action from a
# start state.
function getCost(occupancy_grid::OccupancyGrid, start_state::State, 
  control_action::SpiralControlAction, 
  swath_lut::Dict{UInt128, Set{Tuple{UInt64, UInt64}}})::Float64

  # If the swath leaves the workspace, then it will throw an error.
  # In this case, we have an infeasible path.
  cost::Float64 = 0.0
  try
    transformed_swath = transformSwath(swath_lut[getControlId(control_action)], start_state, control_action)
    for indices in transformed_swath
      cost += occupancy_grid.grid[indices[1], indices[2]] 
    end  
  catch
    return Inf
  end

  return cost

end

# Calculates the occupancy grid cost of a raw swath set.
function getCostFromSwath(occupancy_grid, swath)::Float64
  cost::Float64 = 0.0
  try
    for indices in swath
      cost += occupancy_grid.grid[indices[1], indices[2]]
    end
  catch
    return Inf
  end

  return cost

end

end # module
