module LatticePlanner

using LatticeState, PolynomialSpiral

using Constants, LatticeState, PolynomialSpiral, Geometry, Utils, 
DataStructures, LatticeOccupancyGrid

export planPath, heuristicCost, getResultScores

# Given a control action look up table, a heuristic look up table, 
# a swath look up table, and an occupancy grid, finds the optimal 
# path using A* search to a goal state.
function planPath(control_action_lut::Array{Dict{UInt128, SpiralControlAction}},
  heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}}, 
  swath_lut::Dict{UInt128, Set{Tuple{UInt64, UInt64}}}, 
  occupancy_grid::OccupancyGrid,
  goal::State)::Array{UInt128}

  # The best sequence of control_actions we can find.
  control_actions::Array{UInt128} = []

  # The min heap of states that we have opened.
  min_state_heap::PriorityQueue{UInt128, Float64} = PriorityQueue{UInt128, Float64}()

  # The table of states closed by the algorithm. 
  closed_table::Dict{UInt128, UInt64} = Dict{UInt128, UInt64}()

  # The table of predecessor action ID's for each state.
  predecessor_action_table::Dict{UInt128, UInt128} = Dict{UInt128, UInt128}()

  # The table of predecessor states for each state.
  predecessor_state_table::Dict{UInt128, UInt128} = Dict{UInt128, UInt128}()

  # The non-heuristic cost of each state.
  g_cost_table::Dict{UInt128, Float64} = Dict{UInt128, Float64}()

  origin::State = State(0.0, 0.0, 0.0, 0.0, 0)
  g_cost_table[getSpatialStateId(origin)] = 0.0

  # The origin cost estimate is purely heuristic.
  enqueue!(min_state_heap, getSpatialStateId(origin), 
    0.0 + heuristicCost(heuristic_lut, origin, goal))

  heap_count::UInt64 = 1

  @printf("\n")

  best_goal_score::Float64 = Inf
  best_goal::UInt128 = getSpatialStateId(origin)

  # A* loop over the opened states.
  while !isempty(min_state_heap)
    @printf("\rHeap Count = %d     ", heap_count)
  
    # Get the next node to expand and its cost. 
    u_id::UInt128 = dequeue!(min_state_heap)
    g_u::Float64 = g_cost_table[u_id]
    u::State = getStateFromID(u_id) 

    heap_count -= 1
    
    # This state is infeasible, and should not be expanded. 
    if g_u == Inf
      continue
    end

    # If u doesn't have zero steps, equality won't 
    # hold with the goal.
    @assert(u.steps == 0)

    # Close this node.
    closed_table[u_id] = 1

    u_obj_score::Float64 = objectiveScore(u, goal)
    u_goal_score::Float64 = g_u + u_obj_score
    if (u_goal_score < best_goal_score)
      best_goal_score = u_goal_score
      best_goal = u_id
    end

    # Based on the the angle of the current state u, figure out which
    # control set to iterate over.
    # Handle the numerical error of modulo to ensure that values slightly less
    # than pi/2 round to 0 after the modulo.
    angle_mod::Float64 = u.theta % (pi / 2.0)
    if abs(angle_mod - (pi / 2.0)) < 0.01
      angle_mod = 0.0
    end
    ti_index = getClosestIndex(angle_mod, TI_RANGE)
    @assert(abs(angle_mod - TI_RANGE[ti_index]) < 0.01)

    # Apply each control action to the current state.
    # Only apply control actions that correspond to the current heading of state u.
    for (control_id, control_action) in control_action_lut[ti_index]
      # If this control action doesn't correspond to the current state's
      # initial curvature, then skip it.
      if abs(u.k - control_action.ki) > 0.01
        continue
      end

      @assert(control_id == getControlId(control_action))

      # The control set will be rotated by some multiple of pi/2, since the control
      # set repeats for each quadrant. The difference between the control set angle
      # and the current state's angle must be a multiple of pi/2 to prevent a
      # misalignment with the discretization of the lattice.
      delta_theta::Float64 = u.theta - control_action.ti
      multiple::Float64 = delta_theta / (pi / 2.0)
      @assert((round(multiple) - multiple) < 0.01)

      # To get the final state, rotate the control action by some multiple of
      # pi/2 to match the control action's ti to the state u's ti.
      v::State = transformControlActionFromState(control_action, u)
      v.steps = 0

      # Check to see if a point is outside the allowable workspace.
      if (v.x > GRID_X_END) || (v.x < GRID_X_START) || (v.y > GRID_Y_END) || (v.y < GRID_Y_START)
        continue
      end

      # Check to make sure the calculated x values are grid aligned.   
      @assert((round(v.x / RESOLUTION) - (v.x / RESOLUTION)) < 0.01)
      @assert((round(v.y / RESOLUTION) - (v.y / RESOLUTION)) < 0.01)
      @assert(v.theta - ((u.theta + control_action.tf - control_action.ti) % (2.0 * pi)) < 0.01)

      # Calculate the predecessor state from the control action, to ensure
      # proper alignment with the grid.
      u_temp::State = getPredecessorState(control_action, v) 
      u_temp.steps = 0
      
      # Check to make sure the calculated predecessor state agrees with the actual
      # predecessor state.
      @assert(abs(u_temp.x - u.x) < 0.01)
      @assert(abs(u_temp.y - u.y) < 0.01)
      @assert(abs(u_temp.theta - u.theta) < 0.01)
      @assert(abs(u_temp.k - u.k) < 0.01)

      v_id::UInt128 = getSpatialStateId(v)

      # Check if the new state is on the closed table,
      # in which case it should be ignored.
      if get(closed_table, v_id, 0) == 1
        continue
      end

      # The cost from u to v is the sum of the arc length of the control
      # action and the cost of traversal through the occupancy grid.
      g_u_v::Float64 = control_action.arc_length + getCost(occupancy_grid, u,
        control_action, swath_lut)
      if g_u_v == Inf
        continue
      end

      # Compute the heuristic cost for this node.
      f_v_goal::Float64 = heuristicCost(heuristic_lut, v, goal)

      # If this node is worse than the best one found so far,
      # then there's no point pursuing it further.
      # TODO This is not entirely true, depending on the goal function,
      # and it's a bit of a hack.
      if g_u + g_u_v + f_v_goal > best_goal_score
        continue
      end

      # Check if the new state is already in the heap.
      # If it is, check to see if its cost should be
      # updated. If it is not in the heap, add it.
      if get(min_state_heap, v_id, -1.0) >= 0.0
        g_v::Float64 = g_cost_table[v_id]
        if (g_u + g_u_v) < g_v
          min_state_heap[v_id] = g_u + g_u_v + f_v_goal
          g_cost_table[v_id] = g_u + g_u_v
          predecessor_action_table[v_id] = control_id
          predecessor_state_table[v_id] = u_id
        end
        @assert((abs(min(g_v, g_u + g_u_v) - g_cost_table[v_id]) < 0.001) || g_cost_table[v_id] == Inf)
        @assert(g_cost_table[v_id] != Inf)
      else
        enqueue!(min_state_heap, v_id, g_u + g_u_v + f_v_goal)
        g_cost_table[v_id] = g_u + g_u_v
        heap_count += 1
        predecessor_action_table[v_id] = control_id
        predecessor_state_table[v_id] = u_id
        @assert((abs(g_u + g_u_v + f_v_goal - min_state_heap[v_id]) < 0.001) || min_state_heap[v_id] == Inf)
      end

    end # for control_action ...

  end # while min_heap ...

  # Return the predecessor control id's of the best node found.
  temp = best_goal 
  while temp != getSpatialStateId(origin)
    insert!(control_actions, 1, predecessor_action_table[temp])
    temp = predecessor_state_table[temp]  
  end

  return control_actions

end

# Calculates the heuristic cost from the origin to the goal state.
function heuristicCost(heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}},
  origin::State, goal::State)

  ti_mod::Float64 = origin.theta % (pi / 2.0)
  if abs(ti_mod - (pi / 2.0)) < 0.01
    ti_mod = 0.0
  end
  ti_index::UInt64 = getClosestIndex(ti_mod, TI_RANGE)

  goal_transformed::State = getDeltaState(origin, goal)

  # If a state is not in the heuristic LUT, fall back 
  # onto straight line distance.
  cost::Float64 = 0.0
  try
    cost = heuristic_lut[ti_index][getSpatialStateId(goal_transformed)][2]
  catch
    cost = sqrt(goal_transformed.x^2 + goal_transformed.y^2)
  end

  return cost

end

# Calculates how good a state found so far is as compared
# to the goal state.
# Squared distance is intentional, penalizes goals that
# are farther away.
function objectiveScore(state::State, goal::State)
  dist_sq::Float64 = (goal.x - state.x)^2 + (goal.y - state.y)^2
  delta_theta::Float64 = min(abs(goal.theta-state.theta), abs(goal.theta-(state.theta-2*pi)), abs((goal.theta-2*pi)-state.theta))
  angle_sq::Float64 = delta_theta^2

  return dist_sq + 20*angle_sq

end

function getResultScores(planned_path::Array{UInt128}, control_action_lut::Array{Dict{UInt128, SpiralControlAction}}, goal::State)::Tuple{Float64, Float64, Float64}
  state = State(0.0, 0.0, 0.0, 0.0, 0)

  dist_score = 0.0
  goal_score = 0.0 
  bending_energy_score = 0.0

  for i = 1:size(planned_path, 1)
    ti_mod::Float64 = state.theta % (pi / 2.0)
    ti_index = getClosestIndex(ti_mod, TI_RANGE)
    control_action::SpiralControlAction = control_action_lut[ti_index][planned_path[i]]

    state = transformControlActionFromState(control_action, state)
    dist_score += control_action.arc_length
    bending_energy_score += control_action.bending_energy
  end

  goal_score = sqrt((state.x-goal.x)^2+(state.y-goal.y)^2)

  return (dist_score, goal_score, bending_energy_score)

end 

end # module
