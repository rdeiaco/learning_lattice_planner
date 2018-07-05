module LatticeCompressor

using Constants, PolynomialSpiral, LatticeState, DataStructures, Utils

export compressLattice

# Shrinks the lattice by replacing control actions with combinations
# of shorter control actions, within a certain cost threshold.
function compressLattice(control_action_lut::Array{Dict{UInt128, 
  SpiralControlAction}}, threshold::Float64)::Array{Dict{UInt128, SpiralControlAction}}

  # A threshold less than 1.0 doesn't make sense.
  @assert(threshold > 1.0)

  # The final set of control sets, after performing compression.
  # Initially for each heading they are all empty.
  compressed_lut::Array{Dict{UInt128, SpiralControlAction}} = []
  for ti in TI_RANGE
    push!(compressed_lut, Dict{UInt128, SpiralControlAction}()) 
  end 

  # We need to perform the decomposition for each possible heading
  # in the first quadrant for the xy origin.
  origin_list::Array{State} = []
  for angle in TI_RANGE
    push!(origin_list, State(0.0, 0.0, angle, 0.0, 0))
  end

  for origin in origin_list
    # Next, perform an expansion from the origin to collect all possible
    # predecessors of each node, in order to determine which actions can
    # potentially be removed.
    # The predecessor table contains the list of potential predecessors
    # of each vertex, as well as the associated control action.
    # The threshold table contains the threshold of each vertex found
    # during the search. Initially they are all the same value.
    # The cost table gives the current cost to reach a given vertex. 
    predecessor_table::Dict{UInt128, Array{Tuple{State, SpiralControlAction, SpiralControlAction}}},
      threshold_table::Dict{UInt128, Float64}, 
      cost_table::Dict{UInt128, Float64},
      control_action_max_heap::PriorityQueue{SpiralControlAction, Float64} = generatePredecessors(control_action_lut, threshold, origin)

    # Next, iterate through the max heap of all the used control actions,
    # reducing each one to component actions that satisfy the threshold.
    while !isempty(control_action_max_heap)
      control_action::SpiralControlAction = dequeue!(control_action_max_heap)
      state_f = State(control_action.xf, control_action.yf, control_action.tf, control_action.kf, control_action.line_segment_count) 
      state_f_id = getStateId(state_f)
      predecessor_list = predecessor_table[state_f_id]

      # If there is only one predecessor, then the control action itself
      # is the only way to reach the final node of said control action,
      # so it must be included in the final LUT.
      @assert(size(predecessor_list, 1) > 0)
      if size(predecessor_list, 1) == 1
        ti = predecessor_list[1][1].theta
        ti_mod = ti % (pi / 2.0)
        if abs(ti_mod - (pi / 2.0)) < 0.01
          ti_mod = 0.0
        end
        ti_index::Int64 = getClosestIndex(ti_mod, TI_RANGE)
        @assert(abs(ti_mod - TI_RANGE[ti_index]) < 0.01)
        @assert(abs(predecessor_list[1][2].ti - TI_RANGE[ti_index]) < 0.01)
        control_id = getControlId(predecessor_list[1][2])

        compressed_lut[ti_index][control_id] = predecessor_list[1][2]
        continue
      end

      # Otherwise, we need to iterate over the predecessors,
      # and add one of them to the final control set (if necessary).
      # As a heuristic, we iterate from the cheapest control action
      # to the most expensive (cost in terms of arc length).
      predecessor_min_heap::PriorityQueue{Tuple{State, SpiralControlAction, SpiralControlAction}, Float64} = PriorityQueue{Tuple{State, SpiralControlAction, SpiralControlAction}, Float64}(Base.Order.Forward)

      for predecessor in predecessor_list
        # If the the predecessor has a predecessor, then we need to include
        # both control action costs in our heap cost.
        if getControlId(predecessor[2]) == getControlId(predecessor[3])
          enqueue!(predecessor_min_heap, predecessor, predecessor[2].arc_length)
        else
          enqueue!(predecessor_min_heap, predecessor, predecessor[2].arc_length + predecessor[3].arc_length)
        end
      end

      adjusted::Bool = false
      while !isempty(predecessor_min_heap)
        pred_tuple::Tuple{State, SpiralControlAction, SpiralControlAction} = dequeue!(predecessor_min_heap)
        pred_state::State = pred_tuple[1]

        # If this predecessor action comes from the origin, it's not a useful decomposition,
        # so we ignore it.
        if getStateId(pred_state) == getStateId(origin)
          continue
        end

        pred_to_final_control_action::SpiralControlAction = pred_tuple[2]
        to_pred_control_action::SpiralControlAction = pred_tuple[3]
        pred_to_final_control_action_id = getControlId(pred_to_final_control_action)
        ti_mod = pred_to_final_control_action.ti % (pi / 2.0)
        if abs(ti_mod - (pi / 2.0)) < 0.01
          angle_mod = 0.0
        end
        ti_index = getClosestIndex(ti_mod, TI_RANGE)


        final_state::State = transformControlActionFromState(pred_tuple[2], pred_state)

        # Check if we have already added this control action to the
        # compressed control set.
        if haskey(compressed_lut[ti_index], pred_to_final_control_action_id)
          adjustThreshold(predecessor_table, threshold_table, cost_table, control_action_max_heap, pred_state, final_state, pred_to_final_control_action, to_pred_control_action)   
          adjusted = true
          break
        end
      end

      # If we performed an adjustment using the heuristic above,
      # this edge is taken care of and we can continue.
      if adjusted
        continue
      end

      # If none of the control actions were present in our compressed
      # control set, then we should find the one that minimizes the
      # cost total cost after decomposition and add it to the compressed
      # control set.
      min_cost::Float64 = Inf 
      min_pred_tuple = predecessor_list[1]
      for predecessor in predecessor_list
        temp_pred_state = predecessor[1]

        # If this predecessor action comes from the origin, it's not a useful decomposition,
        # so we ignore it.
        if getStateId(temp_pred_state) == getStateId(origin)
          continue
        end

        temp_cost::Float64 = cost_table[getStateId(temp_pred_state)] + predecessor[2].arc_length
        if temp_cost < min_cost
          min_pred_tuple = predecessor
          min_cost = temp_cost
        end
      end

      @assert(min_cost < Inf)

      # Adjust the threshold based on the minimum cost decomposition.
      final_state = transformControlActionFromState(min_pred_tuple[2], min_pred_tuple[1])
      adjustThreshold(predecessor_table, threshold_table, cost_table, control_action_max_heap, min_pred_tuple[1], final_state, min_pred_tuple[2], min_pred_tuple[3])

      ti_mod = min_pred_tuple[2].ti % (pi / 2.0)
      if abs(ti_mod - (pi / 2.0)) < 0.01
        ti_mod = 0.0
      end
      ti_index = getClosestIndex(ti_mod, TI_RANGE)
      @assert(abs(ti_mod - TI_RANGE[ti_index]) < 0.01)
      control_id = getControlId(min_pred_tuple[2])
      compressed_lut[ti_index][control_id] = min_pred_tuple[2]

    end 

  end

  return compressed_lut

end

# Performs a Dijkstra's elaboration of the control actions in
# order to see the potential predecessors of each node, and their cost. 
# Also populates a threshold table of each state found.
function generatePredecessors(control_action_lut::Array{Dict{UInt128, SpiralControlAction}}, threshold::Float64, origin::State)
  # Initialize the tables as empty.
  predecessor_table::Dict{UInt128, Array{Tuple{State, SpiralControlAction, SpiralControlAction}}} = Dict{UInt128, Array{Tuple{State, SpiralControlAction}}}()
  threshold_table::Dict{UInt128, Float64} = Dict{UInt128, Float64}()
  cost_table::Dict{UInt128, Float64} = Dict{UInt128, Float64}()
  control_action_max_heap::PriorityQueue{SpiralControlAction, Float64} = PriorityQueue{SpiralControlAction, Float64}(Base.Order.Reverse)

  # Table to tell us if a control action was found already.
  control_action_found_table::Dict{UInt128, UInt64} = Dict{UInt128, UInt64}()

  # Priority queue of State ID's, priority ordered by their
  # current accumulated cost.
  min_heap::PriorityQueue{Int128, Float64} = PriorityQueue{Int128, Float64}()

  # The hash table of nodes visited and settled by the algorithm.
  closed::Dict{UInt128, Int64} = Dict{Int128, Int64}()

  # Add the origin to the state heap.
  enqueue!(min_heap, getStateId(origin), 0.0)
  cost_table[getStateId(origin)] = 0.0

  while !isempty(min_heap)
    d_u::Float64 = peek(min_heap).second
    u_id::UInt128 = dequeue!(min_heap)
    u::State = getStateFromID(u_id)

    closed[u_id] = 1

    # This state has exceeded the maximum path length.
    if u.steps > PATH_LENGTH
      continue
    end

    # Based on the the angle of the current state u, figure out which
    # control set to iterate over.
    # Handle the numerical error of modulo to ensure that values slightly less
    # than pi/2 round to 0 after the modulo.
    angle_mod = u.theta % (pi / 2.0)
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
      delta_theta = u.theta - control_action.ti
      multiple = delta_theta / (pi / 2.0)
      @assert((round(multiple) - multiple) < 0.01)

      # To get the final state, rotate the control action by some multiple of
      # pi/2 to match the control action's ti to the state u's ti.
      v::State = transformControlActionEnd(control_action, delta_theta, u.x, u.y, u.steps)

      # Check to see if a point is outside the allowable workspace.
      if (v.x > GRID_X_END) || (v.x < GRID_X_START) || (v.y > GRID_Y_END) || (v.y < GRID_Y_START)
        continue
      end

      # Check to make sure the calculated x values are grid aligned.   
      @assert((round(v.x / RESOLUTION) - (v.x / RESOLUTION)) < 0.01)
      @assert((round(v.y / RESOLUTION) - (v.y / RESOLUTION)) < 0.01)
      @assert(v.theta - ((u.theta + control_action.tf - control_action.ti) % (2.0 * pi)) < 0.01)
      @assert(v.steps == u.steps + size(control_action.path, 1) - 1)

      # Calculate the predecessor state from the control action, to ensure
      # proper alignment with the grid.
      u_temp::State = getPredecessorState(control_action, v) 
      
      # Check to make sure the calculated predecessor state agrees with the actual
      # predecessor state.
      @assert(abs(u_temp.x - u.x) < 0.01)
      @assert(abs(u_temp.y - u.y) < 0.01)
      @assert(abs(u_temp.theta - u.theta) < 0.01)
      @assert(abs(u_temp.k - u.k) < 0.01)

      v_id::UInt128 = getStateId(v)

      # Check if the new state is on the closed table,
      # in which case it should be ignored.
      if get(closed, v_id, 0) == 1
        continue
      end

      # Get the cost from node u to node v.
      d_u_v::Float64 = control_action.arc_length

      # If we're coming from the origin, add the node to the
      # cost table and the threshold table.
      # Add the origin as a predecessor to the node v in the
      # predecessor table.
      # Add v to the heap.
      if u_id == getStateId(origin)
        cost_table[v_id] = d_u_v
        threshold_table[v_id] = threshold
        # The predecessor has no control action leading to it, as the
        # predecessor is the origin. In this case, just duplicate the
        # control action in both slots of the tuple.
        pred_tuple::Tuple{State, SpiralControlAction, SpiralControlAction} = (u, control_action, control_action)

        # Add this control action to the priority queue for decomposition.
        # Since this is the first layer, no control actions should be repeated.
        @assert(!haskey(control_action_found_table, getControlId(control_action)))
        control_action_found_table[getControlId(control_action)] = 1
        enqueue!(control_action_max_heap, control_action, control_action.arc_length)

        # This should be the first time we are visiting this node.
        @assert(!haskey(predecessor_table, v_id))
        pred_list = []
        push!(pred_list, pred_tuple)
        predecessor_table[v_id] = pred_list
        enqueue!(min_heap, v_id, d_u_v)

      # If we're not coming from the origin, we check to see
      # if the node v has been found before. If not, then it
      # doesn't have the origin as a predecessor, and should
      # be ignored. This is because we are at the 2nd layer of
      # Dijkstra if we've reached this step.
      else
        if get(cost_table, v_id, -1.0) >= 0.0
          pred_pred_list = predecessor_table[u_id]
          @assert(getStateId(origin) == getStateId(pred_pred_list[1][1]))
          to_pred_control_action = pred_pred_list[1][2]
          pred_tuple = (u, control_action, to_pred_control_action)
          pred_list = predecessor_table[v_id]
          push!(pred_list, pred_tuple)
          predecessor_table[v_id] = pred_list
        end
      end

    end # for control_action ...

  end # while min_heap ...

  return (predecessor_table, threshold_table, cost_table, control_action_max_heap)

end

# Adjusts the threshold values in the database of nodes depending
# on how the control action was replaced.
function adjustThreshold(predecessor_table::Dict{UInt128, Array{Tuple{State, 
  SpiralControlAction, SpiralControlAction}}}, threshold_table::Dict{UInt128, Float64}, 
  cost_table::Dict{UInt128, Float64}, 
  control_action_max_heap::PriorityQueue{SpiralControlAction, Float64},
  pred_state::State, final_state::State, pred_to_final_control_action::SpiralControlAction,
  to_pred_control_action::SpiralControlAction)

  d_pred::Float64 = cost_table[getStateId(pred_state)]
  d_final::Float64 = cost_table[getStateId(final_state)]
  d_pred_final::Float64 = pred_to_final_control_action.arc_length
  alpha::Float64 = (d_pred + d_pred_final) / d_final
  threshold_new = (threshold_table[getStateId(final_state)] * d_final - d_pred_final) / (alpha * d_final - d_pred_final)

  if threshold_new < threshold_table[getStateId(pred_state)]
    threshold_table[getStateId(pred_state)] = threshold_new
    pred_pred_array = predecessor_table[getStateId(pred_state)]

    # Delete the indices in the predecessor list of pred_state 
    # that violate the new threshold.
    to_delete::Array{UInt64} = []
    for i = 1:size(pred_pred_array, 1)
      pred_pred_temp = pred_pred_array[i]
      if getControlId(pred_pred_temp[2]) == getControlId(pred_pred_temp[3])
        temp_cost = cost_table[getStateId(pred_pred_temp[1])] + pred_pred_temp[2].arc_length
      else
        temp_cost = cost_table[getStateId(pred_pred_temp[1])] + pred_pred_temp[2].arc_length + pred_pred_temp[3].arc_length
      end
      if temp_cost > threshold_table[getStateId(pred_state)] * d_pred
        append!(to_delete, i)
      end

    end

    for index in to_delete
      deleteat!(pred_pred_array, index)
    end
    predecessor_table[getStateId(pred_state)] = pred_pred_array

    if cost_table[getStateId(pred_state)] >= cost_table[getStateId(final_state)]
      # If the pred_state is the origin then we should never reach here
      # since the origin cost is zero.
      enqueue!(control_action_max_heap, to_pred_control_action)
    end

  end

end

end # module
