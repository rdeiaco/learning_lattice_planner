module HLUTGenerator

using Constants, LatticeState, PolynomialSpiral, Geometry, Utils, 
DataStructures

export generateHLUT

# Generates the heuristic look-up table for a given
# input control set for the workspace of the car.
# The HLUT is a dictionary with keys corresponding
# to states in the workspace, and holds the path
# from the origin to those points, as well as the cost.
function generateHLUT(control_action_lut::Array{Dict{UInt128, 
  SpiralControlAction}})::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}}
  hlut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []

  for i = 1:size(TI_RANGE, 1)
    # The min heap of states that we have opened.
    min_state_heap::PriorityQueue{UInt128, Float64} = PriorityQueue{UInt128, Float64}()

    # The table of states closed by the algorithm. 
    closed_table::Dict{UInt128, UInt64} = Dict{UInt128, UInt64}()

    # The table of costs of the nodes we have closed.
    cost_table::Dict{UInt128, Float64} = Dict{UInt128, Float64}()

    # The table of predecessor action ID's for each state.
    predecessor_action_table::Dict{UInt128, UInt128} = Dict{UInt128, UInt128}()

    # The table of predecessor states for each state.
    predecessor_state_table::Dict{UInt128, UInt128} = Dict{UInt128, UInt128}()

    origin::State = State(0.0, 0.0, TI_RANGE[i], 0.0, 0)
    enqueue!(min_state_heap, getSpatialStateId(origin), 0.0)

    heap_count::UInt64 = 1

    @printf("\n")
    
    # Dijkstra's loop over the opened states.
    while !isempty(min_state_heap) 
      #@printf("\rHeap Count = %d     ", heap_count)
    
      # Get the next node to expand and its cost. 
      d_u::Float64 = peek(min_state_heap).second 
      u_id::UInt128 = dequeue!(min_state_heap)
      u::State = getStateFromID(u_id) 

      # We aren't concerned with the steps in the HLUT generation.
      @assert(u.steps == 0)

      heap_count -= 1

      # Close this node.
      closed_table[u_id] = 1
      cost_table[u_id] = d_u

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

        # Calculate the distance between u and v.
        d_u_v::Float64 = control_action.arc_length

        # Check if the new state is already in the heap.
        # If it is, check to see if its cost should be
        # updated. If it is not in the heap, add it.
        if get(min_state_heap, v_id, -1.0) >= 0.0
          d_v::Float64 = min_state_heap[v_id]
          if (d_u + d_u_v) < d_v
            min_state_heap[v_id] = d_u + d_u_v
            predecessor_action_table[v_id] = control_id
            predecessor_state_table[v_id] = u_id
          end
          @assert(abs(min(d_v, d_u + d_u_v) - min_state_heap[v_id]) < 0.001)
        else
          enqueue!(min_state_heap, v_id, d_u + d_u_v)
          heap_count += 1
          predecessor_action_table[v_id] = control_id
          predecessor_state_table[v_id] = u_id
          @assert(abs(d_u + d_u_v - min_state_heap[v_id]) < 0.001)
        end

      end # for control_action ...
    end # while min_state_heap ...
  
    lut::Dict{UInt128, Tuple{Array{UInt128}, Float64}} = Dict{UInt128, Tuple{Array{UInt128}, Float64}}()

    for state_id in keys(predecessor_action_table) 
      control_id_sequence::Array{UInt128} = []
      path_cost::Float64 = cost_table[state_id]

      temp::UInt128 = state_id
      while temp != getSpatialStateId(origin)
        insert!(control_id_sequence, 1, predecessor_action_table[temp])
        temp = predecessor_state_table[temp] 
      end

      lut[state_id] = (control_id_sequence, path_cost)
      
    end

    push!(hlut, lut)

  end

  @printf("\n")

  return hlut

end

end # module
