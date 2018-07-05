module LatticePathFinder

using Constants, LatticeState, PolynomialSpiral, Geometry, Utils, DataStructures

export findClosestPath

function findClosestPath(control_action_lut::Array{Dict{UInt128, SpiralControlAction}}, 
  path::Array{Float64, 2})
  # Priority queue of State ID's, priority ordered by their
  # current accumulated cost.
  min_heap::PriorityQueue{Int128, Float64} = PriorityQueue{Int128, Float64}()

  # The hash table of nodes visited and settled by the algorithm.
  closed::Dict{UInt128, Int64} = Dict{Int128, Int64}()

  # The hash table of the predecessor action that led to this state.
  predecessor_actions::Dict{UInt128, UInt128} = Dict{UInt128, UInt128}()

  # The hash table for the predecessor state of each state.
  predecessor_states::Dict{UInt128, UInt128} = Dict{UInt128, UInt128}() 

  # The terminal node of the best path found so far.
  # The best path must be at least as long as the polygonal
  # path to be considered valid.
  best_node::UInt128 = 0
  best_dl::Float64 = Inf

  origin = State(0.0, 0.0, 0.0, 0.0, 0)
  enqueue!(min_heap, getStateId(origin), 0.0)

  heap_count::Int64 = 1

  # Next, perform a Dijkstra-like search on the rest of the nodes
  # in the heap.
  while !isempty(min_heap)
    #@printf("\rHeap Count = %i               ", heap_count)

    # Need to keep track of the cost as well,
    # so save the value before dequeueing.
    dl_u::Float64 = peek(min_heap).second
    u_id::UInt128 = dequeue!(min_heap) 
    u = getStateFromID(u_id) 
    @assert(getStateId(u) == u_id)

    heap_count -= 1

    # Close this node.
    closed[u_id] = 1

    # If this node is at the end of a path that is at least
    # as long as the polygonal path, and has lattice distance
    # less than the best path found so far, it is the new
    # best path.
    # Even if it isn't, it exceeds the length of the polygonal
    # path, so no further control actions should be taken.
    if u.steps >= (size(path, 1) - 1)
      if dl_u < best_dl 
        best_dl = dl_u
        best_node = u_id
        #@printf("New best node = %i\n", best_node)
        #@printf("New best dl = %f\n", best_dl)
        #@printf("Path segments = %i\n", size(path, 1) - 1)
        #@printf("Lattice path segments = %i\n", u.steps)
      end
      continue
    end

    # If, after popping this node from the heap, the
    # cost to reach it is worse than the best path
    # we've found so far, then all paths out of this
    # node will be worse than our best path. Therefore
    # we do not need to expand it.
    if dl_u > best_dl
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
      try
        @assert((round(multiple) - multiple) < 0.01)
      catch
        @printf("\n")
        show(u)
        @printf("\n")
        show(control_action)
        @printf("\n")
        exit()
      end
        

      # To get the final state, rotate the control action by some multiple of
      # pi/2 to match the control action's ti to the state u's ti.
      v::State = transformControlActionEnd(control_action, delta_theta, u.x, u.y, u.steps)

      # Check to see if a point is outside the allowable workspace.
      if (v.x > GRID_X_END) || (v.x < GRID_X_START) || (v.y > (GRID_Y_END / 2.0)) || (v.y < (GRID_Y_START / 2.0))
        continue
      end

      # Check to make sure the calculated x values are grid aligned.   
      try
        @assert((round(v.x / RESOLUTION) - (v.x / RESOLUTION)) < 0.01)
        @assert((round(v.y / RESOLUTION) - (v.y / RESOLUTION)) < 0.01)
        @assert(v.theta - ((u.theta + control_action.tf - control_action.ti) % (2.0 * pi)) < 0.01)
        @assert(v.steps == u.steps + size(control_action.path, 1) - 1)
      catch
        @printf("\n")
        show(u)
        @printf("\n")
        show(v)
        @printf("\n")
        show(control_action)
        @printf("\n")
        exit()
      end

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

      # The code in this conditional should never get run...
      #if v.steps > size(path, 1)
      #  continue
      #end

      # Check if the new state is on the closed table,
      # in which case it should be ignored.
      if get(closed, v_id, 0) == 1
        continue
      end

      # Calculate the lattice distance between this edge and the
      # polygonal path.
      dl_u_v::Float64 = getLatticeDistance(path, u, v, control_action)

      # If taking this edge causes the lattice cost to exceed
      # our best path so far, then the path through u to this
      # node and all paths that lead out of it will be more
      # expensive than the best path we've found so far.
      if dl_u_v > best_dl
        continue
      end

      # Check if the new state is already in the heap.
      # If it is, check to see if its cost should be
      # updated. If it is not in the heap, add it.
      if get(min_heap, v_id, -1.0) >= 0.0
        dl_v::Float64 = min_heap[v_id]
        if max(dl_u, dl_u_v) < dl_v
          min_heap[v_id] = max(dl_u, dl_u_v)
          predecessor_actions[v_id] = control_id
          predecessor_states[v_id] = u_id
        end
        @assert(abs(min(dl_v, max(dl_u, dl_u_v)) - min_heap[v_id]) < 0.001)

      else
        enqueue!(min_heap, v_id, max(dl_u, dl_u_v))
        heap_count += 1
        predecessor_actions[v_id] = control_id
        predecessor_states[v_id] = u_id
        @assert(abs(max(dl_u, dl_u_v) - min_heap[v_id]) < 0.001)
      end

    end # for control_action ...
  end # while min_heap ...

  # Reconstruct the best path from the backpointers of the
  # best node found.
  best_lattice_path::Array{UInt128} = []
  best_control_action_sequence::Array{UInt128} = []
  temp = best_node
  origin = State(0.0, 0.0, 0.0, 0.0, 0) 
  origin_id = getStateId(origin)

  while temp != origin_id
    insert!(best_lattice_path, 1, temp)
    insert!(best_control_action_sequence, 1, predecessor_actions[temp])
    temp = predecessor_states[temp]
  end
  # Add in the origin.
  insert!(best_lattice_path, 1, temp)
      
  return (best_dl, best_lattice_path, best_control_action_sequence)

end

# Calculates the lattice distance from the polygonal path 
# to a u-v edge in the lattice.
function getLatticeDistance(path::Array{Float64, 2},
  u::State, v::State, control_action::SpiralControlAction, debug::Bool=false)::Float64
  # Arrays are 1-indexed, need to add 1.
  start_steps::Int64 = u.steps + 1

  # We stop calculating lattice distance when the number of
  # accumulated segments exceeds the number of segments on
  # the path. The number of line segments on the path is the
  # number of points on the path minus 1.
  end_steps::Int64 = min(v.steps, size(path, 1) - 1)

  if debug
    @printf("start_steps = %d\n", start_steps)
    @printf("end_steps = %d\n", end_steps)
  end

  max_edge_dist::Float64 = 0.0
  for i in start_steps:end_steps
    u1_v::Array{Float64} = transformControlActionPoint(control_action, u, i - u.steps)
    u2_v::Array{Float64} = transformControlActionPoint(control_action, u, i + 1 - u.steps)
    v1_v::Array{Float64} = path[i, 1:2]
    v2_v::Array{Float64} = path[i+1, 1:2]
    max_edge_dist = max(max_edge_dist, getMaxLineSegmentDistance(u1_v, u2_v, v1_v, v2_v, debug))
  end

  return max_edge_dist

end

end # module
