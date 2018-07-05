module ControlSetOptimizer

export optimizeControlSet

using Constants, LatticeState, PolynomialSpiral, LatticePathFinder, StatsBase

function optimizeControlSet(control_action_lut::Array{Dict{UInt128, SpiralControlAction}},
  paths::Array{Array{Float64, 2}}, lambda::Float64)::Array{Dict{UInt128, SpiralControlAction}}

  initial_control_lut::Array{Dict{UInt128, SpiralControlAction}} = deepcopy(control_action_lut)
  optimized_control_lut::Array{Dict{UInt128, SpiralControlAction}} = []

  # Start off with each control set only containing the basic move
  # straight control action for that specific direction.
  for i = 1:size(TI_RANGE, 1)
    push!(optimized_control_lut, Dict{UInt128, SpiralControlAction}())
    ti = TI_RANGE[i]
    xf = X_BASE_RANGE[i]
    yf = Y_BASE_RANGE[i]
    tf = TI_RANGE[i]
    control_action::SpiralControlAction = SpiralControlAction(xf, yf, ti, tf, 0.0, 0.0)
    optimized_control_lut[i][getControlId(control_action)] = control_action
    delete!(initial_control_lut[i], getControlId(control_action))
  end
  
  # We start off with one control action for each initial heading. 
  num_control_actions::UInt64 = size(TI_RANGE, 1)
  # Parameter that selects the number of paths randomly selected to evaluate.
  num_paths::UInt64 = 128
  # Parameter that controls how many control actions to evaluate for each
  # initial angle.
  num_search_actions::UInt64 = 128

  while (true)
    # First, randomly permute the ti indices to select the order in which we
    # will add control actions to the control set.
    ti_indices::Array{UInt64} = shuffle!(collect(1:size(TI_RANGE, 1)))

    # The randomly selected set of polygonal paths used for this iteration.
    path_indices::Array{UInt64} = sample(collect(1:size(paths, 1)), num_paths, replace=false)

    # Evaluate the objective, this will be our baseline to compare to. 
    objective::Float64 = evaluateObjective(optimized_control_lut, paths, lambda, 
      path_indices, num_control_actions) 

    @printf("Start Objective = %f\n", objective)

    # Iterate over each initial heading. If we can't find a control action
    # to improve the objective over all the indices, then we are done.
    best_objective::Float64 = Inf
    best_control_id::UInt128 = 0
    best_ti_index::UInt64 = size(TI_RANGE, 1) + 1

    for ti_index in ti_indices
    random_control_ids::Array{UInt128} = shuffle(collect(keys(initial_control_lut[ti_index])))

      # For this initial heading, iterate over the randomly selected control
      # actions, and evaluate the objective function.
      for i = 1:min(size(random_control_ids, 1), num_search_actions)
        action_id::UInt128 = random_control_ids[i]
        action::SpiralControlAction = initial_control_lut[ti_index][action_id]
        optimized_control_lut[ti_index][action_id] = action
#DEBUG
        temp_size = 0
        for lut in optimized_control_lut
          temp_size += length(keys(lut))
        end
#END DEBUG
        @assert(temp_size == num_control_actions + 1)

        temp_objective::Float64 = evaluateObjective(optimized_control_lut, paths, 
          lambda, path_indices, num_control_actions + 1)

        if temp_objective < best_objective
          best_objective = temp_objective
          best_control_id = action_id
          best_ti_index = ti_index
        end

        # Remove the temporarily added control action.
        delete!(optimized_control_lut[ti_index], action_id)
#DEBUG
        temp_size = 0
        for lut in optimized_control_lut
          temp_size += length(keys(lut))
        end
#END DEBUG
        @assert(temp_size == num_control_actions)
      end
    end

    @assert(best_control_id != 0)

    if best_objective < objective
      @printf("End Objective = %f\n", best_objective)
      action = initial_control_lut[best_ti_index][best_control_id]
      optimized_control_lut[best_ti_index][best_control_id] = action
      delete!(initial_control_lut[best_ti_index], best_control_id)
      num_control_actions += 1
#DEBUG
      temp_size = 0
      for lut in optimized_control_lut
        temp_size += length(keys(lut))
      end
#END DEBUG
      @assert(temp_size == num_control_actions)
    else
      @printf("No improvement found.\n")
      break
    end

  end

  return optimized_control_lut

end

function evaluateObjective(control_action_lut::Array{Dict{UInt128, SpiralControlAction}},
  paths::Array{Array{Float64, 2}}, lambda::Float64, path_indices::Array{UInt64},
  num_control_actions::UInt64)::Float64

  score::Float64 = 0.0


  for i in path_indices
    path = paths[i]
    result = findClosestPath(control_action_lut, path)
    score += result[1]
  end
  
  score = score / size(path_indices, 1) + lambda * num_control_actions

  return score
      
end

end # module
