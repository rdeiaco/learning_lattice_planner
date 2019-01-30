module ControlSetOptimizer

export optimizeControlSet, optimizeControlSetClustered

using Constants, LatticeState, PolynomialSpiral, LatticePathFinder, StatsBase, LatticePathCluster, CPUTime, Utils

function optimizeControlSet(control_action_lut::Array{Dict{UInt128, SpiralControlAction}},
  paths::Array{Array{Array{Float64, 2}}}, lambda::Float64)::Array{Dict{UInt128, SpiralControlAction}}

  initial_control_lut::Array{Dict{UInt128, SpiralControlAction}} = deepcopy(control_action_lut)
  optimized_control_lut::Array{Dict{UInt128, SpiralControlAction}} = []

  max_control_action_size::Int64 = 0
  for i = 1:size(initial_control_lut, 1)
    for action in values(initial_control_lut[i])
      if action.line_segment_count > max_control_action_size
        max_control_action_size = action.line_segment_count
      end
    end
  end

  # Start off with each control set only containing the basic move
  # straight control action for that specific direction.
  for i = 1:size(TI_RANGE, 1)
    push!(optimized_control_lut, Dict{UInt128, SpiralControlAction}())
    ti = TI_RANGE[i]
    xf = X_BASE_RANGE[i]
    yf = Y_BASE_RANGE[i]
    tf = TI_RANGE[i]
    # TODO 0.2 should be replaced with a parameter, even though it doesn't
    # affect the control actions generated.
    control_action::SpiralControlAction = SpiralControlAction(xf, yf, ti, tf, 0.0, 0.0, 0.2)
    optimized_control_lut[i][getControlId(control_action)] = control_action
    delete!(initial_control_lut[i], getControlId(control_action))
  end
  
  # We start off with one control action for each initial heading. 
  num_control_actions::UInt64 = size(TI_RANGE, 1)
  # Parameter that selects the number of paths randomly selected to evaluate.
  num_paths::UInt64 = min(32, size(paths[1], 1))
  # Parameter that controls how many control actions to evaluate for each
  # initial angle.
  num_search_actions::UInt64 = 124

  termination_count = 0
  while (true)
    # First, randomly permute the ti indices to select the order in which we
    # will add control actions to the control set.
    ti_indices::Array{UInt64} = shuffle!(collect(1:size(TI_RANGE, 1)))

    # The randomly selected set of polygonal paths used for this iteration.
    path_indices::Array{UInt64} = sample(collect(1:size(paths[1], 1)), num_paths, replace=false)

    # Evaluate the objective, this will be our baseline to compare to. 
    objective::Float64 = evaluateObjective(optimized_control_lut, paths, lambda, 
      path_indices, num_control_actions) 
    #objective::Float64 = evaluateObjective2(optimized_control_lut, paths, lambda, 
      #path_indices, num_control_actions, max_control_action_size) 

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

        temp_objective::Float64 = evaluateObjective(optimized_control_lut, paths, lambda, 
          path_indices, num_control_actions + 1) 
        #temp_objective::Float64 = evaluateObjective2(optimized_control_lut, paths, 
        #  lambda, path_indices, num_control_actions + 1, max_control_action_size)

        if temp_objective < best_objective
          best_objective = temp_objective
          best_control_id = action_id
          best_ti_index = ti_index
        end

        # Remove the temporarily added control action.
        delete!(optimized_control_lut[ti_index], action_id)
      end
    end

    @assert(best_control_id != 0)

    if best_objective < objective
      @printf("End Objective = %f\n", best_objective)
      action = initial_control_lut[best_ti_index][best_control_id]
      optimized_control_lut[best_ti_index][best_control_id] = action
      delete!(initial_control_lut[best_ti_index], best_control_id)
      num_control_actions += 1
      termination_count = 0 
    else
      termination_count += 1
      @printf("No improvement found.\n")
      if termination_count == 2
        break
      end
    end

  end

  return optimized_control_lut

end

function optimizeControlSetClustered(control_action_lut::Array{Dict{UInt128, SpiralControlAction}},
  paths::Array{Array{Array{Float64, 2}}}, lambda::Float64, 
  cluster_sets::Array{Array{PathCluster}}, guided_search::Bool=false)::Array{Dict{UInt128, SpiralControlAction}}

  initial_control_lut::Array{Dict{UInt128, SpiralControlAction}} = deepcopy(control_action_lut)
  optimized_control_lut::Array{Dict{UInt128, SpiralControlAction}} = []

  max_control_action_size::Int64 = 0
  for i = 1:size(initial_control_lut, 1)
    for action in values(initial_control_lut[i])
      if action.line_segment_count > max_control_action_size
        max_control_action_size = action.line_segment_count
      end
    end
  end

  # Start off with each control set only containing the basic move
  # straight control action for that specific direction.
  for i = 1:size(TI_RANGE, 1)
    push!(optimized_control_lut, Dict{UInt128, SpiralControlAction}())
    ti = TI_RANGE[i]
    xf = X_BASE_RANGE[i]
    yf = Y_BASE_RANGE[i]
    tf = TI_RANGE[i]
    # TODO 0.2 should be replaced with a parameter, even though it doesn't
    # affect the control actions generated.
    control_action::SpiralControlAction = SpiralControlAction(xf, yf, ti, tf, 0.0, 0.0, 0.2)
    optimized_control_lut[i][getControlId(control_action)] = control_action
    delete!(initial_control_lut[i], getControlId(control_action))
  end
  
  # We start off with one control action for each initial heading. 
  num_control_actions::UInt64 = size(TI_RANGE, 1)
  # Parameter that selects the number of paths randomly selected to evaluate.
  num_paths::UInt64 = min(32, size(paths[1], 1))
  # Parameter that controls how many control actions to evaluate for each
  # initial angle.
  num_search_actions::UInt64 = 124

  # Initialize the guided search array of scores to a high value.
  # As the search proceeds, the values will get updated.
  cluster_scores::Array{Float64} = []
  if guided_search
    for i = 1:size(cluster_sets[1], 1)
      push!(cluster_scores, 10000)
    end
  end

  # The termination counts determine if an improvement is still probable
  # for that specific cluster.
  termination_count::Array{Float64} = zeros(size(cluster_scores, 1))

  while (true)
    # First, randomly permute the ti indices to select the order in which we
    # will add control actions to the control set.
    ti_indices::Array{UInt64} = shuffle!(collect(1:size(TI_RANGE, 1)))

    # If all of the scores are zeroed out, then there is no further progress to be made.
    if sum(cluster_scores) < 1e-5
      break
    end

    # Otherwise, select a cluster to learn from.
    if guided_search
      sample_array::Array{Float64} = cumsum(cluster_scores)
      sample_val = rand()*sample_array[end]
      for i = 1:size(sample_array, 1)
        if sample_val < sample_array[i]
          cluster_index = i
          break
        end 
      end
    else
      cluster_index = sample(collect(1:size(cluster_sets[1], 1)), 1, replace=false)
    end

    cluster_size = size(cluster_sets[1][cluster_index].paths, 1)
    if cluster_size < MAX_NUM_PATHS
      path_indices = collect(1:cluster_size)
    else
      path_indices = sample(collect(1:cluster_size), MAX_NUM_PATHS, replace=false)
    end

    # Evaluate the objective, this will be our baseline to compare to. 
    # If the cluster score is at its initialization value, need to run at least
    # one iteration to get a proper score.
    if cluster_scores[cluster_index] > 100.0
      objective = cluster_scores[cluster_index]
    else
      objective = evaluateClusteredObjective(optimized_control_lut, cluster_sets, 
        lambda, path_indices, cluster_index, num_control_actions) 
    end
    #objective::Float64 = evaluateObjective2(optimized_control_lut, cluster_set, lambda, 
      #path_indices, num_control_actions, max_control_action_size) 

    @printf("Start Objective = %f, Cluster Index = %d\n", objective, cluster_index)
    @printf("Cluster scores = ")
    show(cluster_scores)
    @printf("\n")

    # Iterate over each initial heading. If we can't find a control action
    # to improve the objective over all the indices, then we are done.
    best_objective::Float64 = Inf
    best_control_id::UInt128 = 0
    best_ti_index::UInt64 = size(TI_RANGE, 1) + 1

    for ti_index in ti_indices
      # Select a set of potential control actions to add to the control set.
      random_control_ids::Array{UInt128} = shuffle(collect(keys(initial_control_lut[ti_index])))

      # For this initial heading, iterate over the randomly selected control
      # actions, and evaluate the objective function.
      for i = 1:min(size(random_control_ids, 1), num_search_actions)
        action_id::UInt128 = random_control_ids[i]
        action::SpiralControlAction = initial_control_lut[ti_index][action_id]
        optimized_control_lut[ti_index][action_id] = action

        temp_objective::Float64 = evaluateClusteredObjective(optimized_control_lut, cluster_sets, 
          lambda, path_indices, cluster_index, num_control_actions + 1) 
        #temp_objective::Float64 = evaluateObjective2(optimized_control_lut, cluster_set, 
        #  lambda, path_indices, num_control_actions + 1, max_control_action_size)

        if temp_objective < best_objective
          best_objective = temp_objective
          best_control_id = action_id
          best_ti_index = ti_index
        end

        # Remove the temporarily added control action.
        delete!(optimized_control_lut[ti_index], action_id)
      end
    end

    @assert(best_control_id != 0)

    if best_objective < objective
      @printf("End Objective = %f\n", best_objective)

      # Update this cluster's score to reflect the newly added primitive.
      # This makes it less likely to be selected in the future if it is scoring
      # better than the other clusters.
      cluster_scores[cluster_index] = best_objective

      action = initial_control_lut[best_ti_index][best_control_id]
      optimized_control_lut[best_ti_index][best_control_id] = action
      delete!(initial_control_lut[best_ti_index], best_control_id)
      num_control_actions += 1
    else
      termination_count[cluster_index] += 1
      @printf("No improvement found.\n")
      if termination_count[cluster_index] == 2
        # We have reached the maximum number of iterations for this cluster,
        # It is likely that no further improvement is possible.
        # Zero out the cluster score to make it impossible for it to be selected.
        cluster_scores[cluster_index] = 0.0
      end
    end

  end

  return optimized_control_lut

end

function evaluateObjective(control_action_lut::Array{Dict{UInt128, SpiralControlAction}},
  paths::Array{Array{Array{Float64, 2}}}, lambda::Float64, path_indices::Array{UInt64},
  num_control_actions::UInt64)::Float64

  score::Float64 = 0.0

  for i in path_indices
    for j = 1:size(TI_RANGE, 1)
      path = paths[j][i]
      result = findClosestPath(control_action_lut, path, TI_RANGE[j])
      score += result[1]
    end
  end

  score = score / size(path_indices, 1) / size(TI_RANGE, 1) + lambda * num_control_actions

  return score
      
end

function evaluateObjective2(control_action_lut::Array{Dict{UInt128, SpiralControlAction}},
  paths::Array{Array{Array{Float64, 2}}}, lambda::Float64, path_indices::Array{UInt64},
  num_control_actions::UInt64, max_control_action_size::Int64)::Float64

  score::Float64 = 0.0

  for i in path_indices
    for j = 1:size(TI_RANGE, 1)
      path = paths[j][i]
      result = findClosestPath2(control_action_lut, path, TI_RANGE[j], max_control_action_size)
      score += result[1]
    end
  end
  
  score = score / size(path_indices, 1) / size(TI_RANGE, 1) + lambda * num_control_actions

  return score
      
end

function evaluateClusteredObjective(control_action_lut::Array{Dict{UInt128, SpiralControlAction}},
  cluster_sets::Array{Array{PathCluster}}, lambda::Float64, path_indices::Array{Int64}, cluster_index::Int64, 
  num_control_actions::UInt64)::Float64

  score::Float64 = 0.0

  path_count = 0
  for j in path_indices
    path = cluster_sets[1][cluster_index].paths[j]
    delta_x = path[2, 1] - path[1, 1]
    delta_y = path[2, 2] - path[1, 2]
    ti = TI_RANGE[getClosestIndex(atan2(delta_y, delta_x), TI_RANGE)]

    result = findClosestPath(control_action_lut, path, ti)
    score += result[1]
    path_count += 1
  end

  score = score / path_count + lambda * num_control_actions

  return score
      
end

end # module
