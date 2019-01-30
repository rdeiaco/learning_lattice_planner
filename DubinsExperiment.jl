module DubinsExperiment

#using Constants, LatticeState, DubinsPath, PyPlot, LatticeVisualizer, ControlActionGenerator, ParameterComparison, OptimizationPlanner, LatticeOccupancyGrid, PathExtractor, SwathGenerator, HLUTGenerator, LatticeCompressor, CPUTime, LatticePlanner

using Constants, ControlActionGenerator, LatticeState, LatticeVisualizer, DubinsPath,
LatticePathFinder, Geometry, PathExtractor, PathReader, ControlSetOptimizer, 
SwathGenerator, PathConstructor, LatticeCompressor, LatticeOccupancyGrid, HLUTGenerator,
Utils, JLD, LatticePlanner, CPUTime, LatticePathCluster, OptimizationPlanner, StatsBase, ParameterComparison, Distances, PyPlot, DubinsPlanner, PyCall

@pyimport iisignature

export dubinsExperiment

function dubinsExperiment()

  PASSIVE_CURVATURE = 0.2
  AGGRESSIVE_CURVATURE = 0.4

  # Generate the passive set.
  passive_control_action_lut = generateControlActions(PASSIVE_CURVATURE)
  extractCurvatureFromDubinsControlSet(passive_control_action_lut, "Passive")
  plotControlSetLUT(passive_control_action_lut, "Passive Control Set")

  # For each reachable control action in the passive set, generate an
  # aggressive version of it.
  aggressive_control_action_lut::Array{Dict{UInt128, DubinsAction}} = []
  for i = 1:size(TI_RANGE, 1)
    push!(aggressive_control_action_lut, Dict{UInt128, DubinsAction}())
    ti = TI_RANGE[i]

    for control_action in values(passive_control_action_lut[i])
      aggressive_action = DubinsAction(control_action.xf, control_action.yf, 
        control_action.ti, control_action.tf, 0.0, 0.0, AGGRESSIVE_CURVATURE)
      
      if aggressive_action.feasible
        aggressive_control_action_lut[i][getControlId(aggressive_action)] = aggressive_action 
      else
        @printf("Infeasible control action found\n")
        exit()
      end
    end
  end

  extractCurvatureFromDubinsControlSet(aggressive_control_action_lut, "Aggressive")
  plotControlSetLUT(aggressive_control_action_lut, "Aggressive Control Set")

  # Merge the two sets.
  dense_control_action_lut::Array{Dict{UInt128, DubinsAction}} = Array{Dict{UInt128, DubinsAction}}(size(aggressive_control_action_lut, 1))
  for i = 1:size(dense_control_action_lut, 1)
    dense_control_action_lut[i] = merge(aggressive_control_action_lut[i], passive_control_action_lut[i]) 
  end

  dense_control_set_size::UInt64 = 0
  for i in 1:size(TI_RANGE, 1)
    dense_control_set_size += length(dense_control_action_lut[i])
  end

  # Calculate the amount of aggressive and passive control actions in the set.
  passive_count = 0
  aggressive_count = 0
  for i in 1:size(TI_RANGE, 1)
    for control_id in keys(dense_control_action_lut[i])
      if (control_id >> 70) == round(PASSIVE_CURVATURE / CURVATURE_RESOLUTION)
        passive_count += 1
      elseif (control_id >> 70) == round(AGGRESSIVE_CURVATURE / CURVATURE_RESOLUTION)
        aggressive_count += 1
      else
        @printf("Error, control action with invalid curvature found.\n")
        @printf("Curvature = %f\n", (control_id >> 70) * CURVATURE_RESOLUTION)
        exit()
      end
    end
  end

  @printf("Generating paths...\n")
  num_generated_paths::Int64 = 100
  gen_paths::Array{Array{Float64, 2}} = Array{Array{Float64, 2}}(num_generated_paths)
  occupancy_grids::Array{OccupancyGrid} = Array{OccupancyGrid}(num_generated_paths)
  gen_curvature_vals::Array{Float64} = []
  gen_sigs::Array{Array{Float64}} = Array{Array{Float64}}(num_generated_paths)

  # Curvature Distribution Learning Code
  i::Int64 = 1
  while i <= num_generated_paths

    curvature_rate::Float64 = rand()*0.001
    if rand() > 0.5
      curvature_rate = -curvature_rate
    end

    left_lane::Bool = rand() > 0.5
    curve_split = 0.8
    lane_change_index = rand(175:225)

    occupancy_grid_gen::OccupancyGrid = OccupancyGrid(curvature_rate, 60.0*curve_split, 60.0*(1-curve_split), true, true, left_lane, lane_change_index, true)

    try
      result = planDubinsPath(occupancy_grid_gen.center_line, occupancy_grid_gen, 0.2, 16.0) 
    catch
      @printf("error in optimization.\n")
      continue
    end

    planned_path = result[1]
    append!(gen_curvature_vals, result[2])
    gen_paths[i] = planned_path
    occupancy_grids[i] = occupancy_grid_gen
    gen_sigs[i] = iisignature.sig(planned_path[:, 1:2], 4)
    i += 1

  end

  # Extract curvature distribution from generated path set.
  fig = figure()
  ax = axes()
  result = plt[:hist](gen_curvature_vals, bins=500, range=(-1.0, 1.0), normed=true)
  xlabel("Curvature")
  ylabel("Frequency")
  title("Generated Double Swerve Histogram")
  density = result[1]

  for j = 1:size(density, 1)
    density[j] = density[j] / 100.0
    if density[j] < 1e-10
      density[j] = 1e-10
    end
  end

  savefig("Generated Double Swerve Histogram.png")

  clf()
  close()

  test_curvature_distribution = density

  # Extract training/test sets.
  training_set, test_set, test_set_processed, raw_grid_indices = extractGeneratedPaths(gen_paths, 0.85)

  # Generate the swath lut.
  @printf("Generating Swath LUT...\n")
  swath_lut::Dict{UInt128, Set{Tuple{UInt64, UInt64}}} = generateSwaths(dense_control_action_lut) 
  @printf("Swath LUT generated.\n")

  # Load/generate the dense heuristic LUT.
  @printf("Building dense HLUT...\n")
  dense_heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = generateHLUT(dense_control_action_lut)

  data_log_array::Array{Float64, 2} = Array{Float64, 2}(6, 5)
  # Plan a path using the dense control set.
  @printf("Planning path using dense control set...\n")
  total_dl_score = 0.0
  total_time = 0.0
  dense_planned_paths::Array{Array{Float64, 2}} = []
  dense_path_control_actions::Array{Array{UInt128}} = []
  dense_sigs::Array{Array{Float64}} = []
  for i = 1:size(test_set, 1)
    #occupancy_grid::OccupancyGrid = OccupancyGrid(test_set[i], false)
    #occupancy_grid::OccupancyGrid = OccupancyGrid(test_set[i], true, lane_change_vals[i]<0.5, 10, true)
    occupancy_grid::OccupancyGrid = occupancy_grids[raw_grid_indices[i]]
    CPUtic()
    dense_planned_path::Array{UInt128} = 
      planPath(dense_control_action_lut, dense_heuristic_lut, 
      swath_lut, occupancy_grid, occupancy_grid.goal)
    total_time += CPUtoc()
    push!(dense_path_control_actions, dense_planned_path)
    push!(dense_planned_paths, getPathFromControlActions(dense_planned_path, dense_control_action_lut))

    #plotControlActionsOnOccupancy(swath_lut, dense_control_action_lut, dense_planned_path, 
    #  occupancy_grid, string("Dense Set Planned Path ", string(i)))
  end

  sig_score::Float64 = 0.0
  for i = 1:size(dense_planned_paths, 1)
    push!(dense_sigs, iisignature.sig(dense_planned_paths[i], 4))
    sig_score += cosine_dist(dense_sigs[i], gen_sigs[raw_grid_indices[i]])
  end
  @printf("Dense signature score = %f\n", sig_score)

  CPUtic()
  for i = 1:size(test_set_processed, 1)
    dl_res = findClosestPath(dense_control_action_lut, test_set_processed[i], 0.0)
    dl_score = dl_res[1]
    total_dl_score += dl_score
  end
  @printf("Dense set dl check time = %f\n", CPUtoc())
  
  dense_curvature_distribution = extractMengerCurvatureDistribution(dense_planned_paths, "Dense Set")

  @printf("Menger KL divergence = %f\n", kl_divergence(test_curvature_distribution, dense_curvature_distribution))

  dense_curvature_distribution = extractCurvatureFromDubinsControlSet(dense_control_action_lut, "Dense")

  @printf("Control Set KL divergence = %f\n", kl_divergence(test_curvature_distribution, dense_curvature_distribution))

  @printf("Dense control set average dl score = %f\n", total_dl_score / size(test_set_processed, 1))
  @printf("Dense control set average planning time = %f\n", total_time / size(test_set, 1))
  data_log_array[1, 4] = total_time / size(test_set, 1)
  @printf("Dense Control Set Size = %d\n", dense_control_set_size)
  data_log_array[1, 5] = dense_control_set_size
  @printf("Dense control set aggressive size = %d\n", aggressive_count)
  @printf("Dense control set passive size = %d\n", passive_count)

  # Free some memory.
  @printf("Freeing dense heuristic lut...\n")
  dense_heuristic_lut = []
  gc()

  # Load/generate a lattice compressed via Pivtoraiko's D* algorithm.
  # Note that this algorithm cannot handle duplicate actions, so we use the aggressive control
  # set as the base set to compress (since it reaches the endpoints in lower arc length than the
  # passive set). 
  @printf("Generating Pivtoraiko control action lut...")
  CPUtic()
  pivtoraiko_control_action_lut::Array{Dict{UInt128, DubinsAction}} = compressLattice(aggressive_control_action_lut, 2.0)
  time = CPUtoc()
  @printf("Pivtoraiko generation time = %f.\n", time)

  pivtoraiko_control_set_size::UInt64 = 0
  for i in 1:size(TI_RANGE, 1)
    pivtoraiko_control_set_size += length(pivtoraiko_control_action_lut[i])
  end

  # Calculate the amount of aggressive and passive control actions in the set.
  passive_count = 0
  aggressive_count = 0
  for i in 1:size(TI_RANGE, 1)
    for control_id in keys(pivtoraiko_control_action_lut[i])
      if (control_id >> 70) == round(PASSIVE_CURVATURE / CURVATURE_RESOLUTION)
        passive_count += 1
      elseif (control_id >> 70) == round(AGGRESSIVE_CURVATURE / CURVATURE_RESOLUTION)
        aggressive_count += 1
      else
        @printf("Error, control action with invalid curvature found.\n")
        @printf("Curvature = %f\n", (control_id >> 70) * CURVATURE_RESOLUTION)
        exit()
      end
    end
  end

  # Load/generate the Pivtoraiko heuristic LUT.
  pivtoraiko_heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []
  @printf("Building Pivtoraiko HLUT...\n")
  pivtoraiko_heuristic_lut = generateHLUT(pivtoraiko_control_action_lut)

  # Plan a path using the Pivtoraiko control set.
  @printf("Planning path using Pivtoraiko control set...\n")
  total_dl_score = 0.0
  total_time = 0.0
  pivtoraiko_planned_paths::Array{Array{Float64, 2}} = []
  pivtoraiko_path_control_actions::Array{Array{UInt128}} = []
  pivtoraiko_sigs::Array{Array{Float64}} = []
  for i = 1:size(test_set, 1)
    #occupancy_grid::OccupancyGrid = OccupancyGrid(test_set[i], false)
    #occupancy_grid::OccupancyGrid = OccupancyGrid(test_set[i], true, lane_change_vals[i]<0.5, 10, true)
    occupancy_grid::OccupancyGrid = occupancy_grids[raw_grid_indices[i]]
    CPUtic()
    pivtoraiko_planned_path::Array{UInt128} = 
      planPath(pivtoraiko_control_action_lut, pivtoraiko_heuristic_lut, 
      swath_lut, occupancy_grid, occupancy_grid.goal)
    total_time += CPUtoc()
    push!(pivtoraiko_path_control_actions, pivtoraiko_planned_path)
    push!(pivtoraiko_planned_paths, getPathFromControlActions(pivtoraiko_planned_path, pivtoraiko_control_action_lut))

    
    #plotControlActionsOnOccupancy(swath_lut, pivtoraiko_control_action_lut, pivtoraiko_planned_path, 
    #  occupancy_grid, string("Pivtoraiko Set Planned Path ", string(i)))
  end

  sig_score = 0.0
  for i = 1:size(dense_planned_paths, 1)
    push!(pivtoraiko_sigs, iisignature.sig(pivtoraiko_planned_paths[i], 4)) 
    sig_score += cosine_dist(pivtoraiko_sigs[i], gen_sigs[raw_grid_indices[i]])
  end
  @printf("Pivtoraiko signature score = %f\n", sig_score)

  CPUtic()
  for i = 1:size(test_set_processed, 1)
    dl_res = findClosestPath(pivtoraiko_control_action_lut, test_set_processed[i], 0.0)
    dl_score = dl_res[1]
    total_dl_score += dl_score
  end
  @printf("Pivtoraiko set dl check time = %f\n", CPUtoc())

  pivtoraiko_curvature_distribution = extractMengerCurvatureDistribution(pivtoraiko_planned_paths, "DL Set")
  @printf("Menger KL divergence = %f\n", kl_divergence(test_curvature_distribution, pivtoraiko_curvature_distribution))

  pivtoraiko_curvature_distribution = extractCurvatureFromDubinsControlSet(pivtoraiko_control_action_lut, "DL")

  @printf("Control Set KL divergence = %f\n", kl_divergence(test_curvature_distribution, pivtoraiko_curvature_distribution))

  @printf("Pivtoraiko control set average dl score = %f\n", total_dl_score / size(test_set_processed, 1))
  @printf("Pivtoraiko control set average planning time = %f\n", total_time / size(test_set, 1))
  data_log_array[2, 4] = total_time / size(test_set, 1)
  @printf("Pivtoraiko Control Set Size = %d\n", pivtoraiko_control_set_size)
  data_log_array[2, 5] = pivtoraiko_control_set_size
  @printf("DL control set aggressive size = %d\n", aggressive_count)
  @printf("DL control set passive size = %d\n", passive_count)

  # Free some memory.
  @printf("Freeing Pivtoraiko lut...\n")
  pivtoraiko_control_action_lut = []
  pivtoraiko_heuristic_lut = []
  gc()

  optimized_control_action_lut_1::Array{Dict{UInt128, DubinsAction}} = []
  optimized_heuristic_lut_1::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []
  optimized_control_action_lut_2::Array{Dict{UInt128, DubinsAction}} = []
  optimized_heuristic_lut_2::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []
  optimized_control_action_lut_3::Array{Dict{UInt128, DubinsAction}} = []
  optimized_heuristic_lut_3::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []
  optimized_control_action_lut_4::Array{Dict{UInt128, DubinsAction}} = []
  optimized_heuristic_lut_4::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []

  @printf("Optimizing control sets...\n")
  @printf("Optimized control action lut 3 not present. Rebuilding...\n")
  CPUtic()
  optimized_control_action_lut_3 = optimizeControlSet(dense_control_action_lut, training_set, 0.001)
  time = CPUtoc()
  @printf("Opt 3 generation time = %f.\n", time)
  # Plot the control set.
#  plotControlSetLUT(optimized_control_action_lut_3, "Lambda 0.001 Control Set")
  
  temp_size = 0
  for i = 1:size(optimized_control_action_lut_3, 1)
    temp_size += size(collect(keys(optimized_control_action_lut_3[i])), 1)
  end

  # Calculate the amount of aggressive and passive control actions in the set.
  passive_count = 0
  aggressive_count = 0
  for i in 1:size(TI_RANGE, 1)
    for control_id in keys(optimized_control_action_lut_3[i])
      if (control_id >> 70) == round(PASSIVE_CURVATURE / CURVATURE_RESOLUTION)
        passive_count += 1
      elseif (control_id >> 70) == round(AGGRESSIVE_CURVATURE / CURVATURE_RESOLUTION)
        aggressive_count += 1
      else
        @printf("Error, control action with invalid curvature found.\n")
        @printf("Curvature = %f\n", (control_id >> 70) * CURVATURE_RESOLUTION)
        exit()
      end
    end
  end

  @printf("Building optimized heuristic lut 3...\n")
  optimized_heuristic_lut_3 = generateHLUT(optimized_control_action_lut_3)

  # Plan a path using optimized control set 3.
  @printf("Planning path using optimized control set 3...\n")
  total_dl_score = 0.0
  total_time = 0.0
  optimized_planned_path_3s::Array{Array{Float64, 2}} = []
  optimized_3_path_control_actions::Array{Array{UInt128}} = []
  optimized_3_sigs::Array{Array{Float64}} = []
  for i = 1:size(test_set, 1)
    #occupancy_grid::OccupancyGrid = OccupancyGrid(test_set[i], false)
    #occupancy_grid::OccupancyGrid = OccupancyGrid(test_set[i], true, lane_change_vals[i]<0.5, 10, true)
    occupancy_grid::OccupancyGrid = occupancy_grids[raw_grid_indices[i]]
    CPUtic()
    optimized_planned_path_3::Array{UInt128} = 
      planPath(optimized_control_action_lut_3, optimized_heuristic_lut_3, 
      swath_lut, occupancy_grid, occupancy_grid.goal)
    total_time += CPUtoc()
    push!(optimized_3_path_control_actions, optimized_planned_path_3)
    push!(optimized_planned_path_3s, getPathFromControlActions(optimized_planned_path_3, optimized_control_action_lut_3))

    #plotControlActionsOnOccupancy(swath_lut, optimized_control_action_lut_3, optimized_planned_path_3, 
      #occupancy_grid, string("Optimized Set 3 Planned Path ", string(i)))
  end

  sig_score = 0.0
  for i = 1:size(optimized_planned_path_3s, 1)
    push!(optimized_3_sigs, iisignature.sig(optimized_planned_path_3s[i], 4)) 
    sig_score += cosine_dist(optimized_3_sigs[i], gen_sigs[raw_grid_indices[i]])
  end
  @printf("L1 signature score = %f\n", sig_score)

  CPUtic()
  for i = 1:size(test_set_processed, 1)
    dl_res = findClosestPath(optimized_control_action_lut_3, test_set_processed[i], 0.0)
    dl_score = dl_res[1]
    total_dl_score += dl_score
  end
  @printf("Opt Lut 3 set dl check time = %f\n", CPUtoc())

  optimized_3_curvature_distribution = extractMengerCurvatureDistribution(optimized_planned_path_3s, "L1 Set")
  @printf("KL divergence = %f\n", kl_divergence(test_curvature_distribution, optimized_3_curvature_distribution))

  optimized_3_curvature_distribution = extractCurvatureFromDubinsControlSet(optimized_control_action_lut_3, "L1")

  @printf("Control Set KL divergence = %f\n", kl_divergence(test_curvature_distribution, optimized_3_curvature_distribution))

  @printf("Optimized control set 3 average dl score = %f\n", total_dl_score / size(test_set_processed, 1))
  @printf("Optimized control set 3 average planning time = %f\n", total_time / size(test_set, 1))
  data_log_array[5, 4] = total_time / size(test_set, 1)
  @printf("Lambda = 0.001 size = %d\n", temp_size)
  data_log_array[5, 5] = temp_size
  @printf("L1 control set aggressive size = %d\n", aggressive_count)
  @printf("L1 control set passive size = %d\n", passive_count)

  @printf("Freeing optimized HLUT 3...\n")
  optimized_control_action_lut_3 = []
  optimized_heuristic_lut_3 = []
  gc()

  @printf("Optimized control action lut 4 not present. Rebuilding...\n")
  CPUtic()
  optimized_control_action_lut_4 = optimizeControlSet(dense_control_action_lut, training_set, 0.0001)
  time = CPUtoc()
  @printf("Opt 4 generation time = %f.\n", time)

  # Plot the control set.
#  plotControlSetLUT(optimized_control_action_lut_4, "Lambda 0.0001 Control Set")

  temp_size = 0
  for i = 1:size(optimized_control_action_lut_4, 1)
    temp_size += size(collect(keys(optimized_control_action_lut_4[i])), 1)
  end

  # Calculate the amount of aggressive and passive control actions in the set.
  passive_count = 0
  aggressive_count = 0
  for i in 1:size(TI_RANGE, 1)
    for control_id in keys(optimized_control_action_lut_4[i])
      if (control_id >> 70) == round(PASSIVE_CURVATURE / CURVATURE_RESOLUTION)
        passive_count += 1
      elseif (control_id >> 70) == round(AGGRESSIVE_CURVATURE / CURVATURE_RESOLUTION)
        aggressive_count += 1
      else
        @printf("Error, control action with invalid curvature found.\n")
        @printf("Curvature = %f\n", (control_id >> 70) * CURVATURE_RESOLUTION)
        exit()
      end
    end
  end

  @printf("Building optimized heuristic lut 4...\n")
  optimized_heuristic_lut_4 = generateHLUT(optimized_control_action_lut_4)

  # Plan a path using optimized control set 4.
  @printf("Planning path using optimized control set 4...\n")
  total_dl_score = 0.0
  total_time = 0.0
  optimized_planned_path_4s::Array{Array{Float64, 2}} = []
  optimized_4_path_control_actions::Array{Array{UInt128}} = []
  optimized_4_sigs::Array{Array{Float64}} = []
  for i = 1:size(test_set, 1)
#    occupancy_grid::OccupancyGrid = OccupancyGrid(test_set[i], false)
#    occupancy_grid::OccupancyGrid = OccupancyGrid(test_set[i], true, lane_change_vals[i]<0.5, 10, true)
    occupancy_grid::OccupancyGrid = occupancy_grids[raw_grid_indices[i]]
    CPUtic()
    optimized_planned_path_4::Array{UInt128} = 
      planPath(optimized_control_action_lut_4, optimized_heuristic_lut_4, 
      swath_lut, occupancy_grid, occupancy_grid.goal)
    total_time += CPUtoc()
    push!(optimized_planned_path_4s, getPathFromControlActions(optimized_planned_path_4, optimized_control_action_lut_4))
    push!(optimized_4_path_control_actions, optimized_planned_path_4)

    #plotControlActionsOnOccupancy(swath_lut, optimized_control_action_lut_4, optimized_planned_path_4, 
      #occupancy_grid, string("Optimized Set 4 Planned Path ", string(i)))
  end

  sig_score = 0.0
  for i = 1:size(optimized_planned_path_4s, 1)
    push!(optimized_4_sigs, iisignature.sig(optimized_planned_path_4s[i], 4)) 
    sig_score += cosine_dist(optimized_4_sigs[i], gen_sigs[raw_grid_indices[i]])
  end
  @printf("L2 signature score = %f\n", sig_score)

  CPUtic()
  for i = 1:size(test_set_processed, 1)
    dl_res = findClosestPath(optimized_control_action_lut_4, test_set_processed[i], 0.0)
    dl_score = dl_res[1]
    total_dl_score += dl_score
  end
  @printf("Opt Lut 4 set dl check time = %f\n", CPUtoc())

  optimized_4_curvature_distribution = extractMengerCurvatureDistribution(optimized_planned_path_4s, "L2 Set")

  @printf("KL divergence = %f\n", kl_divergence(test_curvature_distribution, optimized_4_curvature_distribution))

  optimized_4_curvature_distribution = extractCurvatureFromDubinsControlSet(optimized_control_action_lut_4, "L2")

  @printf("Control Set KL divergence = %f\n", kl_divergence(test_curvature_distribution, optimized_4_curvature_distribution))

  @printf("Optimized control set 4 average dl score = %f\n", total_dl_score / size(test_set_processed, 1))
  @printf("Optimized control set 4 average planning time = %f\n", total_time / size(test_set, 1))
  data_log_array[6, 4] = total_time / size(test_set, 1)
  @printf("Lambda = 0.001 size = %d\n", temp_size)
  data_log_array[6, 5] = temp_size
  @printf("L2 control set aggressive size = %d\n", aggressive_count)
  @printf("L2 control set passive size = %d\n", passive_count)

  @printf("Freeing optimized HLUT 4...\n")
  optimized_control_action_lut_4 = []
  optimized_heuristic_lut_4 = []
  gc()
  
  writedlm("data_log.csv", data_log_array)

end

end # module
