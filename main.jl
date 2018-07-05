push!(LOAD_PATH, pwd())

using Constants, ControlActionGenerator, LatticeState, PolynomialSpiral, LatticeVisualizer, 
LatticePathFinder, Geometry, PathExtractor, PathReader, ControlSetOptimizer, 
SwathGenerator, PathConstructor, LatticeCompressor, LatticeOccupancyGrid, HLUTGenerator,
Utils, JLD, LatticePlanner, CPUTime, LatticePathCluster, OptimizationPlanner

function main()
  @printf("Extracting raw paths...\n")
  processed_path_array::Array{Array{Float64, 2}} = []
  raw_path_array::Array{Array{Float64, 2}} = []
  try
    processed_path_array = load("processed_path_array.jld", "processed_path_array")
    raw_path_array = load("raw_path_array.jld", "raw_path_array")
  catch
    processed_path_array, raw_path_array = extractPaths("roundabout_data.csv", 0.85)
    save("processed_path_array.jld", "processed_path_array", processed_path_array)
    save("raw_path_array.jld", "raw_path_array", raw_path_array)
  end
  
  clustered_paths = processed_path_array
  center_line::Array{Float64, 2} = Array{Float64, 2}(size(raw_path_array[1], 1), 3)
  center_line[1, :] = [raw_path_array[1][1, 1], raw_path_array[1][1, 2], 0.0]
  for i = 2:size(center_line, 1)
    center_line[i, 1] = raw_path_array[1][i, 1]
    center_line[i, 2] = raw_path_array[1][i, 2]
    delta_x::Float64 = center_line[i, 1] - center_line[i-1, 1]
    delta_y::Float64 = center_line[i, 2] - center_line[i-1, 2]
    center_line[i, 3] = atan2(delta_y, delta_x)
  end

  occupancy_grid_test::OccupancyGrid = OccupancyGrid(center_line, false, false, 30)
  planned_path = planOptimizerPath(occupancy_grid_test.center_line, occupancy_grid_test) 
  plotPathOnOccupancy(planned_path, occupancy_grid_test)
  
  exit()

  data_log_array::Array{Float64, 2} = Array{Float64, 2}(6, 5)

  # Load/generate the dense set of control actions.
  dense_control_action_lut::Array{Dict{UInt128, SpiralControlAction}} = []

  try
    dense_control_action_lut = load("dense_control_action_lut.jld", "dense_control_action_lut")
    @printf("Loaded Control Action Lut.\n")
  catch
    @printf("Control Action LUT not present. Rebuilding...\n")
    dense_control_action_lut = generateControlActions()
    save("dense_control_action_lut.jld", "dense_control_action_lut", dense_control_action_lut)
  end

  dense_control_set_size::UInt64 = 0
  for i in 1:size(TI_RANGE, 1)
    dense_control_set_size += length(dense_control_action_lut[i])
  end


  # Faster to just regenerate the swath LUT.
  # Note that all control sets are subsets of the dense set,
  # so we can use the same swath LUT for all of them.
  @printf("Generating Swath LUT...\n")
  swath_lut::Dict{UInt128, Set{Tuple{UInt64, UInt64}}} = generateSwaths(dense_control_action_lut) 
  @printf("Swath LUT generated.\n")

  # Load/generate the dense heuristic LUT.
  @printf("Building dense HLUT...\n")
  dense_heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = generateHLUT(dense_control_action_lut)

  # Plan a path using the dense control set.
  @printf("Planning path using dense control set...\n")
  total_dist_score = 0.0
  total_goal_score = 0.0
  total_bending_energy_score = 0.0
  total_time = 0.0
  for i = 1:size(raw_path_array, 1)
    occupancy_grid::OccupancyGrid = OccupancyGrid(raw_path_array[i], true, true, 30)
    CPUtic()
    dense_planned_path::Array{UInt128} = 
      planPath(dense_control_action_lut, dense_heuristic_lut, 
      swath_lut, occupancy_grid, occupancy_grid.goal)
    total_time += CPUtoc()

    dist_score, goal_score, bending_energy_score = getResultScores(dense_planned_path, dense_control_action_lut, occupancy_grid.goal)
    total_dist_score += dist_score
    total_goal_score += goal_score
    total_bending_energy_score += bending_energy_score
    
    plotControlActionsOnOccupancy(swath_lut, dense_control_action_lut, dense_planned_path, 
      occupancy_grid, string("Dense Set Planned Path ", string(i)))
  end

  @printf("Dense control set average distance score = %f\n", total_dist_score / size(raw_path_array, 1))
  data_log_array[1, 1] = total_dist_score / size(raw_path_array, 1)
  @printf("Dense control set average goal score = %f\n", total_goal_score / size(raw_path_array, 1))
  data_log_array[1, 2] = total_goal_score / size(raw_path_array, 1)
  @printf("Dense control set average bending energy score = %f\n", total_bending_energy_score / size(raw_path_array, 1))
  data_log_array[1, 3] = total_bending_energy_score / size(raw_path_array, 1)
  @printf("Dense control set average planning time = %f\n", total_time / size(raw_path_array, 1))
  data_log_array[1, 4] = total_time / size(raw_path_array, 1)
  @printf("Dense Control Set Size = %d\n", dense_control_set_size)
  data_log_array[1, 5] = dense_control_set_size

  # Free some memory.
  @printf("Freeing dense heuristic lut...\n")
  dense_heuristic_lut = []

  # Load/generate a lattice compressed via Pivtoraiko's D* algorithm.
  @printf("Generating Pivtoraiko control action lut...")
  pivtoraiko_control_action_lut::Array{Dict{UInt128, SpiralControlAction}} = compressLattice(dense_control_action_lut, 2.0)

  pivtoraiko_control_set_size::UInt64 = 0
  for i in 1:size(TI_RANGE, 1)
    pivtoraiko_control_set_size += length(pivtoraiko_control_action_lut[i])
  end

  # Load/generate the Pivtoraiko heuristic LUT.
  pivtoraiko_heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []
  @printf("Building Pivtoraiko HLUT...\n")
  pivtoraiko_heuristic_lut = generateHLUT(pivtoraiko_control_action_lut)

  # Plan a path using the Pivtoraiko control set.
  @printf("Planning path using Pivtoraiko control set...\n")
  total_dist_score = 0.0
  total_goal_score = 0.0
  total_bending_energy_score = 0.0
  total_time = 0.0
  for i = 1:size(raw_path_array, 1)
    occupancy_grid::OccupancyGrid = OccupancyGrid(raw_path_array[i], true, true, 30)
    CPUtic()
    pivtoraiko_planned_path::Array{UInt128} = 
      planPath(pivtoraiko_control_action_lut, pivtoraiko_heuristic_lut, 
      swath_lut, occupancy_grid, occupancy_grid.goal)
    total_time += CPUtoc()

    dist_score, goal_score, bending_energy_score = getResultScores(pivtoraiko_planned_path, pivtoraiko_control_action_lut, occupancy_grid.goal)
    total_dist_score += dist_score
    total_goal_score += goal_score
    total_bending_energy_score += bending_energy_score
    
    plotControlActionsOnOccupancy(swath_lut, pivtoraiko_control_action_lut, pivtoraiko_planned_path, 
      occupancy_grid, string("Pivtoraiko Set Planned Path ", string(i)))
  end

  @printf("Pivtoraiko control set average distance score = %f\n", total_dist_score / size(raw_path_array, 1))
  data_log_array[2, 1] = total_dist_score / size(raw_path_array, 1)
  @printf("Pivtoraiko control set average goal score = %f\n", total_goal_score / size(raw_path_array, 1))
  data_log_array[2, 2] = total_goal_score / size(raw_path_array, 1)
  @printf("Pivtoraiko control set average bending energy score = %f\n", total_bending_energy_score / size(raw_path_array, 1))
  data_log_array[2, 3] = total_bending_energy_score / size(raw_path_array, 1)
  @printf("Pivtoraiko control set average planning time = %f\n", total_time / size(raw_path_array, 1))
  data_log_array[2, 4] = total_time / size(raw_path_array, 1)
  @printf("Pivtoraiko Control Set Size = %d\n", pivtoraiko_control_set_size)
  data_log_array[2, 5] = pivtoraiko_control_set_size

  # Free some memory.
  @printf("Freeing Pivtoraiko lut...\n")
  pivtoraiko_control_action_lut = []
  pivtoraiko_heuristic_lut = []

  optimized_control_action_lut_1::Array{Dict{UInt128, SpiralControlAction}} = []
  optimized_heuristic_lut_1::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []
  optimized_control_action_lut_2::Array{Dict{UInt128, SpiralControlAction}} = []
  optimized_heuristic_lut_2::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []
  optimized_control_action_lut_3::Array{Dict{UInt128, SpiralControlAction}} = []
  optimized_heuristic_lut_3::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []
  optimized_control_action_lut_4::Array{Dict{UInt128, SpiralControlAction}} = []
  optimized_heuristic_lut_4::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []

  @printf("Optimizing control sets...\n")
  @printf("Optimized control action lut 1 not present. Rebuilding...\n")
  optimized_control_action_lut_1 = optimizeControlSet(dense_control_action_lut, clustered_paths, 0.1)

  temp_size = 0
  for i = 1:size(optimized_control_action_lut_1, 1)
    temp_size += size(collect(keys(optimized_control_action_lut_1[i])), 1)
  end


  @printf("Building optimized heuristic lut 1...\n")
  optimized_heuristic_lut_1 = generateHLUT(optimized_control_action_lut_1)

  # Plan a path using optimized control set 1.
  @printf("Planning path using optimized control set 1...\n")
  total_dist_score = 0.0
  total_goal_score = 0.0
  total_bending_energy_score = 0.0 
  total_time = 0.0
  for i = 1:size(raw_path_array, 1)
    occupancy_grid::OccupancyGrid = OccupancyGrid(raw_path_array[i], true, true, 30)
    CPUtic()
    optimized_planned_path_1::Array{UInt128} = 
      planPath(optimized_control_action_lut_1, optimized_heuristic_lut_1, 
      swath_lut, occupancy_grid, occupancy_grid.goal)
    total_time += CPUtoc()

    dist_score, goal_score, bending_energy_score = getResultScores(optimized_planned_path_1, optimized_control_action_lut_1, occupancy_grid.goal)
    total_dist_score += dist_score
    total_goal_score += goal_score
    total_bending_energy_score += bending_energy_score
    
    plotControlActionsOnOccupancy(swath_lut, optimized_control_action_lut_1, optimized_planned_path_1, 
      occupancy_grid, string("Optimized Set 1 Planned Path ", string(i)))
  end

  @printf("Optimized control set 1 average distance score = %f\n", total_dist_score / size(raw_path_array, 1))
  data_log_array[3, 1] = total_dist_score / size(raw_path_array, 1)
  @printf("Optimized control set 1 average goal score = %f\n", total_goal_score / size(raw_path_array, 1))
  data_log_array[3, 2] = total_goal_score / size(raw_path_array, 1)
  @printf("Optimized control set 1 average bending energy score = %f\n", total_bending_energy_score / size(raw_path_array, 1))
  data_log_array[3, 3] = total_bending_energy_score / size(raw_path_array, 1)
  @printf("Optimized control set 1 average planning time = %f\n", total_time / size(raw_path_array, 1))
  data_log_array[3, 4] = total_time / size(raw_path_array, 1)
  @printf("Lambda = 0.1 size = %d\n", temp_size)
  data_log_array[3, 5] = temp_size

  @printf("Freeing optimized HLUT 1...\n")
  optimized_control_action_lut_1 = []
  optimized_heuristic_lut_1 = []

  @printf("Optimized control action lut 2 not present. Rebuilding...\n")
  optimized_control_action_lut_2 = optimizeControlSet(dense_control_action_lut, clustered_paths, 0.01)

  temp_size = 0
  for i = 1:size(optimized_control_action_lut_2, 1)
    temp_size += size(collect(keys(optimized_control_action_lut_2[i])), 1)
  end

  @printf("Building optimized heuristic lut 2...\n")
  optimized_heuristic_lut_2 = generateHLUT(optimized_control_action_lut_2)

  # Plan a path using optimized control set 2.
  @printf("Planning path using optimized control set 2...\n")
  total_dist_score = 0.0
  total_goal_score = 0.0
  total_bending_energy_score = 0.0  
  total_time = 0.0
  for i = 1:size(raw_path_array, 1)
    occupancy_grid::OccupancyGrid = OccupancyGrid(raw_path_array[i], true, true, 30)
    CPUtic()
    optimized_planned_path_2::Array{UInt128} = 
      planPath(optimized_control_action_lut_2, optimized_heuristic_lut_2, 
      swath_lut, occupancy_grid, occupancy_grid.goal)
    total_time += CPUtoc()

    dist_score, goal_score, bending_energy_score = getResultScores(optimized_planned_path_2, optimized_control_action_lut_2, occupancy_grid.goal)
    total_dist_score += dist_score
    total_goal_score += goal_score
    total_bending_energy_score += bending_energy_score  
    
    plotControlActionsOnOccupancy(swath_lut, optimized_control_action_lut_2, optimized_planned_path_2, 
      occupancy_grid, string("Optimized Set 2 Planned Path ", string(i)))
  end

  @printf("Optimized control set 2 average distance score = %f\n", total_dist_score / size(raw_path_array, 1))
  data_log_array[4, 1] = total_dist_score / size(raw_path_array, 1)
  @printf("Optimized control set 2 average goal score = %f\n", total_goal_score / size(raw_path_array, 1))
  data_log_array[4, 2] = total_goal_score / size(raw_path_array, 1)
  @printf("Optimized control set 2 average bending energy score = %f\n", total_bending_energy_score / size(raw_path_array, 1))
  data_log_array[4, 3] = total_bending_energy_score / size(raw_path_array, 1)
  @printf("Optimized control set 2 average planning time = %f\n", total_time / size(raw_path_array, 1))
  data_log_array[4, 4] = total_time / size(raw_path_array, 1)
  @printf("Lambda = 0.01 size = %d\n", temp_size)
  data_log_array[4, 5] = temp_size

  @printf("Freeing optimized HLUT 2...\n")
  optimized_control_action_lut_2 = []
  optimized_heuristic_lut_2 = []

  @printf("Optimized control action lut 3 not present. Rebuilding...\n")
  optimized_control_action_lut_3 = optimizeControlSet(dense_control_action_lut, clustered_paths, 0.001)
  
  temp_size = 0
  for i = 1:size(optimized_control_action_lut_3, 1)
    temp_size += size(collect(keys(optimized_control_action_lut_3[i])), 1)
  end

  @printf("Building optimized heuristic lut 3...\n")
  optimized_heuristic_lut_3 = generateHLUT(optimized_control_action_lut_3)

  # Plan a path using optimized control set 3.
  @printf("Planning path using optimized control set 3...\n")
  total_dist_score = 0.0
  total_goal_score = 0.0
  total_bending_energy_score = 0.0
  total_time = 0.0
  for i = 1:size(raw_path_array, 1)
    occupancy_grid::OccupancyGrid = OccupancyGrid(raw_path_array[i], true, true, 30)
    CPUtic()
    optimized_planned_path_3::Array{UInt128} = 
      planPath(optimized_control_action_lut_3, optimized_heuristic_lut_3, 
      swath_lut, occupancy_grid, occupancy_grid.goal)
    total_time += CPUtoc()

    dist_score, goal_score, bending_energy_score = getResultScores(optimized_planned_path_3, optimized_control_action_lut_3, occupancy_grid.goal)
    total_dist_score += dist_score
    total_goal_score += goal_score
    total_bending_energy_score += bending_energy_score
    
    plotControlActionsOnOccupancy(swath_lut, optimized_control_action_lut_3, optimized_planned_path_3, 
      occupancy_grid, string("Optimized Set 3 Planned Path ", string(i)))
  end

  @printf("Optimized control set 3 average distance score = %f\n", total_dist_score / size(raw_path_array, 1))
  data_log_array[5, 1] = total_dist_score / size(raw_path_array, 1)
  @printf("Optimized control set 3 average goal score = %f\n", total_goal_score / size(raw_path_array, 1))
  data_log_array[5, 2] = total_goal_score / size(raw_path_array, 1)
  @printf("Optimized control set 3 average bending energy score = %f\n", total_bending_energy_score / size(raw_path_array, 1))
  data_log_array[5, 3] = total_bending_energy_score / size(raw_path_array, 1)
  @printf("Optimized control set 3 average planning time = %f\n", total_time / size(raw_path_array, 1))
  data_log_array[5, 4] = total_time / size(raw_path_array, 1)
  @printf("Lambda = 0.001 size = %d\n", temp_size)
  data_log_array[5, 5] = temp_size

  @printf("Freeing optimized HLUT 3...\n")
  optimized_control_action_lut_3 = []
  optimized_heuristic_lut_3 = []

  @printf("Optimized control action lut 4 not present. Rebuilding...\n")
  optimized_control_action_lut_4 = optimizeControlSet(dense_control_action_lut, clustered_paths, 0.0001)

  temp_size = 0
  for i = 1:size(optimized_control_action_lut_4, 1)
    temp_size += size(collect(keys(optimized_control_action_lut_4[i])), 1)
  end

  @printf("Building optimized heuristic lut 4...\n")
  optimized_heuristic_lut_4 = generateHLUT(optimized_control_action_lut_4)

  # Plan a path using optimized control set 4.
  @printf("Planning path using optimized control set 4...\n")
  total_dist_score = 0.0
  total_goal_score = 0.0
  total_bending_energy_score = 0.0
  total_time = 0.0
  for i = 1:size(raw_path_array, 1)
    occupancy_grid::OccupancyGrid = OccupancyGrid(raw_path_array[i], true, true, 30)
    CPUtic()
    optimized_planned_path_4::Array{UInt128} = 
      planPath(optimized_control_action_lut_4, optimized_heuristic_lut_4, 
      swath_lut, occupancy_grid, occupancy_grid.goal)
    total_time += CPUtoc()

    dist_score, goal_score, bending_energy_score = getResultScores(optimized_planned_path_4, optimized_control_action_lut_4, occupancy_grid.goal)
    total_dist_score += dist_score
    total_goal_score += goal_score
    total_bending_energy_score += bending_energy_score
    
    plotControlActionsOnOccupancy(swath_lut, optimized_control_action_lut_4, optimized_planned_path_4, 
      occupancy_grid, string("Optimized Set 4 Planned Path ", string(i)))
  end

  @printf("Optimized control set 4 average distance score = %f\n", total_dist_score / size(raw_path_array, 1))
  data_log_array[6, 1] = total_dist_score / size(raw_path_array, 1)
  @printf("Optimized control set 4 average goal score = %f\n", total_goal_score / size(raw_path_array, 1))
  data_log_array[6, 2] = total_goal_score / size(raw_path_array, 1)
  @printf("Optimized control set 4 average bending energy score = %f\n", total_bending_energy_score / size(raw_path_array, 1))
  data_log_array[6, 3] = total_bending_energy_score / size(raw_path_array, 1)
  @printf("Optimized control set 4 average planning time = %f\n", total_time / size(raw_path_array, 1))
  data_log_array[6, 4] = total_time / size(raw_path_array, 1)
  @printf("Lambda = 0.001 size = %d\n", temp_size)
  data_log_array[6, 5] = temp_size

  @printf("Freeing optimized HLUT 4...\n")
  optimized_control_action_lut_4 = []
  optimized_heuristic_lut_4 = []
  
  writedlm("data_log.csv", data_log_array)

end

main()
