module SpiralExperiment

using Constants, ControlActionGenerator, LatticeState, PolynomialSpiral, LatticeVisualizer, 
LatticePathFinder, Geometry, PathExtractor, PathReader, ControlSetOptimizer, 
SwathGenerator, PathConstructor, LatticeCompressor, LatticeOccupancyGrid, HLUTGenerator,
Utils, JLD, LatticePlanner, CPUTime, LatticePathCluster, OptimizationPlanner, StatsBase, 
ParameterComparison, Distances, CurvatureExperiment, PyPlot, PyCall, Formatting

@pyimport iisignature

export spiralExperiment1, spiralExperiment2, spiralExperiment3, closestPathGenerationExperiment,
algorithmTimingComparison, algorithmVertexComparison, closestPathGenerationExperiment,
aggressivePassiveControlSetCurvatureComparison

# Learn spiral control sets from the dataset, evaluated on a full path
# from the test set.
function spiralExperiment1(use_clusters::Bool=false, guided_search::Bool=false, rotate_paths::Bool=true)
  @printf("Extracting raw paths...\n")
  training_set::Array{Array{Array{Float64, 2}}} = []
  test_set::Array{Array{Float64, 2}} = []
  test_set_processed::Array{Array{Float64, 2}} = []
  training_set, test_set, test_set_processed = extractPaths("roundabout_data.csv", 0.85, rotate_paths)
  @printf("Size test set = %d\n", size(test_set, 1))

  # Initialize the cluster set array, and calculate a cluster set for each heading.
  cluster_sets::Array{Array{PathCluster}} = Array{Array{PathCluster}, 1}()
  if use_clusters
    if rotate_paths
      for i = 1:size(TI_RANGE, 1)
        push!(cluster_sets, kMeansPaths(training_set[i], NUM_CLUSTERS, true, false, TI_RANGE[i]))
        @printf("Cluster size = %i\n", size(cluster_sets[i], 1))
        for j = 1:size(cluster_sets[i], 1)
          @printf("Cluster %i path count = %i\n", j, size(cluster_sets[i][j].paths, 1))
        end
      end
    else
      push!(cluster_sets, kMeansPaths(training_set[1], NUM_CLUSTERS, false, false, TI_RANGE[1]))

      # Delete the empty clusters from the set.
      indices_to_delete = []
      for j = 1:size(cluster_sets[1], 1)
        num_paths_in_cluster = size(cluster_sets[1][j].paths, 1)
        if num_paths_in_cluster == 0
          push!(indices_to_delete, j)
        end
      end

      deleteat!(cluster_sets[1], indices_to_delete)

      for j = 1:size(cluster_sets[1], 1)
        @printf("Cluster path count = %i\n", size(cluster_sets[1][j].paths, 1))
      end
    end
  end

  # Change directories.
  try
    mkdir("experiment_1")
  catch
    @printf("Experiment 1 directory already exists.\n")
  end
  cd("experiment_1")

  # Compute a curvature distribution for the generated paths.
  if guided_search
    test_curvature_distribution = extractMengerCurvatureDistribution(test_set, "Experiment 1 Data Curvature Clustered")
  else
    test_curvature_distribution = extractMengerCurvatureDistribution(test_set, "Experiment 1 Data Curvature Standard")
  end

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

  lane_change_vals::Array{Float64} = rand(Float64, size(test_set, 1))

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

  # Compute the required occupancy grids for the test set.
  occupancy_grids::Array{OccupancyGrid} = []
  for i = 1:size(test_set, 1)
    push!(occupancy_grids, OccupancyGrid(test_set[i], false))
  end

  # Compute the dense control set metrics.
  @printf("Computing Dense Control Set Metrics.\n")
  dense_path_control_actions, dense_curvature_deviation_vals = computeControlSetMetrics(test_set, test_set_processed, occupancy_grids, dense_control_action_lut, dense_heuristic_lut, swath_lut, test_curvature_distribution, "Experiment 1 Dense Set", true)

  # Free some memory.
  @printf("Freeing dense heuristic lut...\n")
  dense_heuristic_lut = []
  gc()

  # Load/generate a lattice compressed via Pivtoraiko's D* algorithm.
  @printf("Generating Pivtoraiko control action lut...")
  CPUtic()
  pivtoraiko_control_action_lut::Array{Dict{UInt128, SpiralControlAction}} = compressLattice(dense_control_action_lut, 2.0)
  time = CPUtoc()
  @printf("Pivtoraiko generation time = %f.\n", time)

  # Load/generate the Pivtoraiko heuristic LUT.
  pivtoraiko_heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []
  @printf("Building Pivtoraiko HLUT...\n")
  pivtoraiko_heuristic_lut = generateHLUT(pivtoraiko_control_action_lut)

  # Compute the Pivtoraiko control set metrics.
  @printf("Computing Pivtoraiko Control Set Metrics.\n")
  pivtoraiko_path_control_actions, pivtoraiko_curvature_deviation_vals = computeControlSetMetrics(test_set, test_set_processed, occupancy_grids, pivtoraiko_control_action_lut, pivtoraiko_heuristic_lut, swath_lut, test_curvature_distribution, "Experiment 1 DL Set", true)

  # Free some memory.
  @printf("Freeing Pivtoraiko lut...\n")
  pivtoraiko_control_action_lut = []
  pivtoraiko_heuristic_lut = []
  gc()

  lambda_1_control_action_lut::Array{Dict{UInt128, SpiralControlAction}} = []
  lambda_1_heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []
  lambda_2_control_action_lut::Array{Dict{UInt128, SpiralControlAction}} = []
  lambda_2_heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []

  @printf("Optimizing lambda 1 control set...\n")
  CPUtic()
  if use_clusters
    lambda_1_control_action_lut = optimizeControlSetClustered(dense_control_action_lut, training_set, 0.001, cluster_sets, guided_search)
  else
    lambda_1_control_action_lut = optimizeControlSet(dense_control_action_lut, training_set, 0.001)
  end
  time = CPUtoc()
  @printf("Lambda 1 generation time = %f.\n", time)
  
  @printf("Building Lambda 1 Heuristic LUT...\n")
  lambda_1_heuristic_lut = generateHLUT(lambda_1_control_action_lut)

  # Compute the lambda 1 control set metrics.
  @printf("Computing Lambda 1 Control Set Metrics.\n")
  lambda_1_path_control_actions, lambda_1_curvature_deviation_vals = computeControlSetMetrics(test_set, test_set_processed, occupancy_grids, lambda_1_control_action_lut, lambda_1_heuristic_lut, swath_lut, test_curvature_distribution, "Experiment 1 Lambda 1 Set", true)

  @printf("Freeing lambda 1 HLUT...\n")
  lambda_1_control_action_lut = []
  lambda_1_heuristic_lut = []
  gc()

  @printf("Optimizing lambda 2 control set...\n")
  CPUtic()
  if use_clusters
    lambda_2_control_action_lut = optimizeControlSetClustered(dense_control_action_lut, training_set, 0.0001, cluster_sets, guided_search)
  else
    lambda_2_control_action_lut = optimizeControlSet(dense_control_action_lut, training_set, 0.0001)
  end
  time = CPUtoc()
  @printf("Opt 4 generation time = %f.\n", time)

  @printf("Building optimized heuristic lut 4...\n")
  lambda_2_heuristic_lut = generateHLUT(lambda_2_control_action_lut)

  # Compute the lambda 2 control set metrics.
  @printf("Computing Lambda 2 Control Set Metrics.\n")
  lambda_2_path_control_actions, lambda_2_curvature_deviation_vals = computeControlSetMetrics(test_set, test_set_processed, occupancy_grids, lambda_2_control_action_lut, lambda_2_heuristic_lut, swath_lut, test_curvature_distribution, "Experiment 1 Lambda 2 Set", true)

  @printf("Freeing optimized HLUT 4...\n")
  lambda_2_control_action_lut = []
  lambda_2_heuristic_lut = []
  gc()

  computeCurvatureDeviationValues(dense_curvature_deviation_vals, pivtoraiko_curvature_deviation_vals, lambda_1_curvature_deviation_vals, lambda_2_curvature_deviation_vals, "Experiment 1")

  cd("..")
  
end

# Learn spiral control sets from the dataset, evaluated on a lane change
# from the test set.
function spiralExperiment2(use_clusters::Bool=false, guided_search::Bool=false, rotate_paths::Bool=true)
  @printf("Extracting raw paths...\n")
  training_set::Array{Array{Array{Float64, 2}}} = []
  test_set::Array{Array{Float64, 2}} = []
  test_set_processed::Array{Array{Float64, 2}} = []
  training_set, test_set, test_set_processed = extractPaths("roundabout_data.csv", 0.85, rotate_paths)

  # Initialize the cluster set array, and calculate a cluster set for each heading.
  cluster_sets::Array{Array{PathCluster}} = Array{Array{PathCluster}, 1}()
  if use_clusters
    if rotate_paths
      for i = 1:size(TI_RANGE, 1)
        push!(cluster_sets, kMeansPaths(training_set[i], NUM_CLUSTERS, true, false, TI_RANGE[i]))
        @printf("Cluster size = %i\n", size(cluster_sets[i], 1))
        for j = 1:size(cluster_sets[i], 1)
          @printf("Cluster %i path count = %i\n", j, size(cluster_sets[i][j].paths, 1))
        end
      end
    else
      push!(cluster_sets, kMeansPaths(training_set[1], NUM_CLUSTERS, false, false, TI_RANGE[1]))

      # Delete the empty clusters from the set.
      indices_to_delete = []
      for j = 1:size(cluster_sets[1], 1)
        num_paths_in_cluster = size(cluster_sets[1][j].paths, 1)
        if num_paths_in_cluster == 0
          push!(indices_to_delete, j)
        end
      end

      deleteat!(cluster_sets[1], indices_to_delete)

      for j = 1:size(cluster_sets[1], 1)
        @printf("Cluster path count = %i\n", size(cluster_sets[1][j].paths, 1))
      end
    end
  end

  # Change directories.
  try
    mkdir("experiment_2")
  catch
    @printf("Experiment 2 directory already exists.\n")
  end
  cd("experiment_2")

  # Compute a curvature distribution for the generated paths.
  if guided_search
    test_curvature_distribution = extractMengerCurvatureDistribution(test_set, "Experiment 2 Data Curvature Clustered")
  else
    test_curvature_distribution = extractMengerCurvatureDistribution(test_set, "Experiment 2 Data Curvature Standard")
  end

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

  lane_change_vals::Array{Float64} = rand(Float64, size(test_set, 1))

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

  # Compute the required occupancy grids for the test set.
  occupancy_grids::Array{OccupancyGrid} = []
  for i = 1:size(test_set, 1)
    push!(occupancy_grids, OccupancyGrid(test_set[i], true, lane_change_vals[i]<0.5, 10, true))
  end

  # Compute the dense control set metrics.
  @printf("Computing Dense Control Set Metrics.\n")
  dense_path_control_actions, dense_curvature_deviation_vals = computeControlSetMetrics(test_set, test_set_processed, occupancy_grids, dense_control_action_lut, dense_heuristic_lut, swath_lut, test_curvature_distribution, "Experiment 2 Dense Set", true)

  # Free some memory.
  @printf("Freeing dense heuristic lut...\n")
  dense_heuristic_lut = []
  gc()

  # Load/generate a lattice compressed via Pivtoraiko's D* algorithm.
  @printf("Generating Pivtoraiko control action lut...")
  CPUtic()
  pivtoraiko_control_action_lut::Array{Dict{UInt128, SpiralControlAction}} = compressLattice(dense_control_action_lut, 2.0)
  time = CPUtoc()
  @printf("Pivtoraiko generation time = %f.\n", time)

  # Load/generate the Pivtoraiko heuristic LUT.
  pivtoraiko_heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []
  @printf("Building Pivtoraiko HLUT...\n")
  pivtoraiko_heuristic_lut = generateHLUT(pivtoraiko_control_action_lut)

  # Compute the Pivtoraiko control set metrics.
  @printf("Computing Pivtoraiko Control Set Metrics.\n")
  pivtoraiko_path_control_actions, pivtoraiko_curvature_deviation_vals = computeControlSetMetrics(test_set, test_set_processed, occupancy_grids, pivtoraiko_control_action_lut, pivtoraiko_heuristic_lut, swath_lut, test_curvature_distribution, "Experiment 2 DL Set", true)

  # Free some memory.
  @printf("Freeing Pivtoraiko lut...\n")
  pivtoraiko_heuristic_lut = []
  gc()

  lambda_1_control_action_lut::Array{Dict{UInt128, SpiralControlAction}} = []
  lambda_1_heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []
  lambda_2_control_action_lut::Array{Dict{UInt128, SpiralControlAction}} = []
  lambda_2_heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []

  @printf("Optimizing lambda 1 control set...\n")
  CPUtic()
  if use_clusters
    lambda_1_control_action_lut = optimizeControlSetClustered(dense_control_action_lut, training_set, 0.001, cluster_sets, guided_search)
  else
    lambda_1_control_action_lut = optimizeControlSet(dense_control_action_lut, training_set, 0.001)
  end
  time = CPUtoc()
  @printf("Lambda 1 generation time = %f.\n", time)
  
  @printf("Building Lambda 1 Heuristic LUT...\n")
  lambda_1_heuristic_lut = generateHLUT(lambda_1_control_action_lut)

  # Compute the lambda 1 control set metrics.
  @printf("Computing Lambda 1 Control Set Metrics.\n")
  lambda_1_path_control_actions, lambda_1_curvature_deviation_vals = computeControlSetMetrics(test_set, test_set_processed, occupancy_grids, lambda_1_control_action_lut, lambda_1_heuristic_lut, swath_lut, test_curvature_distribution, "Experiment 2 Lambda 1 Set", true)

  @printf("Freeing lambda 1 HLUT...\n")
  lambda_1_heuristic_lut = []
  gc()

  @printf("Optimizing lambda 2 control set...\n")
  CPUtic()
  if use_clusters
    lambda_2_control_action_lut = optimizeControlSetClustered(dense_control_action_lut, training_set, 0.0001, cluster_sets, guided_search)
  else
    lambda_2_control_action_lut = optimizeControlSet(dense_control_action_lut, training_set, 0.0001)
  end
  time = CPUtoc()
  @printf("Opt 4 generation time = %f.\n", time)

  @printf("Building optimized heuristic lut 4...\n")
  lambda_2_heuristic_lut = generateHLUT(lambda_2_control_action_lut)

  # Compute the lambda 2 control set metrics.
  @printf("Computing Lambda 2 Control Set Metrics.\n")
  lambda_2_path_control_actions, lambda_2_curvature_deviation_vals = computeControlSetMetrics(test_set, test_set_processed, occupancy_grids, lambda_2_control_action_lut, lambda_2_heuristic_lut, swath_lut, test_curvature_distribution, "Experiment 2 Lambda 2 Set", true)

  @printf("Freeing optimized HLUT 4...\n")
  lambda_2_heuristic_lut = []
  gc()

  computeCurvatureDeviationValues(dense_curvature_deviation_vals, pivtoraiko_curvature_deviation_vals, lambda_1_curvature_deviation_vals, lambda_2_curvature_deviation_vals, "Experiment 2")

  cd("..")

end

# Learn spiral control sets from a synthetic dataset, evaluated on a 
# similar synthetic test set.
function spiralExperiment3(use_clusters::Bool=false, guided_search::Bool=false, rotate_paths::Bool=true)
  @printf("Generating paths...\n")
  num_generated_paths::Int64 = 100
  gen_paths::Array{Array{Float64, 2}} = Array{Array{Float64, 2}}(num_generated_paths)
  gen_occupancy_grids::Array{OccupancyGrid} = Array{OccupancyGrid}(num_generated_paths)
  gen_curvature_vals::Array{Float64} = []

  i::Int64 = 1
  while i <= num_generated_paths
    # Generate a path according to a randomly sampled curvature rate, lange change direction,
    # and lane change location.
    curvature_rate::Float64 = rand()*0.001
    if rand() > 0.5
      curvature_rate = -curvature_rate
    end

    left_lane::Bool = rand() > 0.5
    curve_split = 0.8
    lane_change_index = rand(175:225)

    occupancy_grid_gen::OccupancyGrid = OccupancyGrid(curvature_rate, 60.0*curve_split, 60.0*(1-curve_split), true, true, left_lane, lane_change_index, true)

    try
     result = planOptimizerPath(occupancy_grid_gen.center_line, occupancy_grid_gen, 0.05, 16.0) 
     planned_path = result[1]
     append!(gen_curvature_vals, result[2])
     gen_paths[i] = planned_path
     gen_occupancy_grids[i] = occupancy_grid_gen
     i += 1
    catch
      @printf("error in optimization.\n")
      continue
    end

  end

  # Extract a training/test split for the generated paths.
  # The training paths will be sliced and rotate/translated to the origin.
  # The test paths will be left as is.
  @printf("gen_set size = %d\n", size(gen_paths, 1))
  training_set, test_set, test_set_processed, raw_grid_indices = extractGeneratedPaths(gen_paths, 0.85, rotate_paths)

  @printf("training_set size = %d\n", size(training_set[1], 1))
  @printf("test_set size = %d\n", size(test_set, 1))
  @printf("test_set_proc size = %d\n", size(test_set_processed, 1))

  # Change directories.
  try
    mkdir("experiment_3")
  catch
    @printf("Experiment 3 directory already exists.\n")
  end
  cd("experiment_3")

  # Compute a curvature distribution for the generated paths.
  if guided_search
    test_curvature_distribution = extractMengerCurvatureDistribution(test_set, "Experiment 3 Data Curvature Clustered")
  else
    test_curvature_distribution = extractMengerCurvatureDistribution(test_set, "Experiment 3 Data Curvature Standard")
  end
  
  # Compute the required occupancy grids for the test set.
  occupancy_grids::Array{OccupancyGrid} = []
  for i = 1:size(raw_grid_indices, 1)
    push!(occupancy_grids, gen_occupancy_grids[raw_grid_indices[i]])
  end

  # Initialize the cluster set array, and calculate a cluster set for each heading.
  cluster_sets::Array{Array{PathCluster}} = Array{Array{PathCluster}, 1}()
  if use_clusters
    if rotate_paths
      for i = 1:size(TI_RANGE, 1)
        push!(cluster_sets, kMeansPaths(training_set[i], NUM_CLUSTERS, true, false, TI_RANGE[i]))
        @printf("Cluster size = %i\n", size(cluster_sets[i], 1))
        for j = 1:size(cluster_sets[i], 1)
          @printf("Cluster %i path count = %i\n", j, size(cluster_sets[i][j].paths, 1))
        end
      end
    else
      push!(cluster_sets, kMeansPaths(training_set[1], NUM_CLUSTERS, false, false, TI_RANGE[1]))

      # Delete the empty clusters from the set.
      indices_to_delete = []
      for j = 1:size(cluster_sets[1], 1)
        num_paths_in_cluster = size(cluster_sets[1][j].paths, 1)
        if num_paths_in_cluster == 0
          push!(indices_to_delete, j)
        end
      end

      deleteat!(cluster_sets[1], indices_to_delete)

      for j = 1:size(cluster_sets[1], 1)
        @printf("Cluster path count = %i\n", size(cluster_sets[1][j].paths, 1))
      end
    end
  end

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

  lane_change_vals::Array{Float64} = rand(Float64, size(test_set, 1))

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

  # Compute the dense control set metrics.
  @printf("Computing Dense Control Set Metrics.\n")
  dense_path_sequences, dense_curvature_deviation_vals = computeControlSetMetrics(test_set, test_set_processed, occupancy_grids, dense_control_action_lut, dense_heuristic_lut, swath_lut, test_curvature_distribution, "Experiment 3 Dense Set", true)

  # Free some memory.
  @printf("Freeing dense heuristic lut...\n")
  dense_heuristic_lut = []
  gc()

  # Load/generate a lattice compressed via Pivtoraiko's D* algorithm.
  @printf("Generating Pivtoraiko control action lut...")
  CPUtic()
  pivtoraiko_control_action_lut::Array{Dict{UInt128, SpiralControlAction}} = compressLattice(dense_control_action_lut, 2.0)
  time = CPUtoc()
  @printf("Pivtoraiko generation time = %f.\n", time)

  # Load/generate the Pivtoraiko heuristic LUT.
  pivtoraiko_heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []
  @printf("Building Pivtoraiko HLUT...\n")
  pivtoraiko_heuristic_lut = generateHLUT(pivtoraiko_control_action_lut)

  # Compute the Pivtoraiko control set metrics.
  @printf("Computing Pivtoraiko Control Set Metrics.\n")
  pivtoraiko_path_sequences, pivtoraiko_curvature_deviation_vals = computeControlSetMetrics(test_set, test_set_processed, occupancy_grids, pivtoraiko_control_action_lut, pivtoraiko_heuristic_lut, swath_lut, test_curvature_distribution, "Experiment 3 DL Set", true)

  # Free some memory.
  @printf("Freeing Pivtoraiko lut...\n")
  #pivtoraiko_control_action_lut = []
  pivtoraiko_heuristic_lut = []
  gc()

  lambda_1_control_action_lut::Array{Dict{UInt128, SpiralControlAction}} = []
  lambda_1_heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []
  lambda_2_control_action_lut::Array{Dict{UInt128, SpiralControlAction}} = []
  lambda_2_heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}} = []

  @printf("Optimizing lambda 1 control set...\n")
  CPUtic()
  if use_clusters
    lambda_1_control_action_lut = optimizeControlSetClustered(dense_control_action_lut, training_set, 0.001, cluster_sets, guided_search)
  else
    lambda_1_control_action_lut = optimizeControlSet(dense_control_action_lut, training_set, 0.001)
  end
  time = CPUtoc()
  @printf("Lambda 1 generation time = %f.\n", time)
  
  @printf("Building Lambda 1 Heuristic LUT...\n")
  lambda_1_heuristic_lut = generateHLUT(lambda_1_control_action_lut)

  # Compute the lambda 1 control set metrics.
  @printf("Computing Lambda 1 Control Set Metrics.\n")
  lambda_1_path_sequences, lambda_1_curvature_deviation_vals = computeControlSetMetrics(test_set, test_set_processed, occupancy_grids, lambda_1_control_action_lut, lambda_1_heuristic_lut, swath_lut, test_curvature_distribution, "Experiment 3 Lambda 1 Set", true)

  @printf("Freeing lambda 1 HLUT...\n")
  #lambda_1_control_action_lut = []
  lambda_1_heuristic_lut = []
  gc()

  @printf("Optimizing lambda 2 control set...\n")
  CPUtic()
  if use_clusters
    lambda_2_control_action_lut = optimizeControlSetClustered(dense_control_action_lut, training_set, 0.0001, cluster_sets, guided_search)
  else
    lambda_2_control_action_lut = optimizeControlSet(dense_control_action_lut, training_set, 0.0001)
  end
  time = CPUtoc()
  @printf("Opt 4 generation time = %f.\n", time)

  @printf("Building optimized heuristic lut 4...\n")
  lambda_2_heuristic_lut = generateHLUT(lambda_2_control_action_lut)

  # Compute the lambda 2 control set metrics.
  @printf("Computing Lambda 2 Control Set Metrics.\n")
  lambda_2_path_sequences, lambda_2_curvature_deviation_vals = computeControlSetMetrics(test_set, test_set_processed, occupancy_grids, lambda_2_control_action_lut, lambda_2_heuristic_lut, swath_lut, test_curvature_distribution, "Experiment 3 Lambda 2 Set", true)

  @printf("Freeing optimized HLUT 4...\n")
  #lambda_2_control_action_lut = []
  lambda_2_heuristic_lut = []
  gc()

  computeCurvatureDeviationValues(dense_curvature_deviation_vals, pivtoraiko_curvature_deviation_vals, lambda_1_curvature_deviation_vals, lambda_2_curvature_deviation_vals, "Experiment 3")

# REMEMBER TO REMOVE GARBAGE COLLECTION BEFORE USING THIS
  plotMultipleControlActionsOnOccupancy(swath_lut, [dense_control_action_lut, pivtoraiko_control_action_lut, lambda_1_control_action_lut, lambda_2_control_action_lut], [dense_path_sequences, pivtoraiko_path_sequences, lambda_1_path_sequences, lambda_2_path_sequences], gen_occupancy_grids, raw_grid_indices)

  cd("..")
  
end

# Computes all of the required metrics for a given control set.
function computeControlSetMetrics(test_set::Array{Array{Float64, 2}}, test_set_processed::Array{Array{Float64, 2}}, occupancy_grids::Array{OccupancyGrid}, control_action_lut::Array{Dict{UInt128, SpiralControlAction}}, heuristic_lut::Array{Dict{UInt128, Tuple{Array{UInt128}, Float64}}}, swath_lut::Dict{UInt128, Set{Tuple{UInt64, UInt64}}}, test_curvature_distribution::Array{Float64}, prefix::String, plot_paths::Bool=false)
  total_dist_score = 0.0
  total_goal_score = 0.0
  total_bending_energy_score = 0.0
  total_time = 0.0
  planned_paths::Array{Array{Float64, 2}} = []
  path_control_actions::Array{Array{UInt128}} = []
  curvature_deviation_vals::Array{Float64} = []

  for i = 1:size(test_set, 1)
    occupancy_grid::OccupancyGrid = occupancy_grids[i]
    CPUtic()
    planned_control_actions::Array{UInt128} = 
      planPath(control_action_lut, heuristic_lut, 
      swath_lut, occupancy_grid, occupancy_grid.goal)
    total_time += CPUtoc()
    planned_path::Array{Float64, 2} = getPathFromControlActions(planned_control_actions, control_action_lut)
    push!(path_control_actions, planned_control_actions)
    push!(planned_paths, planned_path)

    # Computes the maximum curvature deviation stepwise along the planned path
    # from its associated test path (lane centerline).
    max_curvature_deviation = compareStepwiseCurvature(planned_path, test_set[i])
    push!(curvature_deviation_vals, max_curvature_deviation)

    dist_score, goal_score, bending_energy_score = getResultScores(planned_control_actions, control_action_lut, occupancy_grid.goal)

    total_dist_score += dist_score
    total_goal_score += goal_score
    total_bending_energy_score += bending_energy_score
  
    # Plot the paths.  
    if plot_paths
      plotControlActionsOnOccupancy(swath_lut, control_action_lut, planned_control_actions, 
        occupancy_grid, string(prefix, " Planned Path ", string(i)))
    end
  end

  # Compute a histogram of the max curvature deviation vals.
  computeCurvatureMatchingDistribution(curvature_deviation_vals, string(prefix, " Pointwise Curvature Matching Histogram"))
  max_dl_score, average_dl_score = computeMatchingDistribution(control_action_lut, test_set_processed, string(prefix, " Pointwise Distance Matching Histogram"))
  
  curvature_distribution = extractMengerCurvatureDistribution(planned_paths, string(prefix, " Menger Curvature Histogram"))
  printfmt(string(prefix, " Menger KL divergence = {}\n"), kl_divergence(test_curvature_distribution, curvature_distribution))
  curvature_distribution = extractSpiralCurvatureDistribution(path_control_actions, planned_paths, control_action_lut, string(prefix, " Exact Curvature Histogram"))
  printfmt(string(prefix, " Exact KL divergence = {}\n"), kl_divergence(test_curvature_distribution, curvature_distribution))
  curvature_distribution = extractCurvatureFromControlSet(control_action_lut, string(prefix, " Control Set Curvature Histogram"))
  printfmt(string(prefix, " Control Set KL divergence = {}\n"), kl_divergence(test_curvature_distribution, curvature_distribution))

  printfmt(string(prefix, " Control Set average dl score = {}\n"), average_dl_score)
  printfmt(string(prefix, " Control Set max dl score = {}\n"), max_dl_score)
  printfmt(string(prefix, " Control Set average distance score = {}\n"), total_dist_score / size(test_set, 1))
  printfmt(string(prefix, " Control Set average goal score = {}\n"), total_goal_score / size(test_set, 1))
  printfmt(string(prefix, " Control Set average bending energy score = {}\n"), total_bending_energy_score / size(test_set, 1))
  printfmt(string(prefix, " Control Set average planning time = {}\n"), total_time / size(test_set, 1))

  control_set_size::UInt64 = 0
  for i in 1:size(TI_RANGE, 1)
    control_set_size += length(control_action_lut[i])
  end
  printfmt(string(prefix, " Control Set Size = {}\n"), control_set_size)

  return path_control_actions, curvature_deviation_vals

end

# Iterates stepwise over 2 paths and computes the absolute curvature difference between them.
function compareStepwiseCurvature(planned_path::Array{Float64, 2}, test_path::Array{Float64, 2})
  curvature_1 = extractMengerCurvature(planned_path)
  curvature_2 = extractMengerCurvature(test_path)

  max_curvature_error::Float64 = 0.0
  for i = 1:min(size(curvature_1, 1), size(curvature_2, 1))
    curvature_error = abs(curvature_1[i] - curvature_2[i])
    if curvature_error > max_curvature_error
      max_curvature_error = curvature_error
    end    
  end

  return max_curvature_error
  
end

# Vertex comparison of two algorithms.
function algorithmVertexComparison()
  # Find the size of the largest control action across all sets.
  max_action_size::Int64 = 0
  for control_lut in dense_control_action_lut
    for action in values(control_lut)
      if action.line_segment_count > max_action_size
        max_action_size = action.line_segment_count
      end
    end
  end

  counts_1 = 0
  counts_2 = 0
  for i = 1:size(training_set, 1)
    for j = 1:size(training_set[1], 1)
      res1 = findClosestPath(dense_control_action_lut, training_set[i][j], TI_RANGE[i])
      counts_1 += res1[4]
      res2 = findClosestPath2(dense_control_action_lut, training_set[i][j], TI_RANGE[i], max_action_size)
      counts_2 += res2[5]
    end
  end

  @printf("Average vertex count 1 = %f\n", counts_1 / (size(training_set, 1) * size(training_set[1], 1)))
  @printf("Average vertex count 2 = %f\n", counts_2 / (size(training_set, 1) * size(training_set[1], 1)))

end


# Compare the two closest path algorithms.
function algorithmTimingComparison()
  count = 0
  time_1 = 0.0
  time_2 = 0.0
  greedy_gap = 0.0
  for theta_index in size(TI_RANGE, 1)
    for path in training_set[theta_index]
      # Compute and visualize the closest path using algorithm 1.
      CPUtic()
      result_1 = findClosestPath(dense_control_action_lut, path, TI_RANGE[theta_index]) 
      time_1 += CPUtoc()

      endpoints::Array{Float64} = []
      points::Array{Float64} = []
      endpoint_state = State(0.0, 0.0, TI_RANGE[theta_index], 0.0, 0)
      append!(endpoints, [endpoint_state.x, endpoint_state.y])
      
      point_count::Int64 = 0
      for control_id in result_1[3]
        ti_mod::Float64 = endpoint_state.theta % (pi / 2.0)
        ti_index = getClosestIndex(ti_mod, TI_RANGE)
        control_action::SpiralControlAction = dense_control_action_lut[ti_index][control_id]
        delta_theta = endpoint_state.theta - control_action.ti
    
        # Skip the first point to avoid duplicates
        for i = 2:size(control_action.path, 1)
          point = transformControlActionPoint(control_action, endpoint_state, i)
          append!(points, point)
          point_count += 1
          if point_count == size(path, 1)
            break
          end
        end
    
        if point_count == size(path, 1)
          break
        end
    
        endpoint_state = transformControlActionEnd(control_action, delta_theta, endpoint_state.x, endpoint_state.y, 0)  
    
        append!(endpoints, [endpoint_state.x, endpoint_state.y])
      end
    
      endpoints_final::Array{Float64, 2} = permutedims(reshape(endpoints, 2, :), [2, 1])
      points_final::Array{Float64, 2} = permutedims(reshape(points, 2, :), [2, 1])
      plotLatticePath(path, points_final, endpoints_final)

      # Compute and visualize the closest path using algorithm 2.
      CPUtic()
      result_2 = findClosestPath2(dense_control_action_lut, path, TI_RANGE[theta_index], max_action_size)
      time_2 += CPUtoc()

      greedy_gap += result_2[4] / result_2[1]
      endpoints = []
      points = []
      endpoint_state = State(0.0, 0.0, TI_RANGE[theta_index], 0.0, 0)
      append!(endpoints, [endpoint_state.x, endpoint_state.y])
    
      point_count = 0
      for control_id in result_2[3]
        ti_mod = endpoint_state.theta % (pi / 2.0)
        ti_index = getClosestIndex(ti_mod, TI_RANGE)
        control_action::SpiralControlAction = dense_control_action_lut[ti_index][control_id]
        delta_theta = endpoint_state.theta - control_action.ti
    
        # Skip the first point to avoid duplicates
        for i = 2:size(control_action.path, 1)
          point = transformControlActionPoint(control_action, endpoint_state, i)
          append!(points, point)
          point_count += 1
          if point_count == size(path, 1)
            break
          end
        end
    
        if point_count == size(path, 1)
          break
        end
    
        endpoint_state = transformControlActionEnd(control_action, delta_theta, endpoint_state.x, endpoint_state.y, 0)  
    
        append!(endpoints, [endpoint_state.x, endpoint_state.y])
      end
    
      endpoints_final = permutedims(reshape(endpoints, 2, :), [2, 1])
      points_final = permutedims(reshape(points, 2, :), [2, 1])
      plotLatticePath(path, points_final, endpoints_final)
      
      try
        @assert(abs(result_2[1] - result_1[1]) < 0.01)
      catch
        @printf("Result 1 = %f\n", result_1[1])
        @printf("Result 2 = %f\n", result_2[1])
      end

      @printf("\n")
      count += 1

    end

  end

  @printf("Average Planning Time 1 = %f\n", time_1 / count)
  @printf("Average Planning Time 2 = %f\n", time_2 / count)
  @printf("Average Greedy Gap = %f\n", greedy_gap / count)

end

# Plots a closest path algorithmic run figure.
function closestPathGenerationExperiment()
    proc_index = 27
    result = findClosestPath(dense_control_action_lut, training_set[1][27], TI_RANGE[1])
    @printf("\n")
    show(result[1]) 
    @printf("\n")
    endpoints::Array{Float64} = []
    points::Array{Float64} = []
    endpoint_state = State(0.0, 0.0, TI_RANGE[1], 0.0, 0)
    append!(endpoints, [endpoint_state.x, endpoint_state.y])
  
    point_count::Int64 = 0
    for control_id in result[3]
      ti_mod::Float64 = endpoint_state.theta % (pi / 2.0)
      ti_index = getClosestIndex(ti_mod, TI_RANGE)
      control_action::SpiralControlAction = dense_control_action_lut[ti_index][control_id]
      delta_theta = endpoint_state.theta - control_action.ti
  
      # Skip the first point to avoid duplicates
      for i = 2:size(control_action.path, 1)
        point = transformControlActionPoint(control_action, endpoint_state, i)
        append!(points, point)
        point_count += 1
        @printf("\n")
        show(point)
        @printf("\n")
        if point_count == size(training_set[1][27], 1)
          break
        end
      end
  
      if point_count == size(training_set[1][27], 1)
        break
      end
  
      endpoint_state = transformControlActionEnd(control_action, delta_theta, endpoint_state.x, endpoint_state.y, 0)  
  
      append!(endpoints, [endpoint_state.x, endpoint_state.y])
    end
  
    endpoints_final::Array{Float64, 2} = permutedims(reshape(endpoints, 2, :), [2, 1])
    points_final::Array{Float64, 2} = permutedims(reshape(points, 2, :), [2, 1])
    plotLatticePath(training_set[1][27], points_final, endpoints_final)

end

# Compares the curvature distribution between an aggressive and passive
# control set.
function aggressivePassiveControlSetCurvatureComparison()
  aggressive_dense_set = generateControlActions(0.8)
  plotControlSetLUT(aggressive_dense_set, "Aggressive Control Set")
  extractCurvatureFromControlSet(aggressive_dense_set, "Aggressive")

  passive_dense_set = generateControlActions(0.2) 
  plotControlSetLUT(passive_dense_set, "Passive Dense Set")
  extractCurvatureFromControlSet(passive_dense_set, "Passive")

  passive_mutual_control_action_lut::Array{Dict{UInt128, SpiralControlAction}} = []
  aggressive_mutual_control_action_lut::Array{Dict{UInt128, SpiralControlAction}} = []
  for i = 1:size(TI_RANGE, 1)
    push!(passive_mutual_control_action_lut, Dict{UInt128, SpiralControlAction}())
    push!(aggressive_mutual_control_action_lut, Dict{UInt128, SpiralControlAction}())
  end

  for i in 1:size(TI_RANGE, 1)
    @printf("%d\n", i)
    for (control_id_1, control_action_1) in passive_dense_set[i]
      for (control_id_2, control_action_2) in aggressive_dense_set[i]
        if (control_id_1 & 0x3FFFFFFFFFFFF) == (control_id_2 & 0x3FFFFFFFFFFFF)
          passive_mutual_control_action_lut[i][control_id_1] = control_action_1
          aggressive_mutual_control_action_lut[i][control_id_2] = control_action_2
          break
        end
      end 
    end
  end

  dist1 = extractCurvatureFromControlSet(passive_mutual_control_action_lut, "Passive Mutual") 
  dist2 = extractCurvatureFromControlSet(aggressive_mutual_control_action_lut, "Aggressive Mutual") 
  @printf("Control set kl divergence = %f\n", kl_divergence(dist1, dist2) + kl_divergence(dist2, dist1))

end

end # module
