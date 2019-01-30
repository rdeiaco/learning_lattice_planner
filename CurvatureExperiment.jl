module CurvatureExperiment

export compareCurvatureLimits

using Constants, OptimizationPlanner, Utils, LatticeVisualizer, LatticeOccupancyGrid, PyPlot, Distances, ParameterComparison

function compareCurvatureLimits()

  occupancy_grid::OccupancyGrid = OccupancyGrid(0.0, 0.0, 60.0, false, true, true, 200, true)   
  kmax_vals = [0.1, 0.01] 
  #lookahead_vals = [8.0, 10.0, 12.0, 14.0, 16.0]
  lookahead_vals = [8.0, 18.0]
  densities::Array{Array{Float64}} = Array{Array{Float64}}(size(kmax_vals, 1))
  circle_densities::Array{Array{Float64}} = Array{Array{Float64}}(size(kmax_vals, 1))

  for i = 1:size(kmax_vals, 1)
    plan = planOptimizerPath(occupancy_grid.center_line, occupancy_grid, kmax_vals[i], lookahead_vals[i])
    planned_path = plan[1]
    curvature_vals = plan[2]

    plotPathOnOccupancy(planned_path, occupancy_grid, "Kmax = $(kmax_vals[i]), Lookahead = $(lookahead_vals[i]) Path Plot")

    fig = figure()
    ax = axes()
    plot(planned_path[:, 1], curvature_vals)
    xlabel("X Values")
    ylabel("Curvature Values")
    title("Kmax = $(kmax_vals[i]), Lookahead = $(lookahead_vals[i]) Curvature Plot")

    savefig("Kmax = $(kmax_vals[i]), Lookahead = $(lookahead_vals[i]) Curvature Plot.png")

    clf()
    close()

    fig = figure()
    ax = axes()
    result = plt[:hist](curvature_vals, bins=200, range=(-0.2, 0.2), density=true)
    xlabel("Curvature")
    ylabel("Frequency")
    title("Double Swerve Histogram")
    density = result[1]

    for j = 1:size(density, 1)
      density[j] = density[j] / 100.0
      if density[j] < 1e-10
        density[j] = 1e-10
      end
    end

    densities[i] = density

    savefig("Kmax = $(kmax_vals[i]), Lookahead = $(lookahead_vals[i]) Histogram.png")

    clf()
    close()

    density = extractMengerCurvatureDistribution([planned_path], "Kmax = $(kmax_vals[i]), Lookahead = $(lookahead_vals[i])")

    circle_densities[i] = density
    
  end

  kl_matrix::Array{Float64, 2} = Array{Float64, 2}(size(kmax_vals, 1), size(kmax_vals, 1))

  for i = 1:size(kmax_vals, 1)
    for j = 1:size(kmax_vals, 1)
      kl_matrix[i, j] = kl_divergence(densities[i], densities[j]) + kl_divergence(densities[j], densities[i])
    end
  end

  for i = 1:size(kmax_vals, 1)
    for j = 1:size(kmax_vals, 1)
      @printf("%f ", kl_matrix[i, j])
    end
    @printf("\n")
  end
  @printf("\n")

end

end # module
