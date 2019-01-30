module LatticeVisualizer

export plotControlSetLUT, plotLatticePath, plotProcessedPaths, plotRawPaths, plotOccupancyGrid, plotControlSwath, plotControlActionSequence, plotControlActionsOnOccupancy, plotPathOnOccupancy, plotPathAndSwathOnOccupancy, plotMultipleControlActionsOnOccupancy, plotLattice, plotGeneratedPaths, plotClusters

using PyPlot, Constants, LatticeState, PolynomialSpiral, LatticeOccupancyGrid, Utils, SwathGenerator, LatticePathCluster

function plotPathAndSwathOnOccupancy(path::Array{Float64, 2}, swath::Set{Tuple{UInt64, UInt64}}, occupancy_grid::OccupancyGrid, title_string::String="Path on Occupancy Grid")
  fig = figure()
  ax = subplot(111)
  xlabel("x")
  ylabel("y")
  title(title_string)
  axis([GRID_X_START, GRID_X_END, GRID_Y_START, GRID_Y_END])

  for indices in swath
    point::Array{Float64} = getGridPoint(indices)
    #@assert(occupancy_grid.grid[indices[1], indices[2]] < Inf)
    scatter(point[1], point[2], s=1, color="c")
  end

  for i::UInt64 = 1:size(occupancy_grid.grid, 1)
    for j::UInt64 = 1:size(occupancy_grid.grid, 2)
      if occupancy_grid.grid[i, j] == Inf
        point::Array{Float64} = getGridPoint((i, j))
        scatter(point[1], point[2], s=1, color="r")
      end
    end
  end

  plot(path[:, 1], path[:, 2])
  #plot(center_line[:, 1], center_line[:, 2])

  scatter(occupancy_grid.goal.x, occupancy_grid.goal.y, color="g")
  plot([occupancy_grid.goal.x, occupancy_grid.goal.x + cos(occupancy_grid.goal.theta)],
    [occupancy_grid.goal.y, occupancy_grid.goal.y + sin(occupancy_grid.goal.theta)],
    color="brown")

  savefig(string(title_string, ".png"))
  
  # Close this figure.
  clf()
  close()

end

function plotPathOnOccupancy(path::Array{Float64, 2}, occupancy_grid::OccupancyGrid, title_string::String="Path on Occupancy Grid")
  fig = figure()
  ax = subplot(111)
  xlabel("x")
  ylabel("y")
  title(title_string)
  axis([GRID_X_START, GRID_X_END, GRID_Y_START, GRID_Y_END])

  for i::UInt64 = 1:size(occupancy_grid.grid, 1)
    for j::UInt64 = 1:size(occupancy_grid.grid, 2)
      if occupancy_grid.grid[i, j] == Inf
        point::Array{Float64} = getGridPoint((i, j))
        scatter(point[1], point[2], s=1, color="r")
      end
    end
  end

  plot(path[:, 1], path[:, 2])

  scatter(occupancy_grid.goal.x, occupancy_grid.goal.y, color="g")
  plot([occupancy_grid.goal.x, occupancy_grid.goal.x + cos(occupancy_grid.goal.theta)],
    [occupancy_grid.goal.y, occupancy_grid.goal.y + sin(occupancy_grid.goal.theta)],
    color="brown")

  savefig(string(title_string, ".png"))
  
  # Close this figure.
  clf()
  close()

end

function plotControlActionsOnOccupancy(swath_lut::Dict{UInt128, Set{Tuple{UInt64, UInt64}}}, 
  control_action_lut::Array{Dict{UInt128, SpiralControlAction}}, 
  control_sequence::Array{UInt128}, 
  occupancy_grid::OccupancyGrid, title_string::String="Path On Occupancy Grid")
  fig = figure()
  ax = subplot(111)
  xlabel("x")
  ylabel("y")
  title(title_string)
  axis([GRID_X_START, GRID_X_END, GRID_Y_START, GRID_Y_END])

  for i::UInt64 = 1:size(occupancy_grid.grid, 1)
    for j::UInt64 = 1:size(occupancy_grid.grid, 2)
      if occupancy_grid.grid[i, j] == Inf
        point::Array{Float64} = getGridPoint((i, j))
        scatter(point[1], point[2], s=1, color="r")
      end
    end
  end

  start_state::State = State(0.0, 0.0, 0.0, 0.0, 0)
  path::Array{Float64} = []

  for control_id in control_sequence
    ti_mod::Float64 = start_state.theta % (pi / 2.0)
    ti_index = getClosestIndex(ti_mod, TI_RANGE)

    #transformed_swath::Set{Tuple{UInt64, UInt64}} = transformSwath(swath_lut[control_id], 
    #  start_state, control_action_lut[ti_index][control_id])
    #for indices in transformed_swath
    #  point::Array{Float64} = getGridPoint(indices)
    #  @assert(occupancy_grid.grid[indices[1], indices[2]] < Inf)
    #  scatter(point[1], point[2], s=1, color="c")
    #end

    control_action::SpiralControlAction = control_action_lut[ti_index][control_id]
    delta_theta::Float64 = start_state.theta - control_action.ti
    delta_theta_multiple::Float64 = delta_theta / (pi / 2.0)
    @assert(abs(round(delta_theta_multiple) - delta_theta_multiple) < 0.01)
    end_state::State = transformControlActionEnd(control_action, delta_theta, start_state.x, start_state.y, start_state.steps)

    for i = 1:size(control_action.path, 1)
      point = transformControlActionPoint(control_action, start_state, i)
      append!(path, point)
    end

    start_state = end_state

  end

  path_permuted::Array{Float64, 2} = permutedims(reshape(path, 2, :), [2, 1])

  plot(path_permuted[:, 1], path_permuted[:, 2])

  scatter(occupancy_grid.goal.x, occupancy_grid.goal.y, color="g")
  plot([occupancy_grid.goal.x, occupancy_grid.goal.x + cos(occupancy_grid.goal.theta)],
    [occupancy_grid.goal.y, occupancy_grid.goal.y + sin(occupancy_grid.goal.theta)],
    color="brown")

  savefig(string(title_string, ".png"))
  
  # Close this figure.
  clf()
  close()

end

function plotControlActionSequence(control_action_lut::Array{Dict{UInt128, SpiralControlAction}},
  control_sequence::Array{UInt128})
  fig = figure()
  ax = subplot(111)
  xlabel("x")
  ylabel("y")
  title("Control Action Sequence")
  axis([GRID_X_START, GRID_X_END, GRID_Y_START, GRID_Y_END])

  start_state::State = State(0.0, 0.0, 0.0, 0.0, 0)
  path::Array{Float64} = []


  for control_id in control_sequence
    ti_mod::Float64 = start_state.theta % (pi / 2.0)
    ti_index = getClosestIndex(ti_mod, TI_RANGE)

    control_action::SpiralControlAction = control_action_lut[ti_index][control_id]
    delta_theta::Float64 = start_state.theta - control_action.ti
    delta_theta_multiple::Float64 = delta_theta / (pi / 2.0)
    @assert(abs(round(delta_theta_multiple) - delta_theta_multiple) < 0.01)
    end_state::State = transformControlActionEnd(control_action, delta_theta, start_state.x, start_state.y, start_state.steps)

    for i = 1:size(control_action.path, 1)
      point = transformControlActionPoint(control_action, start_state, i)
      @show((point[1], point[2]))
      @printf("\n")
      append!(path, point)
    end

    @printf("End point = ")
    show((end_state.x, end_state.y, end_state.theta))
    @printf("\n")

    start_state = end_state

  end

  path_permuted::Array{Float64, 2} = permutedims(reshape(path, 2, :), [2, 1])

  plot(path_permuted[:, 1], path_permuted[:, 2])

  show()

end

function plotOccupancyGrid(occupancy_grid::OccupancyGrid)
  fig = figure()
  ax = subplot(111)
  xlabel("x")
  ylabel("y")
  title("Occupancy Grid")
  axis([GRID_X_START, GRID_X_END, GRID_Y_START, GRID_Y_END])

  for i::UInt64 = 1:size(occupancy_grid.grid, 1)
    for j::UInt64 = 1:size(occupancy_grid.grid, 2)
      if occupancy_grid.grid[i, j] == Inf
        point::Array{Float64} = getGridPoint((i, j))
        scatter(point[1], point[2], s=1, color="r")
      end
    end
  end

  scatter(occupancy_grid.goal.x, occupancy_grid.goal.y, color="g")
  plot([occupancy_grid.goal.x, occupancy_grid.goal.x + cos(occupancy_grid.goal.theta)],
    [occupancy_grid.goal.y, occupancy_grid.goal.y + sin(occupancy_grid.goal.theta)],
    color="b")

  show()

end

function plotControlSwath(control_action::SpiralControlAction, swath::Set{Tuple{UInt64, UInt64}})
  fig = figure()
  ax = subplot(111)
  xlabel("x")
  ylabel("y")
  title("Control Swath")
  axis([-GRID_LENGTH / 2.0, GRID_LENGTH / 2.0, -GRID_WIDTH / 2.0, GRID_WIDTH / 2.0])
  
  for indices in swath
    point::Array{Float64} = getGridPoint(indices)
    scatter(point[1], point[2], color="b")
  end

  plot(control_action.path[:, 1], control_action.path[:, 2], color="r")

  show()

end

function plotControlSetLUT(control_set_lut::Array{Dict{UInt128, SpiralControlAction}}, plot_title="Control Set")
  fig = figure()
  ax = subplot(111)
  xlabel("x")
  ylabel("y")
  title(plot_title)
  axis([-1.0, 5.0, -1.0, 5.0])

  colours = ["blue", "green", "red", "cyan", "magenta", "black"]
  colour_count = 1
  for control_set in control_set_lut
    colour = colours[colour_count]
    colour_count += 1
    for (control_id, control_action) in control_set
      plot(control_action.path[:, 1], control_action.path[:, 2], color=colour) 
    end
  end

  savefig(string(plot_title, ".png"))
  
  # Close this figure.
  clf()
  close()

end

function plotLatticePaths(path::Array{Float64, 2}, lattice_results::Array{Tuple{Array{Float64, 2}, Array{Float64, 2}}})

  plot_count::UInt64 = 1
  for result in lattice_results
    fig = figure()
    ax = subplot(111)
    ax[:xaxis][:set_ticks](collect(linspace(0.0, 16.0, round(Int64, 16.0/RESOLUTION) + 1)))
    ax[:yaxis][:set_ticks](collect(linspace(-10.0, 10.0, round(Int64, 20.0/RESOLUTION) + 1)))
    xlabel("x")
    ylabel("y")
    title("Polygonal Path and Lattice Path $(plot_count)")

    plot(path[:, 1], path[:, 2], color="red", label="Polygonal Path")
    plot(result[1][:, 1], result[1][:, 2], label="result $(plot_count)")
    scatter(result[2][:, 1], result[2][:, 2])
 
    axis([0.0, 14.0, -8.0, 8.0])
    grid()
    legend()
    
    plot_count += 1
  end

  show()

end

function plotLatticePath(p_d::Array{Float64, 2}, p_l::Array{Float64, 2}, endpoints::Array{Float64, 2}, title_string="Closest Path Comparison")
  fig = figure()
  ax = subplot(111)
  #ax[:xaxis][:set_ticks](collect(linspace(-0.8, 11.2, round(Int64, 12.0/RESOLUTION) + 1)))
  #ax[:yaxis][:set_ticks](collect(linspace(-6.0, 6.0, round(Int64, 12.0/RESOLUTION) + 1)))
  xlabel("x")
  ylabel("y")
  title("Polygonal Path and Lattice Path")

  plot(p_d[:, 1], p_d[:, 2], color="red", label="Pd")
  plot(p_l[:, 1], p_l[:, 2], color="blue", label="Pl")
  scatter(endpoints[:, 1], endpoints[:, 2])

  axis([-0.8, 11.2, -6.0, 6.0])
  #grid()
  legend()

  savefig(string(title_string, ".png"))

end

function plotRawPaths(paths::Array{Array{Float64, 2}}, title_string::String="Raw Paths from Test Set")
  fig = figure()
  ax = subplot(111)

#  for i = 1:size(TI_RANGE, 1)
  for i = 1:1
    for j = 1:size(paths, 1)
      temp_x::Array{Float64} = []
      temp_y::Array{Float64} = []
      for k = 1:size(paths[j], 1)
        push!(temp_x, paths[j][k, 1]*cos(TI_RANGE[i]) - paths[j][k, 2]*sin(TI_RANGE[i]))
        push!(temp_y, paths[j][k, 1]*sin(TI_RANGE[i]) + paths[j][k, 2]*cos(TI_RANGE[i]))
      end

      plot(temp_x, temp_y)

    end
  end

  xlabel("x")
  ylabel("y")
  title(title_string)

  axis([-10.0, 80.0, -20.0, 70.0])
  
  savefig(string(title_string, ".png"))
  
  # Close this figure.
  clf()
  close()

end

function plotProcessedPaths(paths::Array{Array{Array{Float64, 2}}}, title_string::String="Processed Paths Segments from Training Set")
  fig = figure()
  ax = subplot(111)

  colours = ["blue", "green", "red", "cyan", "magenta", "black"]
  
  for i = 1:size(paths, 1)
    for j = 1:size(paths, 1)
      plot(paths[i][j][:, 1], paths[i][j][:, 2], color=colours[i])
    end
  end

  xlabel("x")
  ylabel("y")
  title(title_string)

  axis([0.0, 20.0, -10.0, 10.0])
  
  savefig(string(title_string, ".png"))
  
  # Close this figure.
  clf()
  close()

end

function plotMultipleControlActionsOnOccupancy(swath_lut::Dict{UInt128, Set{Tuple{UInt64, UInt64}}}, 
  control_action_luts::Array{Array{Dict{UInt128, SpiralControlAction}, 1}, 1}, 
  path_sequences::Array{Array{Array{UInt128}, 1}, 1}, 
  occupancy_grids::Array{OccupancyGrid}, 
  raw_grid_indices::Array{Int64},
  title_string::String="Path On Occupancy Grid ")

  colours = ["blue", "cyan", "magenta", "orange"]
  labels = ["Dense", "DL", "L1", "L2"]

  for l = 1:size(path_sequences[1], 1)
    fig = figure()
    ax = subplot(111)
    xlabel("x")
    ylabel("y")
    title(title_string)
    axis([GRID_X_START, GRID_X_END, GRID_Y_START, GRID_Y_END])

    # Plot the occupancy grid.
    occupancy_grid::OccupancyGrid = occupancy_grids[raw_grid_indices[l]]
    for i::UInt64 = 1:size(occupancy_grid.grid, 1)
      for j::UInt64 = 1:size(occupancy_grid.grid, 2)
        if occupancy_grid.grid[i, j] == Inf
          point::Array{Float64} = getGridPoint((i, j))
          scatter(point[1], point[2], s=1, color="r")
        end
      end
    end

    # For each type of lattice control set, plot its associated path.
    for k = 1:size(path_sequences, 1)
      start_state::State = State(0.0, 0.0, 0.0, 0.0, 0)
      path::Array{Float64} = []
      control_sequence = path_sequences[k][l]

      for control_id in control_sequence
        ti_mod::Float64 = start_state.theta % (pi / 2.0)
        ti_index = getClosestIndex(ti_mod, TI_RANGE)

        control_action::SpiralControlAction = control_action_luts[k][ti_index][control_id]
        delta_theta::Float64 = start_state.theta - control_action.ti
        delta_theta_multiple::Float64 = delta_theta / (pi / 2.0)
        @assert(abs(round(delta_theta_multiple) - delta_theta_multiple) < 0.01)
        end_state::State = transformControlActionEnd(control_action, delta_theta, start_state.x, start_state.y, start_state.steps)

        for i = 1:size(control_action.path, 1)
          point = transformControlActionPoint(control_action, start_state, i)
          append!(path, point)
        end

        start_state = end_state

      end

      path_permuted::Array{Float64, 2} = permutedims(reshape(path, 2, :), [2, 1])

      plot(path_permuted[:, 1], path_permuted[:, 2], color=colours[k], label=labels[k])

    end
    
    # Plot the goal state.
    scatter(occupancy_grid.goal.x, occupancy_grid.goal.y, color="g")
    plot([occupancy_grid.goal.x, occupancy_grid.goal.x + cos(occupancy_grid.goal.theta)],
      [occupancy_grid.goal.y, occupancy_grid.goal.y + sin(occupancy_grid.goal.theta)],
      color="brown")

    legend(fontsize="small")

    savefig(string(string(title_string, string(l)), ".png"))
    
    # Close this figure.
    clf()
    close()

  end    

end

function plotLattice(control_action_lut::Array{Dict{UInt128, SpiralControlAction}})
  # Plot 3 layers
  colours = ["red", "blue", "green"]

  fig = figure()
  ax = subplot(111)
  xlabel("x")
  ylabel("y")
  title("Lattice Plot")
  axis([0.0, 14.0, -7.0, 7.0])

  vertexList = [(0.0, 0.0, 0.0)]

  for i = 1:3
    tempVertexList = []
    for vertex in vertexList
      for control_action in values(control_action_lut[1])
        # Focus on theta = 0.0 to simplify plot.
        if abs(control_action.tf - TI_RANGE[1]) > 0.001
          continue
        end
        if abs(control_action.ti - TI_RANGE[1]) > 0.001
          continue
        end
        
        x_vals = []
        y_vals = [] 

        # Translate the paths.
        for j = 1:size(control_action.path, 1)
          xj = control_action.path[j, 1] 
          yj = control_action.path[j, 2] 
          push!(x_vals, xj + vertex[1])
          push!(y_vals, yj + vertex[2])
        end

        plot(x_vals, y_vals, color=colours[i])
        push!(tempVertexList, (vertex[1]+control_action.xf, vertex[2]+control_action.yf, control_action.tf))

      end
    end

    vertexList = tempVertexList
    @printf("\n")
    show(vertexList)
    @printf("\n")

  end

  savefig("Lattice Plot.png")

  # Close this figure.
  clf()
  close()
end

function plotGeneratedPaths(gen_paths::Array{Array{Float64, 2}})
  fig = figure()
  ax = subplot(111)
  xlabel("x")
  ylabel("y")
  title("Generated Path Plot")
  axis([GRID_X_START, GRID_X_END, GRID_Y_START, GRID_Y_END])

  for i = 1:size(gen_paths, 1)
    plot(gen_paths[i][:, 1], gen_paths[i][:, 2])
  end

  savefig("generated_paths.png")
  
  # Close this figure.
  clf()
  close()

end

function plotClusters(clusters::Array{PathCluster}, title_string="Cluster Plot")
  # Plot the cluster paths.
  fig = figure()
  xlabel("x")
  ylabel("y")
  title(title_string)
  for i = 1:size(clusters, 1)
    if size(clusters[i].paths, 1) < 1
      continue
    end

    colour::String = string("#", hex(rand(1:0xFFFFFF), 6))
    for path in clusters[i].paths
      plot(path[:, 1], path[:, 2], color=colour)
    end
  end

  # Plot the mean path of each cluster.
  for i = 1:size(clusters, 1)
    if size(clusters[i].paths, 1) < 1
      continue
    end

    plot(clusters[i].mean_path[:, 1], clusters[i].mean_path[:, 2], color="k", linestyle="dashed")
  end

  savefig(string(title_string, ".png"))

  clf()
  close()

end

end # module
