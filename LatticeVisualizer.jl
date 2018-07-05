module LatticeVisualizer

export plotControlSetLUT, plotLatticePath, plotPaths, plotOccupancyGrid, plotControlSwath, plotControlActionSequence, plotControlActionsOnOccupancy, plotPathOnOccupancy

using PyPlot, Constants, LatticeState, PolynomialSpiral, LatticeOccupancyGrid, Utils, SwathGenerator

function plotPathOnOccupancy(path::Array{Float64, 2}, occupancy_grid::OccupancyGrid, title_string::String="Path on Occupancy Grid")
  fig = figure()
  ax = subplot(111)
  xlabel("x")
  ylabel("y")
  title(title_string)
  axis([GRID_X_START, GRID_X_END, GRID_Y_START, GRID_Y_END])

  #for indices in swath
  #  point::Array{Float64} = getGridPoint(indices)
  #  #@assert(occupancy_grid.grid[indices[1], indices[2]] < Inf)
  #  scatter(point[1], point[2], s=1, color="c")
  #end

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

  #savefig(string(title_string, ".png"))
  show()
  
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

    transformed_swath::Set{Tuple{UInt64, UInt64}} = transformSwath(swath_lut[control_id], 
      start_state, control_action_lut[ti_index][control_id])
    for indices in transformed_swath
      point::Array{Float64} = getGridPoint(indices)
      @assert(occupancy_grid.grid[indices[1], indices[2]] < Inf)
      scatter(point[1], point[2], s=1, color="c")
    end

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
  axis([-6.0, 6.0, -6.0, 6.0])
  for control_set in control_set_lut
    for (control_id, control_action) in control_set
      plot(control_action.path[:, 1], control_action.path[:, 2]) 
    end
  end

  show()

end

function plotLatticePath(path::Array{Float64, 2}, lattice_results::Array{Tuple{Array{Float64, 2}, Array{Float64, 2}}})

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

function plotPaths(paths::Array{Array{Float64, 2}}, title_string::String="Polygonal Paths from Dataset")
  fig = figure()
  ax = subplot(111)

  for i = 1:size(paths, 1)
    plot(paths[i][:, 1], paths[i][ :, 2])
  end

  xlabel("x")
  ylabel("y")
  title(title_string)

  axis([0.0, 12.0, -10.0, 10.0])

  savefig(string(title_string, ".png"))
  
  # Close this figure.
  clf()
  close()

end

end # module
