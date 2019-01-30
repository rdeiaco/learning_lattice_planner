module ParameterComparison

using Constants, Utils, PolynomialSpiral, PyPlot, DubinsPath, Geometry, LatticePathFinder

export extractCurvatureDistribution, extractSpiralCurvatureDistribution, extractMengerCurvatureDistribution, extractCurvatureFromControlSet, mengerCurvature, extractCurvatureFromDubinsControlSet, computeMatchingDistribution, computeCurvatureMatchingDistribution, extractMengerCurvature, computeCurvatureDeviationValues

function extractCurvatureDistribution(paths::Array{Array{Float64, 2}}, title_string="Histogram")
  curvature_vals::Array{Float64} = []
  local x_vals
  local y_vals
  local curvature
  for path in paths
    x_vals = path[:, 1]
    y_vals = path[:, 2]

    x_grad::Array{Float64} = Array{Float64}(size(x_vals, 1))
    y_grad::Array{Float64} = Array{Float64}(size(y_vals, 1))
    x_grad2::Array{Float64} = Array{Float64}(size(x_vals, 1))
    y_grad2::Array{Float64} = Array{Float64}(size(y_vals, 1))
    curvature::Array{Float64} = Array{Float64}(size(y_vals, 1))

    for i = 2:size(x_vals, 1)-1
      x_grad[i] = (x_vals[i+1] - x_vals[i-1]) / 2.0
    end
    x_grad[1] = x_vals[2] - x_vals[1] 
    x_grad[size(x_vals, 1)] = x_vals[size(x_vals, 1)] - x_vals[size(x_vals, 1)-1]

    for i = 2:size(y_vals, 1)-1
      y_grad[i] = (y_vals[i+1] - y_vals[i-1]) / 2.0
    end
    y_grad[1] = y_vals[2] - y_vals[1] 
    y_grad[size(y_vals, 1)] = y_vals[size(y_vals, 1)] - y_vals[size(y_vals, 1)-1]

    for i = 2:size(x_grad, 1)-1
      x_grad2[i] = (x_grad[i+1] - x_grad[i-1]) / 2.0
    end
    x_grad2[1] = x_grad[2] - x_grad[1] 
    x_grad2[size(x_grad, 1)] = x_grad[size(x_grad, 1)] - x_grad[size(x_grad, 1)-1]

    for i = 2:size(y_grad, 1)-1
      y_grad2[i] = (y_grad[i+1] - y_grad[i-1]) / 2.0
    end
    y_grad2[1] = y_grad[2] - y_grad[1] 
    y_grad2[size(y_grad, 1)] = y_grad[size(y_grad, 1)] - y_grad[size(y_grad, 1)-1]

    for i = 1:size(curvature, 1)
      curvature[i] = x_grad[i]*y_grad2[i] - y_grad[i]*x_grad2[i] / (x_grad[i]^2 + y_grad[i]^2)^1.5
    end

    append!(curvature_vals, lowPassFilter(curvature))
 
  end 

  fig = figure()
  ax = axes()
  result = plt[:hist](curvature_vals, 500, (-0.5, 0.5), true)
  xlabel("Curvature")
  ylabel("Frequency")
  title(title_string)
  density = result[1]

  for i = 1:size(density, 1)
    density[i] = density[i] / 100.0
    if density[i] < 1e-10
      density[i] = 1e-10
    end
  end

  savefig(string(title_string, ".png"))

  clf()
  close()

  #fig = figure()
  #ax = axes()
  #plot(x_vals, y_vals)
  #xlabel("X Values")
  #ylabel("Y Values")
  #title("Path Plot")
  #savefig(string(title_string, " Path Plot.png"))
  #clf()
  #close()

  #fig = figure()
  #ax = axes()
  #plot(x_vals, curvature)
  #xlabel("X Values")
  #ylabel("Curvature Values")
  #title("Curvature Plot")
  #savefig(string(title_string, " Curvature Plot.png"))
  #clf()
  #close()

  #fig = figure()
  #ax = axes()
  #plot(x_vals, lowPassFilter(curvature))
  #xlabel("X Values")
  #ylabel("Curvature Values")
  #title("Filtered Curvature Plot")
  #savefig(string(title_string, " Filtered Curvature Plot.png"))
  #clf()
  #close()

  return density

end

function extractSpiralCurvatureDistribution(paths::Array{Array{UInt128}}, path_positions::Array{Array{Float64, 2}}, control_action_lut::Array{Dict{UInt128, SpiralControlAction}}, title_string="Spiral")
  curvature_vals::Array{Float64} = []

  local x_vals
  local y_vals
  local curvature

  for i in 1:size(paths, 1)
    curvature::Array{Float64} = []
    # Starts off with zero curvature
    push!(curvature, 0.0)
    theta = 0.0
    for control_action_id in paths[i]
      ti_index = getClosestIndex(theta, TI_RANGE)
      control_action = control_action_lut[ti_index][control_action_id]

      # Skip starting point to avoid duplicates.
      for j in 2:size(control_action.path, 1)
        push!(curvature, control_action.coefficients[1] + 0.1*j*control_action.coefficients[2] + (0.1*j)^2*control_action.coefficients[3] + (0.1*j)^3*control_action.coefficients[4])
      end

      theta = control_action.tf

    end

    append!(curvature_vals, curvature)

    x_vals = path_positions[i][:, 1]
    y_vals = path_positions[i][:, 2]

  end

  fig = figure()
  ax = axes()
  result = plt[:hist](curvature_vals, 500, (-0.5, 0.5), true)
  xlabel("Curvature")
  ylabel("Frequency")
  title(title_string)
  density = result[1]

  for i = 1:size(density, 1)
    density[i] = density[i] / 100.0
    if density[i] < 1e-10
      density[i] = 1e-10
    end
  end

  savefig(string(title_string, ".png"))

  clf()
  close()

#  fig = figure()
#  ax = axes()
#  plot(x_vals, y_vals)
#  xlabel("X Values")
#  ylabel("Y Values")
#  title("Path Plot")
#  savefig(string(title_string, " Path Plot.png"))
#  clf()
#  close()
#
#  fig = figure()
#  ax = axes()
#  plot(x_vals, curvature)
#  xlabel("X Values")
#  ylabel("Curvature Values")
#  title("Curvature Plot")
#  savefig(string(title_string, " Curvature Plot.png"))
#  clf()
#  close()

  return density

end

function extractMengerCurvature(path::Array{Float64, 2})
  x_vals = path[:, 1]
  y_vals = path[:, 2]

  curvature::Array{Float64} = Array{Float64}(size(y_vals, 1))

  for i = STEP_RADIUS+1:size(curvature, 1)-STEP_RADIUS
    p1 = [x_vals[i-STEP_RADIUS], y_vals[i-STEP_RADIUS]]
    p2 = [x_vals[i], y_vals[i]]
    p3 = [x_vals[i+STEP_RADIUS], y_vals[i+STEP_RADIUS]]
    curvature[i] = mengerCurvature(p1, p2, p3)

    for i = 1:STEP_RADIUS
      curvature[i] = curvature[STEP_RADIUS+1]
    end

    for i = (size(curvature, 1)-STEP_RADIUS+1):size(curvature, 1)
      curvature[i] = curvature[size(curvature, 1)-STEP_RADIUS]
    end

  end

  return curvature

end

function extractMengerCurvatureDistribution(paths::Array{Array{Float64, 2}}, title_string="Menger Curvature")
  curvature_vals::Array{Float64} = []
  local x_vals
  local y_vals
  local curvature
  for path in paths
    x_vals = path[:, 1]
    y_vals = path[:, 2]

    curvature::Array{Float64} = extractMengerCurvature(path)
    append!(curvature_vals, curvature)
  end 

  fig = figure()
  ax = axes()
  result = plt[:hist](curvature_vals, bins=500, range=(-0.5, 0.5), density=true)
  xlabel("Curvature")
  ylabel("Frequency")
  title(title_string)
  density = result[1]

  for i = 1:size(density, 1)
    density[i] = density[i] / 100.0
    if density[i] < 1e-10
      density[i] = 1e-10
    end
  end

  savefig(string(title_string, ".png"))

  clf()
  close()

  return density

end

function extractCurvatureFromControlSet(control_action_lut::Array{Dict{UInt128, SpiralControlAction}}, title_string="Control Set")
  curvature_vals::Array{Float64} = []
  for i = 1:size(control_action_lut, 1)
    for control_action in values(control_action_lut[i])
      curvature::Array{Float64} = []
      # Starts off with zero curvature
      push!(curvature, 0.0)

      # Skip starting point to avoid duplicates.
      for j in 2:size(control_action.path, 1)
        push!(curvature, control_action.coefficients[1] + 0.1*j*control_action.coefficients[2] + (0.1*j)^2*control_action.coefficients[3] + (0.1*j)^3*control_action.coefficients[4])
      end

      append!(curvature_vals, curvature)

    end

  end

  fig = figure()
  ax = axes()
  result = plt[:hist](curvature_vals, 500, (-0.5, 0.5), true)
  xlabel("Curvature")
  ylabel("Frequency")
  title(title_string)
  density = result[1]
  savefig(string(title_string, ".png"))
  clf()
  close()

  for i = 1:size(density, 1)
    density[i] = density[i] / 100.0
    if density[i] < 1e-10
      density[i] = 1e-10
    end
  end

  return density

end

function extractCurvatureFromDubinsControlSet(control_action_lut::Array{Dict{UInt128, DubinsAction}}, title_string="Control Set")

  curvature_vals::Array{Float64} = []
  for i = 1:size(control_action_lut, 1)
    for control_action in values(control_action_lut[i])
      append!(curvature_vals, control_action.curvature_vals)
    end
  end

  fig = figure()
  ax = axes()
  result = plt[:hist](curvature_vals, 500, (-1.0, 1.0), true)
  xlabel("Curvature")
  ylabel("Frequency")
  title(title_string)
  density = result[1]
  savefig(string(title_string, ".png"))
  clf()
  close()

  for i = 1:size(density, 1)
    density[i] = density[i] / 100.0
    if density[i] < 1e-10
      density[i] = 1e-10
    end
  end

  return density

end

function lowPassFilter(vals::Array{Float64})
  FILTER_ORDER = 19
  x_prev::Array{Float64} = Array{Float64}(FILTER_ORDER)
  filtered_vals::Array{Float64} = Array{Float64}(size(vals, 1))
  local curvature

  for i = 1:size(vals, 1)
    temp = vals[i]
    filtered_val = 0.0
    for j = 1:FILTER_ORDER
      filtered_val += 1.0 / (FILTER_ORDER+1) * x_prev[j]
    end
    filtered_val += 1.0 / (FILTER_ORDER+1) * temp
    filtered_vals[i] = filtered_val

    for j = FILTER_ORDER:-1:2
      x_prev[j] = x_prev[j-1] 
    end
    x_prev[1] = temp

  end

  return filtered_vals

end

function computeMatchingDistribution(control_lut::Array{Dict{UInt128, SpiralControlAction}}, test_set_processed::Array{Array{Float64, 2}}, title_string::String="Experiment 1 Dense Set Matching Distribution Histogram")
  max_dl_score::Float64 = 0.0
  total_dl_score::Float64 = 0.0
  dl_scores::Array{Float64} = []

  for i = 1:size(test_set_processed, 1)
    # Compute the closest path, as well as its matching score.
    # Keep track of the maximum distance, as well as a running total for computing the
    # average. 
    dl_res = findClosestPath(control_lut, test_set_processed[i], 0.0)
    dl_score::Float64 = dl_res[1]
    total_dl_score += dl_score
    if dl_score > max_dl_score
      max_dl_score = dl_score
    end

    # Keep track of the values to generate a histogram.
    push!(dl_scores, dl_score)
  end

  # Generate the histogram.
  fig = figure()
  ax = axes()
  result = plt[:hist](dl_scores, bins=500, range=(0.0, 2.0), density=true)
  xlabel("Matching Score")
  ylabel("Frequency")
  title(title_string)
  density = result[1]

  for i = 1:size(density, 1)
    density[i] = density[i] / 100.0
    if density[i] < 1e-10
      density[i] = 1e-10
    end
  end

  savefig(string(title_string, ".png"))

  clf()
  close()

  return (max_dl_score, total_dl_score / size(test_set_processed, 1))

end

# Plots a histogram of the curvature deviation values.
function computeCurvatureMatchingDistribution(curvature_deviation_vals::Array{Float64}, title_string::String="Experiment 1 Dense Set Curvature Matching Histogram")

  # Generate the histogram.
  fig = figure()
  ax = axes()
  result = plt[:hist](curvature_deviation_vals, bins=500, range=(0.0, 2.0), density=true)
  xlabel("Curvature Matching Score")
  ylabel("Frequency")
  title(title_string)
  density = result[1]

  for i = 1:size(density, 1)
    density[i] = density[i] / 100.0
    if density[i] < 1e-10
      density[i] = 1e-10
    end
  end

  savefig(string(title_string, ".png"))

  clf()
  close()

end

function computeCurvatureDeviationValues(dense_curvature_deviation_vals::Array{Float64}, pivtoraiko_curvature_deviation_vals::Array{Float64}, lambda_1_curvature_deviation_vals::Array{Float64}, lambda_2_curvature_deviation_vals::Array{Float64}, prefix::String="Experiment 1")
 
  straight_line_x = collect(0.0:0.05:0.4) 
  straight_line_y = straight_line_x

  # Dense vs. DL
  fig = figure()
  ax = axes()
  axis([0.05, 0.25, 0.05, 0.25])
  xlabel("Dense Curvature Matching Score (1/m)")
  ylabel("DL Curvature Matching Score (1/m)")
  title(string(prefix, " Dense vs. DL Curvature Matching Score"))
  dense_win_count = 0
  dl_win_count = 0
  for i = 1:size(dense_curvature_deviation_vals, 1)
    scatter(dense_curvature_deviation_vals[i], pivtoraiko_curvature_deviation_vals[i], color="g", s=8)
    if dense_curvature_deviation_vals[i] < pivtoraiko_curvature_deviation_vals[i]
      dense_win_count += 1
    else
      dl_win_count += 1
    end
  end
  plot(straight_line_x, straight_line_y, color="k")
  
  savefig(string(prefix, " Dense vs. DL Curvature Matching Score.png"))
  clf()
  close()
  @printf("Dense vs. DL:\n")
  @printf("Dense Win Count = %d\n", dense_win_count)
  @printf("DL Win Count = %d\n", dl_win_count)
    
  # Dense vs. L1
  fig = figure()
  ax = axes()
  axis([0.05, 0.25, 0.05, 0.25])
  xlabel("Dense Curvature Matching Score (1/m)")
  ylabel("Lambda 1 Curvature Matching Score (1/m)")
  title(string(prefix, " Dense vs. Lambda 1 Curvature Matching Score"))
  dense_win_count = 0
  l1_win_count = 0
  for i = 1:size(dense_curvature_deviation_vals, 1)
    scatter(dense_curvature_deviation_vals[i], lambda_1_curvature_deviation_vals[i], color="g", s=8)
    if dense_curvature_deviation_vals[i] < lambda_1_curvature_deviation_vals[i]
      dense_win_count += 1
    else
      l1_win_count += 1
    end
  end
  plot(straight_line_x, straight_line_y, color="k")

  savefig(string(prefix, " Dense vs. Lambda 1 Curvature Matching Score.png"))
  clf()
  close()
  @printf("Dense vs. L1:\n")
  @printf("Dense Win Count = %d\n", dense_win_count)
  @printf("Lambda 1 Win Count = %d\n", l1_win_count)

  # Dense vs. L2
  fig = figure()
  ax = axes()
  axis([0.05, 0.25, 0.05, 0.25])
  xlabel("Dense Curvature Matching Score (1/m)")
  ylabel("Lambda 2 Curvature Matching Score (1/m)")
  title(string(prefix, " Dense vs. Lambda 2 Curvature Matching Score"))
  dense_win_count = 0
  l2_win_count = 0
  for i = 1:size(dense_curvature_deviation_vals, 1)
    scatter(dense_curvature_deviation_vals[i], lambda_2_curvature_deviation_vals[i], color="g", s=8)
    if dense_curvature_deviation_vals[i] < lambda_2_curvature_deviation_vals[i]
      dense_win_count += 1
    else
      l2_win_count += 1
    end
  end
  plot(straight_line_x, straight_line_y, color="k")

  savefig(string(prefix, " Dense vs. Lambda 2 Curvature Matching Score.png"))
  clf()
  close()
  @printf("Dense vs. L2:\n")
  @printf("Dense Win Count = %d\n", dense_win_count)
  @printf("Lambda 2 Win Count = %d\n", l2_win_count)

  # DL vs. L1
  fig = figure()
  ax = axes()
  axis([0.05, 0.25, 0.05, 0.25])
  xlabel("DL Curvature Matching Score (1/m)")
  ylabel("Lambda 1 Curvature Matching Score (1/m)")
  title(string(prefix, " DL vs. Lambda 1 Curvature Matching Score"))
  dl_win_count = 0
  l1_win_count = 0
  for i = 1:size(pivtoraiko_curvature_deviation_vals, 1)
    scatter(pivtoraiko_curvature_deviation_vals[i], lambda_1_curvature_deviation_vals[i], color="g", s=8)
    if pivtoraiko_curvature_deviation_vals[i] < lambda_1_curvature_deviation_vals[i]
      dl_win_count += 1
    else
      l1_win_count += 1
    end
  end
  plot(straight_line_x, straight_line_y, color="k")

  savefig(string(prefix, " DL vs. Lambda 1 Curvature Matching Score.png"))
  clf()
  close()
  @printf("DL vs. L1:\n")
  @printf("DL Win Count = %d\n", dl_win_count)
  @printf("Lambda 1 Win Count = %d\n", l1_win_count)

  # DL vs. L2
  fig = figure()
  ax = axes()
  axis([0.05, 0.25, 0.05, 0.25])
  xlabel("DL Curvature Matching Score (1/m)")
  ylabel("Lambda 2 Curvature Matching Score (1/m)")
  title(string(prefix, " DL vs. Lambda 2 Curvature Matching Score"))
  dl_win_count = 0
  l2_win_count = 0
  for i = 1:size(pivtoraiko_curvature_deviation_vals, 1)
    scatter(pivtoraiko_curvature_deviation_vals[i], lambda_2_curvature_deviation_vals[i], color="g", s=8)
    if pivtoraiko_curvature_deviation_vals[i] < lambda_2_curvature_deviation_vals[i]
      dl_win_count += 1
    else
      l2_win_count += 1
    end
  end
  plot(straight_line_x, straight_line_y, color="k")

  savefig(string(prefix, " DL vs. Lambda 2 Curvature Matching Score.png"))
  clf()
  close()
  @printf("DL vs. L2:\n")
  @printf("DL Win Count = %d\n", dl_win_count)
  @printf("Lambda 2 Win Count = %d\n", l2_win_count)

  # L1 vs. L2
  fig = figure()
  ax = axes()
  axis([0.05, 0.25, 0.05, 0.25])
  xlabel("Lambda 1 Curvature Matching Score (1/m)")
  ylabel("Lambda 2 Curvature Matching Score (1/m)")
  title(string(prefix, " Lambda 1 vs. Lambda 2 Curvature Matching Score"))
  l1_win_count = 0
  l2_win_count = 0
  for i = 1:size(lambda_1_curvature_deviation_vals, 1)
    scatter(lambda_1_curvature_deviation_vals[i], lambda_2_curvature_deviation_vals[i], color="g", s=8)
    if lambda_1_curvature_deviation_vals[i] < lambda_2_curvature_deviation_vals[i]
      l1_win_count += 1
    else
      l2_win_count += 1
    end
  end
  plot(straight_line_x, straight_line_y, color="k")

  savefig(string(prefix, " Lambda 1 vs. Lambda 2 Curvature Matching Score.png"))
  clf()
  close()
  @printf("L1 vs. L2:\n")
  @printf("Lambda 1 Win Count = %d\n", l1_win_count)
  @printf("Lambda 2 Win Count = %d\n", l2_win_count)

end

end # module
