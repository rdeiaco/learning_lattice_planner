module PathExtractor

using Constants, Dierckx, LatticeOccupancyGrid, StatsBase

export extractGeneratedPaths, extractPaths, extractProcessedPath, extractRawPath

# Extacts paths according to a given train/test split from an array of paths.
function extractGeneratedPaths(paths::Array{Array{Float64, 2}}, train_fraction::Float64=0.8, 
  rotate_paths::Bool=true)::Tuple{Array{Array{Array{Float64, 2}}}, 
  Array{Array{Float64, 2}}, Array{Array{Float64, 2}}, Array{Int64}}
  # Go through each row in the data array, and extract paths
  # from it.
  # This is an array of path sets. Each path set corresponds to a path
  # segment in the dataset rotated to one of the initial angles in the dataset.
  training_set::Array{Array{Array{Float64, 2}, 1}, 1} = [] 
  test_set::Array{Array{Float64, 2}} = []
  test_set_processed::Array{Array{Float64, 2}} = []
  raw_grid_indices::Array{Int64} = []

  # Initialize the processed path arrays so we can push to each required
  # heading.
  # If we are not rotating the paths to the origin, then there is only 1 path set.
  if rotate_paths
    for i = 1:size(TI_RANGE, 1)
      push!(training_set, Array{Array{Float64, 2}, 1}())
    end
  else
    push!(training_set, Array{Array{Float64, 2}, 1}())
  end

  indices_to_delete::Array{Int} = []
  for i = 1:size(paths, 1)
    x_vals = paths[i][:, 1]
    y_vals = paths[i][:, 2]  

    duplicate_indices::Array{Int64} = []
    for j = 1:length(x_vals)-1
      if x_vals[j] == x_vals[j+1]
        push!(duplicate_indices, j)
      end
    end

    deleteat!(x_vals, duplicate_indices)
    deleteat!(y_vals, duplicate_indices)

    # Check if the x_vals will satisfy our interpolation requirements.
    try
      for j = 1:length(x_vals)-1
        @assert(x_vals[j+1] > x_vals[j])
      end
    catch
      @printf("Skipping path\n")
      push!(indices_to_delete, i)
      continue
    end
  end
  paths = paths[sort(collect(setdiff(Set(1:size(paths, 1)), Set(indices_to_delete)))), :]
  test_indices = sample(collect(1:size(paths, 1)), trunc(Int, (1-train_fraction) * size(paths, 1)), replace=false)

  @printf("Number of training paths = %d\n", size(paths, 1) - size(test_indices, 1))
  @printf("Number of test paths = %d\n", size(test_indices, 1))

  for i = 1:size(paths, 1)
    x_vals = paths[i][:, 1]
    y_vals = paths[i][:, 2]  
    if !(i in test_indices)
      processed_paths::Array{Array{Float64, 2}} = extractProcessedPaths(x_vals, y_vals, rotate_paths)

      if rotate_paths
        # For each heading, recreate the path.
        for j = 1:size(TI_RANGE, 1)
          ti = TI_RANGE[j]
          for k = 1:size(processed_paths, 1)
            # For each path, rotate it by the starting angle.
            path_rotated::Array{Float64, 2} = zeros(size(processed_paths[k]))
            for l = 1:size(processed_paths[k], 1)
              x1 = processed_paths[k][l, 1]
              y1 = processed_paths[k][l, 2]
              path_rotated[l, 1] = x1*cos(ti) - y1*sin(ti)
              path_rotated[l, 2] = x1*sin(ti) + y1*cos(ti)
            end
            push!(training_set[j], path_rotated)
          end
        end
      else
        # Otherwise, we are not rotating the paths to the origin.
        # Add the translated (but not rotated) paths to the training set.
        for k = 1:size(processed_paths, 1)
          push!(training_set[1], processed_paths[k])
        end
      end
    else
      push!(test_set, extractRawPath(x_vals, y_vals))
      test_set_processed_paths::Array{Array{Float64, 2}} = extractProcessedPaths(x_vals, y_vals)
      for k = 1:size(test_set_processed_paths, 1)
        push!(test_set_processed, test_set_processed_paths[k])
      end
      push!(raw_grid_indices, i)
    end

  end

  return (training_set, test_set, test_set_processed, raw_grid_indices)

end

# Extracts path from the DataFromSky csv format (which is not csv compliant).
function extractPaths(filename, train_fraction::Float64=0.8, 
  rotate_paths::Bool=true)::Tuple{Array{Array{Array{Float64, 2}}}, 
  Array{Array{Float64, 2}}, Array{Array{Float64, 2}}}

  data = readdlm(filename, ';', header=true)
  data = data[1][:, 9:end]

  path_count::Int64 = 0
 
  # Go through each row in the data array, and extract paths
  # from it.
  training_set::Array{Array{Array{Float64, 2}}} = []
  if rotate_paths
    for i = 1:size(TI_RANGE, 1)
      push!(training_set, Array{Array{Float64, 2}, 1}())
    end
  else
    push!(training_set, Array{Array{Float64, 2}, 1}())
  end

  indices_to_delete::Array{Int} = []
  test_set::Array{Array{Float64, 2}} = []
  test_set_processed::Array{Array{Float64, 2}} = []
  min_arc_length = Inf
  max_arc_length = 0.0

  for i = 1:size(data, 1)
    x_vals, y_vals = extractTransformedVals(data[i, :])
    # Check if the x_vals will satisfy our interpolation requirements.
    try
      for j = 1:length(x_vals)-1
        @assert(x_vals[j+1] > x_vals[j])
      end
      path_count += 1
    catch
      @printf("Skipping path\n")
      push!(indices_to_delete, i)
      continue
    end
  end
  data = data[sort(collect(setdiff(Set(1:size(data, 1)), Set(indices_to_delete)))), :]
  test_indices = sample(collect(1:size(data, 1)), trunc(Int, (1-train_fraction) * size(data, 1)), replace=false)

  @printf("Number of training paths = %d\n", size(data, 1) - size(test_indices, 1))
  @printf("Number of test paths = %d\n", size(test_indices, 1))

  for i = 1:size(data, 1)
    x_vals, y_vals = extractTransformedVals(data[i, :])
    if !(i in test_indices)
      processed_paths::Array{Array{Float64, 2}} = extractProcessedPaths(x_vals, y_vals, rotate_paths)
      if rotate_paths
        # For each heading, recreate the path.
        for j = 1:size(TI_RANGE, 1)
          ti = TI_RANGE[j]
          for k = 1:size(processed_paths, 1)
            # For each path, rotate it by the starting angle.
            path_rotated::Array{Float64, 2} = zeros(Float64, size(processed_paths[k]))
            for l = 1:size(processed_paths[k], 1)
              x1 = processed_paths[k][l, 1]
              y1 = processed_paths[k][l, 2]
              path_rotated[l, 1] = x1*cos(ti) - y1*sin(ti)
              path_rotated[l, 2] = x1*sin(ti) + y1*cos(ti)
            end
            push!(training_set[j], path_rotated)
          end
        end
      else
        for k = 1:size(processed_paths, 1)
          push!(training_set[1], processed_paths[k])
        end
      end
    else
      push!(test_set, extractRawPath(x_vals, y_vals))
      test_set_processed_paths::Array{Array{Float64, 2}} = extractProcessedPaths(x_vals, y_vals)
      for k = 1:size(test_set_processed_paths, 1)
        push!(test_set_processed, test_set_processed_paths[k])
      end
      arc_length = 0.0
      for i = 2:size(x_vals, 1)
        dx = x_vals[i] - x_vals[i-1]
        dy = y_vals[i] - y_vals[i-1]
        arc_length += sqrt(dx^2 + dy^2)
      end
      if arc_length > max_arc_length
        max_arc_length = arc_length
      end
      if arc_length < min_arc_length
        min_arc_length = arc_length
      end
    end

  end

  @printf("Number of paths in dataset = %d.\n", path_count)
  @printf("Min arc length = %f.\n", min_arc_length)
  @printf("Max arc length = %f.\n", max_arc_length)

  return (training_set, test_set, test_set_processed)

end

function extractProcessedPaths(x_vals::Array{Float64}, y_vals::Array{Float64}, rotate_paths::Bool=true)::Array{Array{Float64, 2}}
  # Interpolate the data using a cubic spline so we can re-sample the
  # data at the required arc length distance.
  spline = Spline1D(x_vals, y_vals, k=3, bc="error")

  # Resample the data, each time offset by 10 times the path resolution.
  offset::Float64 = 0.0
  path_available::Bool = true
  path_array::Array{Array{Float64, 2}} = []
  while(true)
    path_x::Array{Float64, 1} = []
    path_y::Array{Float64, 1} = []
    x1::Float64 = 0.0 + offset
    x2::Float64 = x1 + ARC_LENGTH_RESOLUTION

    try
      spline(x1)
    catch
      path_available = false
      @printf("spline x1 failed\n")
      @printf("x1 = %f\n", x1)
      @printf("x_vals[1] = %f\n", x_vals[1])
      @printf("y_vals[1] = %f\n", y_vals[1])
      break
    end

    push!(path_x, x1)
    push!(path_y, spline(x1))

    while length(path_x) < PATH_LENGTH
      if norm([x1, spline(x1)] - [x2, spline(x2)]) > PATH_RESOLUTION
        push!(path_x, x2)
        push!(path_y, spline(x2)) 
        x1 = x2
        x2 = x1 + ARC_LENGTH_RESOLUTION
      else
        x2 += ARC_LENGTH_RESOLUTION
      end
      
      # Check to see if we have exceeded the supplied data's range.
      try
        spline(x2)
      catch
        path_available = false
        break 
      end
    end

    if path_available == false
      break
    end

    @assert(length(path_x) == PATH_LENGTH)
    @assert(length(path_y) == PATH_LENGTH)

    # Translate the extracted path to the origin, then rotate it by the negative
    # of its initial heading to set its initial heading to zero.
    dx = path_x[2] - path_x[1]
    dy = path_y[2] - path_y[1]
    theta = atan2(dy, dx)
    x_offset = path_x[1]
    y_offset = path_y[1]
    
    # rotate_paths denotes if we rotate the path to start with zero heading.
    # If not, then we only rotate that path such that it starts in the first quadrant.
    if rotate_paths
      transformed = transformPath(path_x, path_y, -theta, x_offset, y_offset)
    else
      delta_theta = theta - (theta % (pi / 2.0))
      transformed = transformPath(path_x, path_y, -delta_theta, x_offset, y_offset)
    end
    path_x = transformed[1]
    path_y = transformed[2]

    sub_path_array::Array{Float64} = []
    for i = 1:length(path_x)
      push!(sub_path_array, path_x[i], path_y[i])
    end

    push!(path_array, permutedims(reshape(sub_path_array, 2, :), [2, 1]))

    offset += 10 * PATH_RESOLUTION 

  end

  return path_array

end

function transformPath(x_vals, y_vals, theta, x_offset, y_offset)
  x_transformed::Array{Float64, 1} = [] 
  y_transformed::Array{Float64, 1} = [] 

  for i = 1:length(x_vals)
    x_temp = x_vals[i] - x_offset
    y_temp = y_vals[i] - y_offset
    push!(x_transformed, x_temp * cos(theta) - y_temp * sin(theta))
    push!(y_transformed, x_temp * sin(theta) + y_temp * cos(theta))
  end

  return (x_transformed, y_transformed)

end

function extractRawPath(x_vals::Array{Float64}, y_vals::Array{Float64})::Array{Float64}
  raw_path::Array{Float64, 2} = Array{Float64, 2}(size(x_vals, 1), 2)

  for i = 1:size(x_vals, 1)
    raw_path[i, 1] = x_vals[i]
    raw_path[i, 2] = y_vals[i]
  end

  return raw_path

end

function extractTransformedVals(data)
    j::UInt64 = 1
    x_vals::Array{Float64, 1} = []
    y_vals::Array{Float64, 1} = []

    while (data[j] != "") && (data[j] != " ")
      push!(x_vals, data[j])
      push!(y_vals, data[j+1])
      j += 6 
    end

    dx::Float64 = x_vals[2] - x_vals[1]
    dy::Float64 = y_vals[2] - y_vals[1]
    theta::Float64 = atan2(dy, dx)
    x_offset::Float64 = x_vals[1]
    y_offset::Float64 = y_vals[1]
   
    # Translate the path to the origin, then rotate it by the negative
    # of its initial heading to set its initial heading to zero.
    transformed = transformPath(x_vals, y_vals, -theta, x_offset, y_offset)

    return transformed

end

end # module
