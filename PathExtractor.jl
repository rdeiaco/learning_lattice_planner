module PathExtractor

using Constants, Dierckx

export extractPaths, extractProcessedPath, extractRawPath

function extractPaths(filename, train_fraction=0.8)::Tuple{Array{Array{Float64, 2}}, Array{Array{Float64, 2}}}
  data = readdlm(filename, ';', header=true)
  data = data[1][:, 9:end]
 
  # Go through each row in the data array, and extract paths
  # from it.
  path_count::UInt64 = 1
  processed_path_array::Array{Array{Float64, 2}} = []
  raw_path_array::Array{Array{Float64, 2}} = []
  for i = 1:size(data, 1)
    x_vals, y_vals = extractTransformedVals(data[i, :])
    # Check if the x_vals will satisfy our interpolation requirements.
    try
      for i = 1:length(x_vals)-1
        @assert(x_vals[i+1] > x_vals[i])
      end
    catch
      @printf("Skipping path\n")
      continue
    end

    if rand() < train_fraction
      append!(processed_path_array, extractProcessedPaths(x_vals, y_vals))
    else
      push!(raw_path_array, extractRawPath(x_vals, y_vals))
    end

  end

  return (processed_path_array, raw_path_array)

end

function extractProcessedPaths(x_vals::Array{Float64}, y_vals::Array{Float64})::Array{Array{Float64, 2}}
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
    transformed = transformPath(path_x, path_y, -theta, x_offset, y_offset)
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
