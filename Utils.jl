module Utils

using Constants

export wrapTo2Pi, getClosestIndex

# Bounds an angle to the range [0, 2*pi). 
function wrapTo2Pi(angle::Float64)::Float64
  angle = angle % (2*pi) 
  if angle < 0.0
    angle += (2*pi)
  end

  if abs(2*pi - angle) < 0.01
    angle = 0.0
  end

  return angle

end

# Finds the index in the given range of values that is closest to the
# given input.
function getClosestIndex(input_val::Float64, val_list::Array{Float64})::Int64
  diff::Float64 = Inf
  index = 0

  for i in 1:size(val_list, 1)
    temp_diff = abs(input_val - val_list[i])
    if temp_diff < diff
      diff = temp_diff
      index = i
    end
  end

  return index

end

end # module
