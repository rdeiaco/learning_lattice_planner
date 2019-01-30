module SwathGenerator

using Constants, LatticeState, PolynomialSpiral, LatticeOccupancyGrid

export generateSwaths, transformSwath, generateSwathFromPath

# Generates the swaths of a car following each control action.
function generateSwaths(control_action_lut::Array{Dict{UInt128, SpiralControlAction}})::Dict{UInt128, Set{Tuple{UInt64, UInt64}}}
  swath_lut = Dict{UInt128, Set{Tuple{UInt64, UInt64}}}()

  for i = 1:size(control_action_lut, 1)
    for (control_id, control_action) in control_action_lut[i]
      swath::Set{Tuple{UInt64, UInt64}} = Set{Tuple{UInt64, UInt64}}()

      for j = 1:size(control_action.path, 1)
        point::Array{Float64} = control_action.path[j, :]

        x_offset::Float64 = 0.0
        while x_offset < CAR_LENGTH
          y_offset::Float64 = -CAR_WIDTH / 2.0
          while y_offset < CAR_WIDTH / 2.0
            x_point::Float64 = point[1] + cos(point[3]) * x_offset - sin(point[3]) * y_offset
            y_point::Float64 = point[2] + sin(point[3]) * x_offset + cos(point[3]) * y_offset

            local indices::Tuple{UInt64, UInt64}
            try
              indices = getGridIndices([x_point, y_point])
            catch
              @printf("Violation x = %f, y = %f\n", x_point, y_point)
              show(control_action.xf)
              @printf("\n")
              show(control_action.yf)
              @printf("\n")
              show(control_action.ti)
              @printf("\n")
              show(control_action.tf)
              @printf("\n")
              show(control_action.kmax)
              @printf("\n")
              exit()
            end
            
            push!(swath, indices)
            
            y_offset += GRID_RESOLUTION
          end
          x_offset += GRID_RESOLUTION
        end
      end

      swath_lut[control_id] = swath

    end
  end

  return swath_lut

end

# Transforms the indices of a swath based on the input 
# starting state and the relevant control action.
function transformSwath(swath::Set{Tuple{UInt64, UInt64}}, state::State, 
  control_action::SpiralControlAction)::Set{Tuple{UInt64, UInt64}}
  delta_theta::Float64 = (state.theta - control_action.ti) % (2.0 * pi)
  multiple::Float64 = delta_theta / (pi / 2.0)
  @assert(abs(multiple - round(multiple)) < 0.01)

  transformed_swath = Set{Tuple{UInt64, UInt64}}()
 
  for indices in swath
    point::Array{Float64} = getGridPoint(indices) 
    xf::Float64 = state.x + point[1] * cos(delta_theta) - point[2] * sin(delta_theta)
    yf::Float64 = state.y + point[1] * sin(delta_theta) + point[2] * cos(delta_theta)
    push!(transformed_swath, getGridIndices([xf, yf]))
  end
  
  return transformed_swath

end

# Creates a swath from a raw path, not necessarily corresponding
# to a grid-aligned control action.
function generateSwathFromPath(path::Array{Float64, 2})::Set{Tuple{UInt64, UInt64}}
  swath = Set{Tuple{UInt64, UInt64}}()

  for i = 1:size(path, 1)
    point::Array{Float64} = path[i, :]

    x_offset::Float64 = 0.0
    while x_offset < CAR_LENGTH
      y_offset::Float64 = -CAR_WIDTH / 2.0
      while y_offset < CAR_WIDTH / 2.0
        x_point::Float64 = point[1] + cos(point[3]) * x_offset - sin(point[3]) * y_offset
        y_point::Float64 = point[2] + sin(point[3]) * x_offset + cos(point[3]) * y_offset

        #local indices::Tuple{UInt64, UInt64}
        #try
        #  indices = getGridIndices([x_point, y_point])
        #catch
        #  @printf("Indices went out of bounds.\n")
        #  y_offset += GRID_RESOLUTION
        #  continue
        #end

        indices::Tuple{UInt64, UInt64} = getGridIndices([x_point, y_point])
        
        push!(swath, indices)
        
        y_offset += GRID_RESOLUTION

      end
      x_offset += GRID_RESOLUTION
    end
  end

  return swath

end

end # module
