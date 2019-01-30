module ControlActionGenerator

using Constants, LatticeState, PolynomialSpiral

export generateControlActions

# Generates the lookup table for the control actions
# of the lattice.
function generateControlActions(kmax=0.5)::Array{Dict{UInt128, SpiralControlAction}}
  @printf("Control Action LUT not present. Rebuilding...\n")

  # Add a lookup table for each index in the initial angle array.
  control_action_lut::Array{Dict{UInt128, SpiralControlAction}} = []
  num_control_actions = size(TI_RANGE, 1) * 9 * 11 * size(TF_RANGE, 1) / 2
  count = 0
  feasible_count = 0

  @printf("Number of control actions: %i\n", num_control_actions)
  @printf("0%% of control actions complete...")

  # Populate each lookup table for each initial heading.
  for i = 1:size(TI_RANGE, 1)
    push!(control_action_lut, Dict{UInt128, SpiralControlAction}())
    sub_count = 0

    ti = TI_RANGE[i]
    for j = 1:10
      for k = -5:6
        if k != 0
          ratio::Float64 = convert(Float64, j) / convert(Float64, k)
          # If the endpoint lies outside of a cone, do not
          # create a control action for that point as it will be poor.
          if abs(ratio) < 2.0
              continue
          end
        end

        xf = RESOLUTION*(round(cos(ti)*j - sin(ti)*k))
        yf = RESOLUTION*(round(sin(ti)*j + cos(ti)*k))
        
        for tf in TF_RANGE
          # Assume zero initial and final curvature to simplify the problem.
          action = SpiralControlAction(xf, yf, ti, tf, 0.0, 0.0, kmax)

          # Due to discretization rounding, it may be possible for there
          # to be duplicates.
          key = getControlId(action)
          if !haskey(control_action_lut[i], key)
            count += 1
            if action.feasible
              control_action_lut[i][key] = action
              feasible_count += 1
              sub_count += 1
            end
          end

          @printf("\r%4.2f%% of control actions complete...  ", 
            (100.0 * convert(Float64, count)) / convert(Float64, num_control_actions))
        end
      end
    end

    sub_count = 0

  end

  # Next, ensure that each heading has at least one "basic"
  # path available to it, essentially a straight line to the
  # nearest node.
  for i = 1:size(X_BASE_RANGE, 1)
    xf = X_BASE_RANGE[i]
    yf = Y_BASE_RANGE[i]
    ti = TI_RANGE[i]
    tf = TI_RANGE[i]
    action = SpiralControlAction(xf, yf, ti, tf, 0.0, 0.0, kmax)

    if !action.feasible
      @printf("Basic Control (%f, %f, %f, %f) is not feasible.\n", xf, yf, ti, tf)
      exit()
    end

    key = getControlId(action)
    try
      control_action_lut[i][key]
    catch
      control_action_lut[i][key] = action
      @printf("Basic Control (%f, %f, %f, %f) was not present.\n", xf, yf, ti, tf)
      feasible_count += 1
    end
  end

  # Line separator after all LUTs have been populated.
  @printf("\n")

  @printf("%d Feasible control actions found out of %d.\n", feasible_count, count)

  return control_action_lut

end

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
            indices::Tuple{UInt64, UInt64} = getGridIndices([x_point, y_point])
            
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

end # module
