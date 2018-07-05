module PathReader

using Constants

export readPaths

function readPaths(file::String)::Array{Float64, 3}
  data::Array{Float64, 2} = readdlm(file, '\t', Float64)

  paths::Array{Float64} = reshape(permutedims(data[:, 2:3], [2, 1]), :) 

  paths_final = permutedims(reshape(paths, 2, PATH_LENGTH, :), [3, 2, 1])

  return paths_final
end

end # module
