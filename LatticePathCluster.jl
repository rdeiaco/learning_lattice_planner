module LatticePathCluster

export agglomeratePaths

struct PathCluster
  paths::Array{Array{Float64, 2}}
  mean_path::Array{Float64, 2}

  function PathCluster(paths::Array{Array{Float64, 2}})
    num_paths::UInt64 = size(paths, 1)
    path_length::UInt64 = size(paths[1], 1)
    mean_path::Array{Float64, 2} = zeros(path_length, 2)


    for i = 1:num_paths
      path::Array{Float64} = paths[i]

      for j = 1:path_length
        mean_path[j, 1] += path[j, 1]
        mean_path[j, 2] += path[j, 2] 
      end

    end

    for j = 1:path_length
      mean_path[j, 1] /= path_length
    end

    new(paths, mean_path)
    
  end

  function PathCluster(paths::Array{Array{Float64, 2}}, mean_path::Array{Float64, 2})
    new(paths, mean_path) 
  end

  function PathCluster(path::Array{Float64, 2})
    new([path], path)
  end

end

# Merge 2 different clusters. Make sure to preserve the ratio
# of the cluster averages depending on the size of each cluster.
function mergeClusters(cluster_1::PathCluster, cluster_2::PathCluster)::PathCluster
  num_paths_1::UInt64 = size(cluster_1.paths, 1)
  num_paths_2::UInt64 = size(cluster_2.paths, 1)
 
  # The new set of paths contains both sets of paths
  # from the 2 clusters to merge.
  paths::Array{Array{Float64, 2}} = []
  append!(paths, cluster_1.paths)
  append!(paths, cluster_2.paths)

  # The new mean path is the weighted mean of both cluster's
  # mean path.
  path_length = size(cluster_1.mean_path, 1)
  mean_path::Array{Float64, 2} = zeros(path_length, 2)
  for i = 1:path_length
#    mean_path[i] = (cluster_1.mean_path[i] * num_paths_1 + cluster_2.mean_path[i] * num_paths_2) / (num_paths_1 + num_paths_2)
    mean_path[i] = (cluster_1.mean_path[i] + cluster_2.mean_path[i]) / 2.0
  end

  return PathCluster(paths, mean_path)
    
end

# Scores how similar two clusters are, based on their mean paths.
function scoreClusters(cluster_1::PathCluster, cluster_2::PathCluster)::Float64
  @assert(size(cluster_1.mean_path, 1) == size(cluster_2.mean_path, 1))

  score::Float64 = 0.0
  for i = 1:size(cluster_1.mean_path, 1)
    score += (cluster_1.mean_path[i, 1] - cluster_2.mean_path[i, 1])^2
    score += (cluster_1.mean_path[i, 2] - cluster_2.mean_path[i, 2])^2
  end

  return score

end

# Takes in an array of paths, and returns an array of clusters
# of the given size after performing an agglomerative clustering
# algorithm.
function agglomeratePaths(paths::Array{Array{Float64, 2}}, cluster_size::Int64)::Array{Array{Float64, 2}}
  clusters::Array{PathCluster} = []
  for i = 1:size(paths, 1)
    push!(clusters, PathCluster(paths[i]))
  end 

  @printf("\n")
  while size(clusters, 1) > cluster_size
    @printf("\rCluster size = %d     ", size(clusters, 1))
    i::UInt64 = rand(1:size(clusters, 1))
    min_score::Float64 = Inf
    min_index::UInt64 = i
    for j = 1:size(clusters, 1)
      if j == i
        continue
      end
      
      temp_score::Float64 = scoreClusters(clusters[i], clusters[j])
      if temp_score < min_score
        min_score = temp_score
        min_index = j
      end
    end

    new_cluster::PathCluster = mergeClusters(clusters[i], clusters[min_index])
    deleteat!(clusters, sort([i, min_index]))
    push!(clusters, new_cluster)

  end

  clustered_paths::Array{Array{Float64, 2}} = []
  for i = 1:size(clusters, 1)
    push!(clustered_paths, clusters[i].mean_path)
  end

  return clustered_paths

end

end # module
