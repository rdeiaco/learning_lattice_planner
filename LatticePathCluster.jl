module LatticePathCluster

export PathCluster, agglomeratePaths, kMeansPaths, dunnIndex, euclideanPathSum, maxPathDist

mutable struct PathCluster
  paths::Array{Array{Float64, 2}, 1}
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
      mean_path[j, 1] /= num_paths
    end

    new(paths, mean_path)
    
  end

  function PathCluster(paths::Array{Array{Float64, 2}}, mean_path::Array{Float64, 2})
    new(paths, mean_path) 
  end

  # Include flag denotes whether the original path should be included
  # as part of the cluster.
  function PathCluster(path::Array{Float64, 2}, include_flag::Bool=true)
    if include_flag
      return new([path], path)
    else
      return new(Array{Array{Float64, 2}, 1}(), path)
    end
  end

end

# Merge 2 different clusters. Make sure to preserve the ratio
# of the cluster averages depending on the size of each cluster.
function mergeClusters(cluster_1::PathCluster, cluster_2::PathCluster, weighted_agglom::Bool=false)::PathCluster
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
    # Weighted mean
    if weighted_agglom
      mean_path[i] = (cluster_1.mean_path[i] * num_paths_1 + cluster_2.mean_path[i] * num_paths_2) / (num_paths_1 + num_paths_2)
    # Unweighted mean
    else
      mean_path[i] = (cluster_1.mean_path[i] + cluster_2.mean_path[i]) / 2.0
    end
  end

  return PathCluster(paths, mean_path)
    
end

# Scores how similar two clusters are, based on the maximum distance between
# any two paths within each cluster.
function scoreClusters(cluster_1::PathCluster, cluster_2::PathCluster)::Float64
  @assert(size(cluster_1.mean_path, 1) == size(cluster_2.mean_path, 1))

  score::Float64 = 0.0
  for i = 1:size(cluster_1.paths, 1)
    for j = 1:size(cluster_2.paths, 1)
      temp_score::Float64 = 0.0
      for k = 1:size(cluster_1.paths[i], 1)
        temp_score += ((cluster_1.paths[i][k, 1] - cluster_2.paths[j][k, 1])^2 + (cluster_1.paths[i][k, 2] - cluster_2.paths[j][k, 2])^2)^0.5
      end
      if temp_score > score
        score = temp_score
      end
    end
  end

  return score

end

# Takes in an array of paths, and returns an array of clusters
# of the given size after performing an agglomerative clustering
# algorithm.
function agglomeratePaths(paths::Array{Array{Float64, 2}}, num_clusters::Int64, weighted_agglom::Bool=false)::Array{PathCluster}
  clusters::Array{PathCluster} = []
  for i = 1:size(paths, 1)
    push!(clusters, PathCluster(paths[i]))
  end 

  @printf("\n")
  while size(clusters, 1) > num_clusters
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

    new_cluster::PathCluster = mergeClusters(clusters[i], clusters[min_index], weighted_agglom)
    deleteat!(clusters, sort([i, min_index]))
    push!(clusters, new_cluster)

  end

  return clusters

end

# Computes the summed Euclidean distance between two paths.
function euclideanPathSum(path1::Array{Float64, 2}, path2::Array{Float64, 2})::Float64
  path_sum::Float64 = 0.0
  for i = 1:size(path1, 1)
    path_sum += ((path1[i, 1] - path2[i, 1])^2 + (path1[i, 2] - path2[i, 2])^2)^0.5
  end

  return path_sum

end

# Computes the max pointwise distance between two paths.
function maxPathDist(path1::Array{Float64, 2}, path2::Array{Float64, 2})::Float64
  max_dist::Float64 = 0.0
  for i = 1:size(path1, 1)
    max_dist = max(max_dist, ((path1[i, 1] - path2[i, 1])^2 + (path1[i, 2] - path2[i, 2])^2)^0.5)
  end

  return max_dist

end

# Takes in an array of paths, and returns an array of clusters of the given
# size after performing a k-means update using the sum of Euclidean distances
# along the path.
function kMeansPaths(paths::Array{Array{Float64, 2}}, num_clusters::Int64, random_init::Bool=false, use_max::Bool=false, ti::Float64=0.0)::Array{PathCluster}

  clusters::Array{PathCluster} = []
  path_length = size(paths[1], 1)
  # Initialize num_clusters clusters to be random walks in the domain.
  if random_init
    for i = 1:num_clusters
      path::Array{Float64, 2} = Array{Float64, 2}(path_length, 2)
      path[1, :] = [0.0, 0.0]
      for j = 2:path_length
        path[j, :] = [path[j-1, 1] + 0.1*cos(ti+-0.1+0.2*rand()), path[j-1, 2] + 0.1*sin(ti+-0.1+0.2*rand())]
      end
      push!(clusters, PathCluster(path, false))
    end

  # Initialize num_clusters clusters to be rays that equally divide the 
  # first quadrant.
  else
    for i = 0:num_clusters-1
      path::Array{Float64, 2} = Array{Float64, 2}(path_length, 2)
      for j = 1:path_length
        # Subtract one from j to start at (0.0, 0.0).
        x = ((j-1)*10.0)/path_length*cos((i*pi/2.0)/num_clusters)
        y = ((j-1)*10.0)/path_length*sin((i*pi/2.0)/num_clusters)
        path[j, 1] = x
        path[j, 2] = y
      end
      push!(clusters, PathCluster(path, false))
    end
  end

  # Next, assign each path to an initial cluster, according to the cluster
  # whose mean path is minimally distant according to the Euclidean metric.
  # Keep track of each path's cluster to help in the EM process later.
  path_cluster_indices::Array{Int64} = Array{Int64}(size(paths, 1))
  for i = 1:size(paths, 1)
    path::Array{Float64, 2} = paths[i]
    min_path_sum::Float64 = Inf
    min_cluster_index::Int64 = -1
    for j = 1:num_clusters
      if use_max
        path_sum = maxPathDist(path, clusters[j].mean_path)
      else
        path_sum = euclideanPathSum(path, clusters[j].mean_path)
      end

      if path_sum < min_path_sum
        min_path_sum = path_sum
        min_cluster_index = j
      end
    end  
    
    path_cluster_indices[i] = min_cluster_index

    push!(clusters[min_cluster_index].paths, paths[i])
      
  end

  # Next, run 10 iterations of EM updates for each cluster. 
  # Could also terminate if the clusters converge. 
  for i = 1:200
    # First, update the mean for each of the clusters.
    for j = 1:num_clusters
      if size(clusters[j].paths, 1) < 1
        continue
      end

      mean_path::Array{Float64, 2} = Array{Float64, 2}(size(clusters[j].paths[1], 1), 2)
      for k = 1:size(clusters[j].paths, 1)
        for l = 1:size(mean_path, 1)
          mean_path[l, 1] += clusters[j].paths[k][l, 1]  
          mean_path[l, 2] += clusters[j].paths[k][l, 2]  
        end
      end

      for l = 1:size(mean_path, 1)
        mean_path[l, 1] /= size(clusters[j].paths, 1) 
        mean_path[l, 2] /= size(clusters[j].paths, 1)  
      end

      clusters[j].mean_path = mean_path
      arc_length = 0.0
      for l = 1:size(mean_path, 1)-1
        arc_length += ((mean_path[l+1, 1] - mean_path[l, 1])^2 + (mean_path[l+1, 2] - mean_path[l, 2])^2)^0.5

      end
    end

    # Then, clear the paths of each cluster and re-assign the clusters according 
    # to the new mean path. 
    for i = 1:num_clusters
      clusters[i].paths = []
    end

    for i = 1:size(paths, 1)
      path::Array{Float64, 2} = paths[i]
      min_path_sum::Float64 = Inf
      min_cluster_index::Int64 = -1
      for j = 1:num_clusters
        if use_max
          path_sum = maxPathDist(path, clusters[j].mean_path)
        else
          path_sum = euclideanPathSum(path, clusters[j].mean_path)
        end

        if path_sum < min_path_sum
          min_path_sum = path_sum
          min_cluster_index = j
        end
      end  
      
      path_cluster_indices[i] = min_cluster_index
      push!(clusters[min_cluster_index].paths, paths[i])
        
    end

  end

  return clusters

end

# Computes the Dunn index for a set of clusters.
# The Dunn index is the ratio of the minimum intercluster
# distance to the maximum intracluster distance.
# A larger Dunn index corresponds to a more desirable clustering result.
function dunnIndex(clusters::Array{PathCluster}, use_max::Bool=false)
  # Compute the minimum intercluster distance.
  min_intercluster_distance::Float64 = Inf
  for i = 1:size(clusters, 1)-1
    for j = i+1:size(clusters, 1)
      temp_min_intercluster_distance::Float64 = Inf
      for k = 1:size(clusters[i].paths, 1)
        for l = 1:size(clusters[j].paths, 1)
          if use_max
            path_sum = maxPathDist(clusters[i].paths[k], clusters[j].paths[l])
          else
            path_sum = euclideanPathSum(clusters[i].paths[k], clusters[j].paths[l])
          end

          if path_sum < temp_min_intercluster_distance
            temp_min_intercluster_distance = path_sum
          end
        end
      end

      if temp_min_intercluster_distance < min_intercluster_distance
        min_intercluster_distance = temp_min_intercluster_distance
      end
    end
  end
  @printf("min_intercluster_distance = %f\n", min_intercluster_distance)

  # Compute the maximum intracluster distance.
  max_intracluster_distance::Float64 = 0.0
  for i = 1:size(clusters, 1)
    # Distance measurement is symmetrical for the Euclidean
    # path metric.
    temp_max_intracluster_distance::Float64 = 0.0
    for j = 1:size(clusters[i].paths, 1)-1
      for k = j+1:size(clusters[i].paths, 1)
        if use_max
          path_sum = maxPathDist(clusters[i].paths[j], clusters[i].paths[k])
        else
          path_sum = euclideanPathSum(clusters[i].paths[j], clusters[i].paths[k])
        end

        if path_sum > temp_max_intracluster_distance
          temp_max_intracluster_distance = path_sum
        end
      end
    end
    
    if temp_max_intracluster_distance > max_intracluster_distance
      max_intracluster_distance = temp_max_intracluster_distance
    end

  end
  @printf("max_intracluster_distance = %f\n", max_intracluster_distance)

  return min_intercluster_distance / max_intracluster_distance

end

end # module
