module ClusterExperiment

using LatticePathCluster, LatticeVisualizer, PathExtractor, JLD

export clusterExperiment

function clusterExperiment()

  @printf("Extracting raw paths...\n")
  training_set::Array{Array{Array{Float64, 2}}} = []
  test_set::Array{Array{Float64, 2}} = []
  test_set_processed::Array{Array{Float64, 2}} = []
  try
    training_set = load("training_set.jld", "training_set")
    test_set = load("test_set.jld", "test_set")
    test_set_processed = load("test_set_processed.jld", "test_set_processed")
  catch
    training_set, test_set, test_set_processed = extractPaths("roundabout_data.csv", 0.85)
    save("training_set.jld", "training_set", training_set)
    save("test_set.jld", "test_set", test_set)
    save("test_set_processed.jld", "test_set_processed", test_set_processed)
  end

  @printf("Computing k-means rand-init clusters...\n")
  k_means_rand_init_clusters::Array{PathCluster} = kMeansPaths(training_set[1], 12, true, false) 
  plotClusters(k_means_rand_init_clusters, "K-means Rand Init Clusters")

  for i = 1:size(k_means_rand_init_clusters, 1)
    @printf("i = %i\n", i)
    cluster = k_means_rand_init_clusters[i]
    for j = 1:size(cluster.paths, 1)
      path = cluster.paths[j]
      path_dist = euclideanPathSum(path, cluster.mean_path)
      for k = 1:size(k_means_rand_init_clusters, 1)
        if i == k
          continue
        end

        temp_path_dist = euclideanPathSum(path, k_means_rand_init_clusters[k].mean_path)
        if temp_path_dist < path_dist
          @printf("Error: path found in incorrect cluster.\n")
          exit()
        end

      end
    end
  end
    
  @printf("All paths are in the correct cluster.\n") 

  @printf("Computing k-means rand-init clusters with max...\n")
  k_means_rand_init_clusters = kMeansPaths(training_set[1], 12, true, true) 
  plotClusters(k_means_rand_init_clusters, "K-means Rand Init Clusters")

  for i = 1:size(k_means_rand_init_clusters, 1)
    @printf("i = %i\n", i)
    cluster = k_means_rand_init_clusters[i]
    for j = 1:size(cluster.paths, 1)
      path = cluster.paths[j]
      path_dist = maxPathDist(path, cluster.mean_path)
      for k = 1:size(k_means_rand_init_clusters, 1)
        if i == k
          continue
        end

        temp_path_dist = maxPathDist(path, k_means_rand_init_clusters[k].mean_path)
        if temp_path_dist < path_dist
          @printf("Error: path found in incorrect cluster.\n")
          exit()
        end

      end
    end
  end
    
  @printf("All paths are in the correct cluster.\n") 
  exit()


  plotClusters(k_means_rand_init_clusters, "K-means Rand Init Clusters")
  @printf("Computing k-means rand-init clusters Dunn Index...\n")
  k_means_rand_init_dunn_index::Float64 = dunnIndex(k_means_rand_init_clusters)
  @printf("K means rand init Dunn Index = %f\n", k_means_rand_init_dunn_index)

  @printf("Computing k-means clusters...\n")
  k_means_clusters::Array{PathCluster} = kMeansPaths(paths, 12, false) 
  plotClusters(k_means_clusters, "K-means Ray Init Clusters")
  @printf("Computing k-means clusters Dunn Index...\n")
  k_means_dunn_index::Float64 = dunnIndex(k_means_clusters)
  @printf("K means Dunn Index = %f\n", k_means_dunn_index)

  @printf("Computing hierarchical unweighted agglomerative clusters...\n")
  hierarchical_unweighted_clusters::Array{PathCluster} = agglomeratePaths(paths, 12, false)
  plotClusters(hierarchical_unweighted_clusters, "Hierarchical Unweighted Agglomerative Clusters")
  @printf("Computing hierarchical unweighted agglomerative clusters Dunn Index...\n")
  hier_unw_agg_dunn_index::Float64 = dunnIndex(hierarchical_unweighted_clusters)
  @printf("Hierarchical agglomerative clusters Dunn Index = %f\n", hier_unw_agg_dunn_index)

  @printf("Computing hierarchical weighted agglomerative clusters...\n")
  hierarchical_weighted_clusters::Array{PathCluster} = agglomeratePaths(paths, 12, true)
  plotClusters(hierarchical_weighted_clusters, "Hierarchical Weighted Agglomerative Clusters")
  @printf("Computing hierarchical weighted agglomerative clusters Dunn Index...\n")
  hier_w_agg_dunn_index::Float64 = dunnIndex(hierarchical_weighted_clusters)
  @printf("Hierarchical weighted agglomerative clusters Dunn Index = %f\n", hier_w_agg_dunn_index)

end

end # module
