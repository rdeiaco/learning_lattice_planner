push!(LOAD_PATH, pwd())

using ClusterExperiment, SpiralExperiment

function main()

  # Guided clustered search.
#  @printf("Running clustered search with non-rotated paths.\n")
#  spiralExperiment1(true, true, false)
  
#  @printf("Running clustered search with non-rotated paths, with lane change.\n")
#  spiralExperiment2(true, true, false)

  # Synthetic data guided clustered search.
  @printf("Running clustered search with generated paths.\n")
  spiralExperiment3(true, true, false)

end


main()
