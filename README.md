# Learning Lattice Planner

This repository contains the experimental code used in "Learning a Lattice Planner Control Set for Autonomous Vehicles".

Experiments 1 and 2 require a DataFromSky dataset, their website is [here](http://datafromsky.com/).
Experiment 3 generates synthetic data for the experiment, and can be run without any external dataset.

To run Experiment 3, run
`julia main.jl`.

This repository requires Julia 0.6.2, and the following packages:
StatsBase
CPUTime
JLD
PyCall
JuMP
Ipopt
Distances
PyPlot
DataStructures
Dierckx
Formatting
