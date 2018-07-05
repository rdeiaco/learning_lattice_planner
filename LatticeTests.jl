push!(LOAD_PATH, pwd())

using Constants, LatticeState, PolynomialSpiral, Geometry

function getLatticeDistanceTest()
 
  control_action::SpiralControlAction = SpiralControlAction(0.41, 0.0, 0.0, 0.0, 0.0, 0.0)
  u = State(0.0, 0.0, 0.0, 0.0, 0)
  v = transformControlActionEnd(control_action, u.theta, u.x, u.y, u.steps)
  path = [[0.0, 0.0] [PATH_RESOLUTION/sqrt(2.0), PATH_RESOLUTION/sqrt(2.0)] [PATH_RESOLUTION + PATH_RESOLUTION/sqrt(2.0), PATH_RESOLUTION/sqrt(2.0)]]
  path = permutedims(path, [2 1])
  distance = getLatticeDistance(path, u, v, control_action)  
  show(distance)

end

function getDeltaStateTest()
  origin = State(4.0, 2.0, TI_RANGE[3] + 3 * pi / 2.0, 0.0, 0)
  goal = State(6.4, -6.0, TI_RANGE[4] + pi / 2.0, 0.0, 0)

  show(origin)
  @printf("\n")
  show(goal)
  @printf("\n")
  show(getDeltaState(origin, goal))
  @printf("\n")

end

getDeltaStateTest()
