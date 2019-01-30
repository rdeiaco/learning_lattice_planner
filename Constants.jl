module Constants

export XF_RANGE, YF_RANGE, TI_RANGE, TF_RANGE, X_BASE_RANGE, Y_BASE_RANGE, KI_RANGE, KF_RANGE,
  ANGLE_RANGE, RESOLUTION, ANGLE_RESOLUTION, CURVATURE_RESOLUTION, PATH_RESOLUTION, 
  GRID_LENGTH, GRID_WIDTH, PATH_LENGTH_M, GRID_RESOLUTION, LANE_WIDTH, LANE_LENGTH,
  CAR_LENGTH, CAR_WIDTH, GRID_X_START, GRID_X_END, GRID_Y_START, GRID_Y_END,
  ARC_LENGTH_RESOLUTION, X_LENGTH, Y_LENGTH, PATH_LENGTH, RED, GREEN, BLUE, WHITE, 
  BLACK, SCREEN_WIDTH, SCREEN_HEIGHT, PIXELS_PER_GRID_POINT, GOAL_ARRAY_SIZE, 
  GOAL_ARRAY_CENTER_INDEX, GOAL_ARRAY_STEP, LOOKAHEAD, CAR_STEP, LANE_CHANGE_OBSTACLE_INDEX_GAP,
  NUM_CLUSTERS, MAX_NUM_PATHS, STEP_SIZE, STEP_RADIUS

# Planning Constants
const RESOLUTION = 0.4
const ANGLE_RESOLUTION = 10.0*pi/180.0
const CURVATURE_RESOLUTION = 0.1
const PATH_RESOLUTION = 0.1
const ARC_LENGTH_RESOLUTION = 0.001
#const X_LENGTH = 120.0
#const Y_LENGTH = 120.0
const X_LENGTH = 80.0
const Y_LENGTH = 80.0
const PATH_LENGTH = 101
const PATH_LENGTH_M = 10.0

# Range values that define the lattice "grid".
const XF_RANGE = [4.0, 8.0]
const YF_RANGE = [-1.2, 0.0, 1.2]
const TI_RANGE = [0.0, atan(1.0/3.0), atan(1.0/2.0), atan(1.0/1.0), 
  atan(2.0/1.0), atan(3.0/1.0)]
const TF_RANGE = [3*pi/2.0, 2*pi-atan(3.0/1.0), 2*pi-atan(2.0/1.0), 
  2*pi-atan(1.0/1.0), 2*pi-atan(1.0/2.0), 2*pi-atan(1.0/3.0), 
  0.0, atan(1.0/3.0), atan(1.0/2.0), atan(1.0/1.0), atan(2.0/1.0), 
  atan(3.0/1.0), pi/2.0]
const X_BASE_RANGE = [1.0*RESOLUTION, 3.0*RESOLUTION, 2.0*RESOLUTION, 1.0*RESOLUTION, 1.0*RESOLUTION, 1.0*RESOLUTION]
const Y_BASE_RANGE = [0.0*RESOLUTION, 1.0*RESOLUTION, 1.0*RESOLUTION, 1.0*RESOLUTION, 2.0*RESOLUTION, 3.0*RESOLUTION]
const KI_RANGE = [0.0]
const KF_RANGE = [0.0]

const ANGLE_RANGE = [0.0, atan(1.0/3.0), atan(1.0/2.0), atan(1.0), 
  atan(2.0), atan(3.0), pi/2.0, pi/2.0 + atan(1.0/3.0), 
  pi/2.0 + atan(1.0/2.0), pi/2.0 + atan(1.0), 
  pi/2.0 + atan(2.0/1.0), pi/2.0 + atan(3.0/1.0), pi, 
  pi + atan(1.0/3.0), pi + atan(1.0/2.0), pi + atan(1.0), 
  pi + atan(2.0), pi + atan(3.0), 1.5*pi, 
  1.5*pi + atan(1.0/3.0), 1.5*pi + atan(1.0/2.0), 
  1.5*pi + atan(1.0), 1.5*pi + atan(2.0), 1.5*pi + atan(3.0)]

@assert(X_LENGTH / RESOLUTION < 1024)
@assert(2*Y_LENGTH / RESOLUTION < 1024)
@assert(2*pi / ANGLE_RESOLUTION < 1024)
@assert(2*0.5 / CURVATURE_RESOLUTION < 1024)

# Note that GRID_WIDTH is along y, GRID_LENGTH is along x.
const GRID_WIDTH = Y_LENGTH
const GRID_LENGTH = X_LENGTH
const GRID_X_START = -10.0
const GRID_X_END = GRID_X_START + GRID_LENGTH
const GRID_Y_START = -GRID_WIDTH / 2.0
const GRID_Y_END = GRID_Y_START + GRID_WIDTH
const GRID_RESOLUTION = PATH_RESOLUTION
const LANE_WIDTH = 3.7
const LANE_LENGTH = 70.0
const CAR_LENGTH = 4.5
const CAR_WIDTH = 1.7
const CAR_X_START = -1.0
const CAR_Y_START = -CAR_WIDTH / 2.0
const LANE_CHANGE_OBSTACLE_INDEX_GAP = 100

# Optimization Planner Constants
const GOAL_ARRAY_SIZE = 11
const GOAL_ARRAY_CENTER_INDEX = 6
const GOAL_ARRAY_STEP = 0.5
const LOOKAHEAD = 10.0
const CAR_STEP = 1.0

# Visualization Constants
const RED = (255, 0, 0)
const GREEN = (0, 255, 0)
const BLUE = (0, 0, 255)
const WHITE = (255, 255, 255)
const BLACK = (0, 0, 0)

# Menger Curvature Constants
const STEP_SIZE = 21
const STEP_RADIUS = 10

# Clustering Constants
const NUM_CLUSTERS = 12
const MAX_NUM_PATHS = 192

end # module
