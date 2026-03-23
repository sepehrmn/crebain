# CREBAIN Radar Detection Message
# Single radar return/track

std_msgs/Header header

# Unique detection ID
string id

# Detection in spherical coordinates (sensor frame)
float64 range          # meters
float64 azimuth        # radians
float64 elevation      # radians

# Radial velocity (positive = approaching)
float64 radial_velocity  # m/s

# Radar cross section in dBsm
float64 rcs_dbsm

# Detection confidence [0.0 - 1.0]
float64 confidence

# Classification label
string classification
