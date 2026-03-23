# CREBAIN Thermal Detection Message
# Single thermal signature detection

std_msgs/Header header

# Unique detection ID
string id

# Position in sensor frame
geometry_msgs/Point position

# Temperature in Kelvin
float64 temperature_kelvin

# Estimated signature area in m^2
float64 signature_area

# Detection confidence [0.0 - 1.0]
float64 confidence

# Classification label
string classification
