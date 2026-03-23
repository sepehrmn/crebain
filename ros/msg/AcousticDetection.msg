# CREBAIN Acoustic Detection Message
# Single acoustic signature detection from microphone array

std_msgs/Header header

# Unique detection ID
string id

# Direction of arrival (spherical coordinates)
float64 azimuth        # radians from sensor forward
float64 elevation      # radians from horizontal

# Estimated range in meters (may be uncertain)
float64 range_estimate

# Sound pressure level in dB
float64 spl_db

# Dominant frequency in Hz
float64 dominant_frequency_hz

# Doppler shift for velocity estimation
float64 doppler_hz

# Detection confidence [0.0 - 1.0]
float64 confidence

# Classification label (e.g., "quadcopter", "fixed_wing", "unknown")
string classification
