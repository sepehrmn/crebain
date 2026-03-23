# CREBAIN Interception Status Message
# Status update for an active interception mission

std_msgs/Header header

# Associated command
crebain_msgs/InterceptionCommand command

# Mission status
uint8 STATUS_PENDING = 0
uint8 STATUS_ACTIVE = 1
uint8 STATUS_COMPLETED = 2
uint8 STATUS_ABORTED = 3
uint8 STATUS_FAILED = 4
uint8 status

# Current distance to target in meters
float64 distance_to_target

# Estimated time remaining
duration estimated_time_remaining
