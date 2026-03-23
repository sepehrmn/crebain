# CREBAIN Interception Command Message
# Command to initiate or update an interception

std_msgs/Header header

# Target drone ID
string target_id

# Interceptor drone ID
string interceptor_id

# Calculated intercept point
geometry_msgs/Point intercept_point

# Estimated time to intercept
duration time_to_intercept

# Interception strategy
uint8 STRATEGY_PURSUIT = 0
uint8 STRATEGY_LEAD = 1
uint8 STRATEGY_PARALLEL = 2
uint8 STRATEGY_AMBUSH = 3
uint8 strategy
