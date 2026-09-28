# One object instance in the persistent semantic map.
#
# header.frame_id is the fixed frame the position is expressed in (normally
# "map"); header.stamp is the time the object was last observed.
std_msgs/Header header

# Class name from the detector, e.g. "refrigerator", "chair".
string label

# Stable instance id assigned by the semantic map (unique per map, not per label).
uint32 id

# Estimated object centroid in header.frame_id, metres.
geometry_msgs/Point position

# Aggregated detection confidence in [0, 1].
float32 confidence

# Number of detections that were associated with this object.
uint32 observation_count
