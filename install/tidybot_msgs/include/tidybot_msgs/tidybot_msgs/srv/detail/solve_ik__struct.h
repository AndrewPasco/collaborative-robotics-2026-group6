// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tidybot_msgs:srv/SolveIK.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__SOLVE_IK__STRUCT_H_
#define TIDYBOT_MSGS__SRV__DETAIL__SOLVE_IK__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'arm_name'
#include "rosidl_runtime_c/string.h"
// Member 'target_pose'
#include "geometry_msgs/msg/detail/pose__struct.h"
// Member 'seed_joint_positions'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in srv/SolveIK in the package tidybot_msgs.
typedef struct tidybot_msgs__srv__SolveIK_Request
{
  /// Arm to solve IK for: "right" or "left"
  rosidl_runtime_c__String arm_name;
  /// Target end-effector pose in base_link frame
  geometry_msgs__msg__Pose target_pose;
  /// Optional: Initial joint positions as seed for solver (6 values)
  /// If empty, uses current joint positions
  rosidl_runtime_c__double__Sequence seed_joint_positions;
  /// Whether to consider orientation in the solution
  /// If false, only position is matched
  bool use_orientation;
  /// Maximum time allowed for solving in seconds
  double timeout;
} tidybot_msgs__srv__SolveIK_Request;

// Struct for a sequence of tidybot_msgs__srv__SolveIK_Request.
typedef struct tidybot_msgs__srv__SolveIK_Request__Sequence
{
  tidybot_msgs__srv__SolveIK_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tidybot_msgs__srv__SolveIK_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'joint_positions'
// already included above
// #include "rosidl_runtime_c/primitives_sequence.h"
// Member 'message'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/SolveIK in the package tidybot_msgs.
typedef struct tidybot_msgs__srv__SolveIK_Response
{
  /// Whether a valid IK solution was found
  bool success;
  /// Joint positions solution (6 values)
  /// Order: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
  rosidl_runtime_c__double__Sequence joint_positions;
  /// Residual position error in meters
  double position_error;
  /// Residual orientation error in radians
  double orientation_error;
  /// Status message (e.g., "Success", "No solution found", "Timeout")
  rosidl_runtime_c__String message;
} tidybot_msgs__srv__SolveIK_Response;

// Struct for a sequence of tidybot_msgs__srv__SolveIK_Response.
typedef struct tidybot_msgs__srv__SolveIK_Response__Sequence
{
  tidybot_msgs__srv__SolveIK_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tidybot_msgs__srv__SolveIK_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TIDYBOT_MSGS__SRV__DETAIL__SOLVE_IK__STRUCT_H_
