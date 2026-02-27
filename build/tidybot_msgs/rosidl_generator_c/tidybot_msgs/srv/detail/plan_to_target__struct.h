// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tidybot_msgs:srv/PlanToTarget.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__STRUCT_H_
#define TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__STRUCT_H_

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

/// Struct defined in srv/PlanToTarget in the package tidybot_msgs.
typedef struct tidybot_msgs__srv__PlanToTarget_Request
{
  /// Which arm to move: "right" or "left"
  rosidl_runtime_c__String arm_name;
  /// Target end-effector pose in base_link frame
  geometry_msgs__msg__Pose target_pose;
  /// Whether to match orientation or position only
  bool use_orientation;
  /// Whether to execute the motion if planning succeeds
  bool execute;
  /// Motion duration in seconds (for execution)
  double duration;
  /// Maximum allowed condition number for Jacobian (singularity threshold)
  /// Lower = more conservative. Default 100.0 is reasonable.
  double max_condition_number;
} tidybot_msgs__srv__PlanToTarget_Request;

// Struct for a sequence of tidybot_msgs__srv__PlanToTarget_Request.
typedef struct tidybot_msgs__srv__PlanToTarget_Request__Sequence
{
  tidybot_msgs__srv__PlanToTarget_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tidybot_msgs__srv__PlanToTarget_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'joint_positions'
#include "rosidl_runtime_c/primitives_sequence.h"
// Member 'message'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/PlanToTarget in the package tidybot_msgs.
typedef struct tidybot_msgs__srv__PlanToTarget_Response
{
  /// Whether planning succeeded (collision-free, no singularity)
  bool success;
  /// Joint positions solution (6 values if success)
  /// Order: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
  rosidl_runtime_c__double__Sequence joint_positions;
  /// Whether execution was attempted (only if success=true and execute=true)
  bool executed;
  /// Position error of IK solution (meters)
  double position_error;
  /// Orientation error of IK solution (radians)
  double orientation_error;
  /// Jacobian condition number at solution (lower = further from singularity)
  double condition_number;
  /// Detailed status message
  rosidl_runtime_c__String message;
} tidybot_msgs__srv__PlanToTarget_Response;

// Struct for a sequence of tidybot_msgs__srv__PlanToTarget_Response.
typedef struct tidybot_msgs__srv__PlanToTarget_Response__Sequence
{
  tidybot_msgs__srv__PlanToTarget_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tidybot_msgs__srv__PlanToTarget_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__STRUCT_H_
