// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tidybot_msgs:srv/GetJointState.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__GET_JOINT_STATE__STRUCT_H_
#define TIDYBOT_MSGS__SRV__DETAIL__GET_JOINT_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'joint_names'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/GetJointState in the package tidybot_msgs.
typedef struct tidybot_msgs__srv__GetJointState_Request
{
  /// Optional: Specific joint names to query
  /// If empty, returns all joints
  rosidl_runtime_c__String__Sequence joint_names;
} tidybot_msgs__srv__GetJointState_Request;

// Struct for a sequence of tidybot_msgs__srv__GetJointState_Request.
typedef struct tidybot_msgs__srv__GetJointState_Request__Sequence
{
  tidybot_msgs__srv__GetJointState_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tidybot_msgs__srv__GetJointState_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'state'
#include "sensor_msgs/msg/detail/joint_state__struct.h"

/// Struct defined in srv/GetJointState in the package tidybot_msgs.
typedef struct tidybot_msgs__srv__GetJointState_Response
{
  /// Current joint state
  sensor_msgs__msg__JointState state;
} tidybot_msgs__srv__GetJointState_Response;

// Struct for a sequence of tidybot_msgs__srv__GetJointState_Response.
typedef struct tidybot_msgs__srv__GetJointState_Response__Sequence
{
  tidybot_msgs__srv__GetJointState_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tidybot_msgs__srv__GetJointState_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TIDYBOT_MSGS__SRV__DETAIL__GET_JOINT_STATE__STRUCT_H_
