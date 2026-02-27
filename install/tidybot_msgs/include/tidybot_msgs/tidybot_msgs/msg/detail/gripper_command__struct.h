// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tidybot_msgs:msg/GripperCommand.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__GRIPPER_COMMAND__STRUCT_H_
#define TIDYBOT_MSGS__MSG__DETAIL__GRIPPER_COMMAND__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"

/// Struct defined in msg/GripperCommand in the package tidybot_msgs.
/**
  * Gripper command for Robotiq 2F-85 gripper
  * Normalized position control
 */
typedef struct tidybot_msgs__msg__GripperCommand
{
  std_msgs__msg__Header header;
  /// Gripper position: 0.0 = fully open, 1.0 = fully closed
  double position;
  /// Optional effort limit: 0.0 = minimum force, 1.0 = maximum force
  /// Default to 0.5 if not specified
  double effort;
} tidybot_msgs__msg__GripperCommand;

// Struct for a sequence of tidybot_msgs__msg__GripperCommand.
typedef struct tidybot_msgs__msg__GripperCommand__Sequence
{
  tidybot_msgs__msg__GripperCommand * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tidybot_msgs__msg__GripperCommand__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TIDYBOT_MSGS__MSG__DETAIL__GRIPPER_COMMAND__STRUCT_H_
