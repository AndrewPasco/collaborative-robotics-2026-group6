// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tidybot_msgs:msg/ArmVelocityCommand.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__ARM_VELOCITY_COMMAND__STRUCT_H_
#define TIDYBOT_MSGS__MSG__DETAIL__ARM_VELOCITY_COMMAND__STRUCT_H_

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

/// Struct defined in msg/ArmVelocityCommand in the package tidybot_msgs.
/**
  * Joint velocity command for a 6-DOF WX250s arm
  * Joint order: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
 */
typedef struct tidybot_msgs__msg__ArmVelocityCommand
{
  std_msgs__msg__Header header;
  /// Target velocities in rad/s for each joint
  double joint_velocities[6];
  /// Duration to apply velocity in seconds (0 = until next command)
  double duration;
} tidybot_msgs__msg__ArmVelocityCommand;

// Struct for a sequence of tidybot_msgs__msg__ArmVelocityCommand.
typedef struct tidybot_msgs__msg__ArmVelocityCommand__Sequence
{
  tidybot_msgs__msg__ArmVelocityCommand * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tidybot_msgs__msg__ArmVelocityCommand__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TIDYBOT_MSGS__MSG__DETAIL__ARM_VELOCITY_COMMAND__STRUCT_H_
