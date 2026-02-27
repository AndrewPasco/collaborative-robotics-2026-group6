// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tidybot_msgs:msg/PanTilt.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__PAN_TILT__STRUCT_H_
#define TIDYBOT_MSGS__MSG__DETAIL__PAN_TILT__STRUCT_H_

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

/// Struct defined in msg/PanTilt in the package tidybot_msgs.
/**
  * Pan-tilt camera gimbal command
  * Controls the RealSense D435 camera orientation
 */
typedef struct tidybot_msgs__msg__PanTilt
{
  std_msgs__msg__Header header;
  /// Pan angle in radians
  /// Range: [-1.57, 1.57] (approximately +/- 90 degrees)
  double pan;
  /// Tilt angle in radians
  /// Range: [-1.57, 1.31] (approximately -90 to +75 degrees)
  double tilt;
} tidybot_msgs__msg__PanTilt;

// Struct for a sequence of tidybot_msgs__msg__PanTilt.
typedef struct tidybot_msgs__msg__PanTilt__Sequence
{
  tidybot_msgs__msg__PanTilt * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tidybot_msgs__msg__PanTilt__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TIDYBOT_MSGS__MSG__DETAIL__PAN_TILT__STRUCT_H_
