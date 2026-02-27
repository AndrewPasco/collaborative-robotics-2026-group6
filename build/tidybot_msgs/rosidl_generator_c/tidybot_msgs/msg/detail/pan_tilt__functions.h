// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from tidybot_msgs:msg/PanTilt.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__PAN_TILT__FUNCTIONS_H_
#define TIDYBOT_MSGS__MSG__DETAIL__PAN_TILT__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "tidybot_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "tidybot_msgs/msg/detail/pan_tilt__struct.h"

/// Initialize msg/PanTilt message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * tidybot_msgs__msg__PanTilt
 * )) before or use
 * tidybot_msgs__msg__PanTilt__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
bool
tidybot_msgs__msg__PanTilt__init(tidybot_msgs__msg__PanTilt * msg);

/// Finalize msg/PanTilt message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
void
tidybot_msgs__msg__PanTilt__fini(tidybot_msgs__msg__PanTilt * msg);

/// Create msg/PanTilt message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * tidybot_msgs__msg__PanTilt__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
tidybot_msgs__msg__PanTilt *
tidybot_msgs__msg__PanTilt__create();

/// Destroy msg/PanTilt message.
/**
 * It calls
 * tidybot_msgs__msg__PanTilt__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
void
tidybot_msgs__msg__PanTilt__destroy(tidybot_msgs__msg__PanTilt * msg);

/// Check for msg/PanTilt message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
bool
tidybot_msgs__msg__PanTilt__are_equal(const tidybot_msgs__msg__PanTilt * lhs, const tidybot_msgs__msg__PanTilt * rhs);

/// Copy a msg/PanTilt message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
bool
tidybot_msgs__msg__PanTilt__copy(
  const tidybot_msgs__msg__PanTilt * input,
  tidybot_msgs__msg__PanTilt * output);

/// Initialize array of msg/PanTilt messages.
/**
 * It allocates the memory for the number of elements and calls
 * tidybot_msgs__msg__PanTilt__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
bool
tidybot_msgs__msg__PanTilt__Sequence__init(tidybot_msgs__msg__PanTilt__Sequence * array, size_t size);

/// Finalize array of msg/PanTilt messages.
/**
 * It calls
 * tidybot_msgs__msg__PanTilt__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
void
tidybot_msgs__msg__PanTilt__Sequence__fini(tidybot_msgs__msg__PanTilt__Sequence * array);

/// Create array of msg/PanTilt messages.
/**
 * It allocates the memory for the array and calls
 * tidybot_msgs__msg__PanTilt__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
tidybot_msgs__msg__PanTilt__Sequence *
tidybot_msgs__msg__PanTilt__Sequence__create(size_t size);

/// Destroy array of msg/PanTilt messages.
/**
 * It calls
 * tidybot_msgs__msg__PanTilt__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
void
tidybot_msgs__msg__PanTilt__Sequence__destroy(tidybot_msgs__msg__PanTilt__Sequence * array);

/// Check for msg/PanTilt message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
bool
tidybot_msgs__msg__PanTilt__Sequence__are_equal(const tidybot_msgs__msg__PanTilt__Sequence * lhs, const tidybot_msgs__msg__PanTilt__Sequence * rhs);

/// Copy an array of msg/PanTilt messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
bool
tidybot_msgs__msg__PanTilt__Sequence__copy(
  const tidybot_msgs__msg__PanTilt__Sequence * input,
  tidybot_msgs__msg__PanTilt__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // TIDYBOT_MSGS__MSG__DETAIL__PAN_TILT__FUNCTIONS_H_
