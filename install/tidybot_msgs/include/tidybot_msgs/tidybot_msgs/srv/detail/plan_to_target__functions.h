// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from tidybot_msgs:srv/PlanToTarget.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__FUNCTIONS_H_
#define TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "tidybot_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "tidybot_msgs/srv/detail/plan_to_target__struct.h"

/// Initialize srv/PlanToTarget message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * tidybot_msgs__srv__PlanToTarget_Request
 * )) before or use
 * tidybot_msgs__srv__PlanToTarget_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
bool
tidybot_msgs__srv__PlanToTarget_Request__init(tidybot_msgs__srv__PlanToTarget_Request * msg);

/// Finalize srv/PlanToTarget message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
void
tidybot_msgs__srv__PlanToTarget_Request__fini(tidybot_msgs__srv__PlanToTarget_Request * msg);

/// Create srv/PlanToTarget message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * tidybot_msgs__srv__PlanToTarget_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
tidybot_msgs__srv__PlanToTarget_Request *
tidybot_msgs__srv__PlanToTarget_Request__create();

/// Destroy srv/PlanToTarget message.
/**
 * It calls
 * tidybot_msgs__srv__PlanToTarget_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
void
tidybot_msgs__srv__PlanToTarget_Request__destroy(tidybot_msgs__srv__PlanToTarget_Request * msg);

/// Check for srv/PlanToTarget message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
bool
tidybot_msgs__srv__PlanToTarget_Request__are_equal(const tidybot_msgs__srv__PlanToTarget_Request * lhs, const tidybot_msgs__srv__PlanToTarget_Request * rhs);

/// Copy a srv/PlanToTarget message.
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
tidybot_msgs__srv__PlanToTarget_Request__copy(
  const tidybot_msgs__srv__PlanToTarget_Request * input,
  tidybot_msgs__srv__PlanToTarget_Request * output);

/// Initialize array of srv/PlanToTarget messages.
/**
 * It allocates the memory for the number of elements and calls
 * tidybot_msgs__srv__PlanToTarget_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
bool
tidybot_msgs__srv__PlanToTarget_Request__Sequence__init(tidybot_msgs__srv__PlanToTarget_Request__Sequence * array, size_t size);

/// Finalize array of srv/PlanToTarget messages.
/**
 * It calls
 * tidybot_msgs__srv__PlanToTarget_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
void
tidybot_msgs__srv__PlanToTarget_Request__Sequence__fini(tidybot_msgs__srv__PlanToTarget_Request__Sequence * array);

/// Create array of srv/PlanToTarget messages.
/**
 * It allocates the memory for the array and calls
 * tidybot_msgs__srv__PlanToTarget_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
tidybot_msgs__srv__PlanToTarget_Request__Sequence *
tidybot_msgs__srv__PlanToTarget_Request__Sequence__create(size_t size);

/// Destroy array of srv/PlanToTarget messages.
/**
 * It calls
 * tidybot_msgs__srv__PlanToTarget_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
void
tidybot_msgs__srv__PlanToTarget_Request__Sequence__destroy(tidybot_msgs__srv__PlanToTarget_Request__Sequence * array);

/// Check for srv/PlanToTarget message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
bool
tidybot_msgs__srv__PlanToTarget_Request__Sequence__are_equal(const tidybot_msgs__srv__PlanToTarget_Request__Sequence * lhs, const tidybot_msgs__srv__PlanToTarget_Request__Sequence * rhs);

/// Copy an array of srv/PlanToTarget messages.
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
tidybot_msgs__srv__PlanToTarget_Request__Sequence__copy(
  const tidybot_msgs__srv__PlanToTarget_Request__Sequence * input,
  tidybot_msgs__srv__PlanToTarget_Request__Sequence * output);

/// Initialize srv/PlanToTarget message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * tidybot_msgs__srv__PlanToTarget_Response
 * )) before or use
 * tidybot_msgs__srv__PlanToTarget_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
bool
tidybot_msgs__srv__PlanToTarget_Response__init(tidybot_msgs__srv__PlanToTarget_Response * msg);

/// Finalize srv/PlanToTarget message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
void
tidybot_msgs__srv__PlanToTarget_Response__fini(tidybot_msgs__srv__PlanToTarget_Response * msg);

/// Create srv/PlanToTarget message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * tidybot_msgs__srv__PlanToTarget_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
tidybot_msgs__srv__PlanToTarget_Response *
tidybot_msgs__srv__PlanToTarget_Response__create();

/// Destroy srv/PlanToTarget message.
/**
 * It calls
 * tidybot_msgs__srv__PlanToTarget_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
void
tidybot_msgs__srv__PlanToTarget_Response__destroy(tidybot_msgs__srv__PlanToTarget_Response * msg);

/// Check for srv/PlanToTarget message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
bool
tidybot_msgs__srv__PlanToTarget_Response__are_equal(const tidybot_msgs__srv__PlanToTarget_Response * lhs, const tidybot_msgs__srv__PlanToTarget_Response * rhs);

/// Copy a srv/PlanToTarget message.
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
tidybot_msgs__srv__PlanToTarget_Response__copy(
  const tidybot_msgs__srv__PlanToTarget_Response * input,
  tidybot_msgs__srv__PlanToTarget_Response * output);

/// Initialize array of srv/PlanToTarget messages.
/**
 * It allocates the memory for the number of elements and calls
 * tidybot_msgs__srv__PlanToTarget_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
bool
tidybot_msgs__srv__PlanToTarget_Response__Sequence__init(tidybot_msgs__srv__PlanToTarget_Response__Sequence * array, size_t size);

/// Finalize array of srv/PlanToTarget messages.
/**
 * It calls
 * tidybot_msgs__srv__PlanToTarget_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
void
tidybot_msgs__srv__PlanToTarget_Response__Sequence__fini(tidybot_msgs__srv__PlanToTarget_Response__Sequence * array);

/// Create array of srv/PlanToTarget messages.
/**
 * It allocates the memory for the array and calls
 * tidybot_msgs__srv__PlanToTarget_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
tidybot_msgs__srv__PlanToTarget_Response__Sequence *
tidybot_msgs__srv__PlanToTarget_Response__Sequence__create(size_t size);

/// Destroy array of srv/PlanToTarget messages.
/**
 * It calls
 * tidybot_msgs__srv__PlanToTarget_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
void
tidybot_msgs__srv__PlanToTarget_Response__Sequence__destroy(tidybot_msgs__srv__PlanToTarget_Response__Sequence * array);

/// Check for srv/PlanToTarget message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_tidybot_msgs
bool
tidybot_msgs__srv__PlanToTarget_Response__Sequence__are_equal(const tidybot_msgs__srv__PlanToTarget_Response__Sequence * lhs, const tidybot_msgs__srv__PlanToTarget_Response__Sequence * rhs);

/// Copy an array of srv/PlanToTarget messages.
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
tidybot_msgs__srv__PlanToTarget_Response__Sequence__copy(
  const tidybot_msgs__srv__PlanToTarget_Response__Sequence * input,
  tidybot_msgs__srv__PlanToTarget_Response__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__FUNCTIONS_H_
