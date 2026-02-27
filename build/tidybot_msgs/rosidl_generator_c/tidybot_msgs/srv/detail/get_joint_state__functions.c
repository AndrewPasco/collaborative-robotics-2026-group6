// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from tidybot_msgs:srv/GetJointState.idl
// generated code does not contain a copyright notice
#include "tidybot_msgs/srv/detail/get_joint_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `joint_names`
#include "rosidl_runtime_c/string_functions.h"

bool
tidybot_msgs__srv__GetJointState_Request__init(tidybot_msgs__srv__GetJointState_Request * msg)
{
  if (!msg) {
    return false;
  }
  // joint_names
  if (!rosidl_runtime_c__String__Sequence__init(&msg->joint_names, 0)) {
    tidybot_msgs__srv__GetJointState_Request__fini(msg);
    return false;
  }
  return true;
}

void
tidybot_msgs__srv__GetJointState_Request__fini(tidybot_msgs__srv__GetJointState_Request * msg)
{
  if (!msg) {
    return;
  }
  // joint_names
  rosidl_runtime_c__String__Sequence__fini(&msg->joint_names);
}

bool
tidybot_msgs__srv__GetJointState_Request__are_equal(const tidybot_msgs__srv__GetJointState_Request * lhs, const tidybot_msgs__srv__GetJointState_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // joint_names
  if (!rosidl_runtime_c__String__Sequence__are_equal(
      &(lhs->joint_names), &(rhs->joint_names)))
  {
    return false;
  }
  return true;
}

bool
tidybot_msgs__srv__GetJointState_Request__copy(
  const tidybot_msgs__srv__GetJointState_Request * input,
  tidybot_msgs__srv__GetJointState_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // joint_names
  if (!rosidl_runtime_c__String__Sequence__copy(
      &(input->joint_names), &(output->joint_names)))
  {
    return false;
  }
  return true;
}

tidybot_msgs__srv__GetJointState_Request *
tidybot_msgs__srv__GetJointState_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__srv__GetJointState_Request * msg = (tidybot_msgs__srv__GetJointState_Request *)allocator.allocate(sizeof(tidybot_msgs__srv__GetJointState_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tidybot_msgs__srv__GetJointState_Request));
  bool success = tidybot_msgs__srv__GetJointState_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tidybot_msgs__srv__GetJointState_Request__destroy(tidybot_msgs__srv__GetJointState_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tidybot_msgs__srv__GetJointState_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tidybot_msgs__srv__GetJointState_Request__Sequence__init(tidybot_msgs__srv__GetJointState_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__srv__GetJointState_Request * data = NULL;

  if (size) {
    data = (tidybot_msgs__srv__GetJointState_Request *)allocator.zero_allocate(size, sizeof(tidybot_msgs__srv__GetJointState_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tidybot_msgs__srv__GetJointState_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tidybot_msgs__srv__GetJointState_Request__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
tidybot_msgs__srv__GetJointState_Request__Sequence__fini(tidybot_msgs__srv__GetJointState_Request__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      tidybot_msgs__srv__GetJointState_Request__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

tidybot_msgs__srv__GetJointState_Request__Sequence *
tidybot_msgs__srv__GetJointState_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__srv__GetJointState_Request__Sequence * array = (tidybot_msgs__srv__GetJointState_Request__Sequence *)allocator.allocate(sizeof(tidybot_msgs__srv__GetJointState_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tidybot_msgs__srv__GetJointState_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tidybot_msgs__srv__GetJointState_Request__Sequence__destroy(tidybot_msgs__srv__GetJointState_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tidybot_msgs__srv__GetJointState_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tidybot_msgs__srv__GetJointState_Request__Sequence__are_equal(const tidybot_msgs__srv__GetJointState_Request__Sequence * lhs, const tidybot_msgs__srv__GetJointState_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tidybot_msgs__srv__GetJointState_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tidybot_msgs__srv__GetJointState_Request__Sequence__copy(
  const tidybot_msgs__srv__GetJointState_Request__Sequence * input,
  tidybot_msgs__srv__GetJointState_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tidybot_msgs__srv__GetJointState_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tidybot_msgs__srv__GetJointState_Request * data =
      (tidybot_msgs__srv__GetJointState_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tidybot_msgs__srv__GetJointState_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tidybot_msgs__srv__GetJointState_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tidybot_msgs__srv__GetJointState_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `state`
#include "sensor_msgs/msg/detail/joint_state__functions.h"

bool
tidybot_msgs__srv__GetJointState_Response__init(tidybot_msgs__srv__GetJointState_Response * msg)
{
  if (!msg) {
    return false;
  }
  // state
  if (!sensor_msgs__msg__JointState__init(&msg->state)) {
    tidybot_msgs__srv__GetJointState_Response__fini(msg);
    return false;
  }
  return true;
}

void
tidybot_msgs__srv__GetJointState_Response__fini(tidybot_msgs__srv__GetJointState_Response * msg)
{
  if (!msg) {
    return;
  }
  // state
  sensor_msgs__msg__JointState__fini(&msg->state);
}

bool
tidybot_msgs__srv__GetJointState_Response__are_equal(const tidybot_msgs__srv__GetJointState_Response * lhs, const tidybot_msgs__srv__GetJointState_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // state
  if (!sensor_msgs__msg__JointState__are_equal(
      &(lhs->state), &(rhs->state)))
  {
    return false;
  }
  return true;
}

bool
tidybot_msgs__srv__GetJointState_Response__copy(
  const tidybot_msgs__srv__GetJointState_Response * input,
  tidybot_msgs__srv__GetJointState_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // state
  if (!sensor_msgs__msg__JointState__copy(
      &(input->state), &(output->state)))
  {
    return false;
  }
  return true;
}

tidybot_msgs__srv__GetJointState_Response *
tidybot_msgs__srv__GetJointState_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__srv__GetJointState_Response * msg = (tidybot_msgs__srv__GetJointState_Response *)allocator.allocate(sizeof(tidybot_msgs__srv__GetJointState_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tidybot_msgs__srv__GetJointState_Response));
  bool success = tidybot_msgs__srv__GetJointState_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tidybot_msgs__srv__GetJointState_Response__destroy(tidybot_msgs__srv__GetJointState_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tidybot_msgs__srv__GetJointState_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tidybot_msgs__srv__GetJointState_Response__Sequence__init(tidybot_msgs__srv__GetJointState_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__srv__GetJointState_Response * data = NULL;

  if (size) {
    data = (tidybot_msgs__srv__GetJointState_Response *)allocator.zero_allocate(size, sizeof(tidybot_msgs__srv__GetJointState_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tidybot_msgs__srv__GetJointState_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tidybot_msgs__srv__GetJointState_Response__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
tidybot_msgs__srv__GetJointState_Response__Sequence__fini(tidybot_msgs__srv__GetJointState_Response__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      tidybot_msgs__srv__GetJointState_Response__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

tidybot_msgs__srv__GetJointState_Response__Sequence *
tidybot_msgs__srv__GetJointState_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__srv__GetJointState_Response__Sequence * array = (tidybot_msgs__srv__GetJointState_Response__Sequence *)allocator.allocate(sizeof(tidybot_msgs__srv__GetJointState_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tidybot_msgs__srv__GetJointState_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tidybot_msgs__srv__GetJointState_Response__Sequence__destroy(tidybot_msgs__srv__GetJointState_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tidybot_msgs__srv__GetJointState_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tidybot_msgs__srv__GetJointState_Response__Sequence__are_equal(const tidybot_msgs__srv__GetJointState_Response__Sequence * lhs, const tidybot_msgs__srv__GetJointState_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tidybot_msgs__srv__GetJointState_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tidybot_msgs__srv__GetJointState_Response__Sequence__copy(
  const tidybot_msgs__srv__GetJointState_Response__Sequence * input,
  tidybot_msgs__srv__GetJointState_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tidybot_msgs__srv__GetJointState_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tidybot_msgs__srv__GetJointState_Response * data =
      (tidybot_msgs__srv__GetJointState_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tidybot_msgs__srv__GetJointState_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tidybot_msgs__srv__GetJointState_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tidybot_msgs__srv__GetJointState_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
