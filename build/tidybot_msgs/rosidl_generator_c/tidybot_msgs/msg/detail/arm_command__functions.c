// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from tidybot_msgs:msg/ArmCommand.idl
// generated code does not contain a copyright notice
#include "tidybot_msgs/msg/detail/arm_command__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
tidybot_msgs__msg__ArmCommand__init(tidybot_msgs__msg__ArmCommand * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    tidybot_msgs__msg__ArmCommand__fini(msg);
    return false;
  }
  // joint_positions
  // duration
  return true;
}

void
tidybot_msgs__msg__ArmCommand__fini(tidybot_msgs__msg__ArmCommand * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // joint_positions
  // duration
}

bool
tidybot_msgs__msg__ArmCommand__are_equal(const tidybot_msgs__msg__ArmCommand * lhs, const tidybot_msgs__msg__ArmCommand * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // joint_positions
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->joint_positions[i] != rhs->joint_positions[i]) {
      return false;
    }
  }
  // duration
  if (lhs->duration != rhs->duration) {
    return false;
  }
  return true;
}

bool
tidybot_msgs__msg__ArmCommand__copy(
  const tidybot_msgs__msg__ArmCommand * input,
  tidybot_msgs__msg__ArmCommand * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // joint_positions
  for (size_t i = 0; i < 6; ++i) {
    output->joint_positions[i] = input->joint_positions[i];
  }
  // duration
  output->duration = input->duration;
  return true;
}

tidybot_msgs__msg__ArmCommand *
tidybot_msgs__msg__ArmCommand__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__msg__ArmCommand * msg = (tidybot_msgs__msg__ArmCommand *)allocator.allocate(sizeof(tidybot_msgs__msg__ArmCommand), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tidybot_msgs__msg__ArmCommand));
  bool success = tidybot_msgs__msg__ArmCommand__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tidybot_msgs__msg__ArmCommand__destroy(tidybot_msgs__msg__ArmCommand * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tidybot_msgs__msg__ArmCommand__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tidybot_msgs__msg__ArmCommand__Sequence__init(tidybot_msgs__msg__ArmCommand__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__msg__ArmCommand * data = NULL;

  if (size) {
    data = (tidybot_msgs__msg__ArmCommand *)allocator.zero_allocate(size, sizeof(tidybot_msgs__msg__ArmCommand), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tidybot_msgs__msg__ArmCommand__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tidybot_msgs__msg__ArmCommand__fini(&data[i - 1]);
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
tidybot_msgs__msg__ArmCommand__Sequence__fini(tidybot_msgs__msg__ArmCommand__Sequence * array)
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
      tidybot_msgs__msg__ArmCommand__fini(&array->data[i]);
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

tidybot_msgs__msg__ArmCommand__Sequence *
tidybot_msgs__msg__ArmCommand__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__msg__ArmCommand__Sequence * array = (tidybot_msgs__msg__ArmCommand__Sequence *)allocator.allocate(sizeof(tidybot_msgs__msg__ArmCommand__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tidybot_msgs__msg__ArmCommand__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tidybot_msgs__msg__ArmCommand__Sequence__destroy(tidybot_msgs__msg__ArmCommand__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tidybot_msgs__msg__ArmCommand__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tidybot_msgs__msg__ArmCommand__Sequence__are_equal(const tidybot_msgs__msg__ArmCommand__Sequence * lhs, const tidybot_msgs__msg__ArmCommand__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tidybot_msgs__msg__ArmCommand__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tidybot_msgs__msg__ArmCommand__Sequence__copy(
  const tidybot_msgs__msg__ArmCommand__Sequence * input,
  tidybot_msgs__msg__ArmCommand__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tidybot_msgs__msg__ArmCommand);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tidybot_msgs__msg__ArmCommand * data =
      (tidybot_msgs__msg__ArmCommand *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tidybot_msgs__msg__ArmCommand__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tidybot_msgs__msg__ArmCommand__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tidybot_msgs__msg__ArmCommand__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
