// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from tidybot_msgs:srv/AudioRecord.idl
// generated code does not contain a copyright notice
#include "tidybot_msgs/srv/detail/audio_record__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

bool
tidybot_msgs__srv__AudioRecord_Request__init(tidybot_msgs__srv__AudioRecord_Request * msg)
{
  if (!msg) {
    return false;
  }
  // start
  return true;
}

void
tidybot_msgs__srv__AudioRecord_Request__fini(tidybot_msgs__srv__AudioRecord_Request * msg)
{
  if (!msg) {
    return;
  }
  // start
}

bool
tidybot_msgs__srv__AudioRecord_Request__are_equal(const tidybot_msgs__srv__AudioRecord_Request * lhs, const tidybot_msgs__srv__AudioRecord_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // start
  if (lhs->start != rhs->start) {
    return false;
  }
  return true;
}

bool
tidybot_msgs__srv__AudioRecord_Request__copy(
  const tidybot_msgs__srv__AudioRecord_Request * input,
  tidybot_msgs__srv__AudioRecord_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // start
  output->start = input->start;
  return true;
}

tidybot_msgs__srv__AudioRecord_Request *
tidybot_msgs__srv__AudioRecord_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__srv__AudioRecord_Request * msg = (tidybot_msgs__srv__AudioRecord_Request *)allocator.allocate(sizeof(tidybot_msgs__srv__AudioRecord_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tidybot_msgs__srv__AudioRecord_Request));
  bool success = tidybot_msgs__srv__AudioRecord_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tidybot_msgs__srv__AudioRecord_Request__destroy(tidybot_msgs__srv__AudioRecord_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tidybot_msgs__srv__AudioRecord_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tidybot_msgs__srv__AudioRecord_Request__Sequence__init(tidybot_msgs__srv__AudioRecord_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__srv__AudioRecord_Request * data = NULL;

  if (size) {
    data = (tidybot_msgs__srv__AudioRecord_Request *)allocator.zero_allocate(size, sizeof(tidybot_msgs__srv__AudioRecord_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tidybot_msgs__srv__AudioRecord_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tidybot_msgs__srv__AudioRecord_Request__fini(&data[i - 1]);
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
tidybot_msgs__srv__AudioRecord_Request__Sequence__fini(tidybot_msgs__srv__AudioRecord_Request__Sequence * array)
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
      tidybot_msgs__srv__AudioRecord_Request__fini(&array->data[i]);
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

tidybot_msgs__srv__AudioRecord_Request__Sequence *
tidybot_msgs__srv__AudioRecord_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__srv__AudioRecord_Request__Sequence * array = (tidybot_msgs__srv__AudioRecord_Request__Sequence *)allocator.allocate(sizeof(tidybot_msgs__srv__AudioRecord_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tidybot_msgs__srv__AudioRecord_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tidybot_msgs__srv__AudioRecord_Request__Sequence__destroy(tidybot_msgs__srv__AudioRecord_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tidybot_msgs__srv__AudioRecord_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tidybot_msgs__srv__AudioRecord_Request__Sequence__are_equal(const tidybot_msgs__srv__AudioRecord_Request__Sequence * lhs, const tidybot_msgs__srv__AudioRecord_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tidybot_msgs__srv__AudioRecord_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tidybot_msgs__srv__AudioRecord_Request__Sequence__copy(
  const tidybot_msgs__srv__AudioRecord_Request__Sequence * input,
  tidybot_msgs__srv__AudioRecord_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tidybot_msgs__srv__AudioRecord_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tidybot_msgs__srv__AudioRecord_Request * data =
      (tidybot_msgs__srv__AudioRecord_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tidybot_msgs__srv__AudioRecord_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tidybot_msgs__srv__AudioRecord_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tidybot_msgs__srv__AudioRecord_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `message`
#include "rosidl_runtime_c/string_functions.h"
// Member `audio_data`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
tidybot_msgs__srv__AudioRecord_Response__init(tidybot_msgs__srv__AudioRecord_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  // message
  if (!rosidl_runtime_c__String__init(&msg->message)) {
    tidybot_msgs__srv__AudioRecord_Response__fini(msg);
    return false;
  }
  // audio_data
  if (!rosidl_runtime_c__float__Sequence__init(&msg->audio_data, 0)) {
    tidybot_msgs__srv__AudioRecord_Response__fini(msg);
    return false;
  }
  // sample_rate
  // duration
  return true;
}

void
tidybot_msgs__srv__AudioRecord_Response__fini(tidybot_msgs__srv__AudioRecord_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
  // message
  rosidl_runtime_c__String__fini(&msg->message);
  // audio_data
  rosidl_runtime_c__float__Sequence__fini(&msg->audio_data);
  // sample_rate
  // duration
}

bool
tidybot_msgs__srv__AudioRecord_Response__are_equal(const tidybot_msgs__srv__AudioRecord_Response * lhs, const tidybot_msgs__srv__AudioRecord_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  // message
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->message), &(rhs->message)))
  {
    return false;
  }
  // audio_data
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->audio_data), &(rhs->audio_data)))
  {
    return false;
  }
  // sample_rate
  if (lhs->sample_rate != rhs->sample_rate) {
    return false;
  }
  // duration
  if (lhs->duration != rhs->duration) {
    return false;
  }
  return true;
}

bool
tidybot_msgs__srv__AudioRecord_Response__copy(
  const tidybot_msgs__srv__AudioRecord_Response * input,
  tidybot_msgs__srv__AudioRecord_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  // message
  if (!rosidl_runtime_c__String__copy(
      &(input->message), &(output->message)))
  {
    return false;
  }
  // audio_data
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->audio_data), &(output->audio_data)))
  {
    return false;
  }
  // sample_rate
  output->sample_rate = input->sample_rate;
  // duration
  output->duration = input->duration;
  return true;
}

tidybot_msgs__srv__AudioRecord_Response *
tidybot_msgs__srv__AudioRecord_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__srv__AudioRecord_Response * msg = (tidybot_msgs__srv__AudioRecord_Response *)allocator.allocate(sizeof(tidybot_msgs__srv__AudioRecord_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tidybot_msgs__srv__AudioRecord_Response));
  bool success = tidybot_msgs__srv__AudioRecord_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tidybot_msgs__srv__AudioRecord_Response__destroy(tidybot_msgs__srv__AudioRecord_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tidybot_msgs__srv__AudioRecord_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tidybot_msgs__srv__AudioRecord_Response__Sequence__init(tidybot_msgs__srv__AudioRecord_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__srv__AudioRecord_Response * data = NULL;

  if (size) {
    data = (tidybot_msgs__srv__AudioRecord_Response *)allocator.zero_allocate(size, sizeof(tidybot_msgs__srv__AudioRecord_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tidybot_msgs__srv__AudioRecord_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tidybot_msgs__srv__AudioRecord_Response__fini(&data[i - 1]);
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
tidybot_msgs__srv__AudioRecord_Response__Sequence__fini(tidybot_msgs__srv__AudioRecord_Response__Sequence * array)
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
      tidybot_msgs__srv__AudioRecord_Response__fini(&array->data[i]);
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

tidybot_msgs__srv__AudioRecord_Response__Sequence *
tidybot_msgs__srv__AudioRecord_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tidybot_msgs__srv__AudioRecord_Response__Sequence * array = (tidybot_msgs__srv__AudioRecord_Response__Sequence *)allocator.allocate(sizeof(tidybot_msgs__srv__AudioRecord_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tidybot_msgs__srv__AudioRecord_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tidybot_msgs__srv__AudioRecord_Response__Sequence__destroy(tidybot_msgs__srv__AudioRecord_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tidybot_msgs__srv__AudioRecord_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tidybot_msgs__srv__AudioRecord_Response__Sequence__are_equal(const tidybot_msgs__srv__AudioRecord_Response__Sequence * lhs, const tidybot_msgs__srv__AudioRecord_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tidybot_msgs__srv__AudioRecord_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tidybot_msgs__srv__AudioRecord_Response__Sequence__copy(
  const tidybot_msgs__srv__AudioRecord_Response__Sequence * input,
  tidybot_msgs__srv__AudioRecord_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tidybot_msgs__srv__AudioRecord_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tidybot_msgs__srv__AudioRecord_Response * data =
      (tidybot_msgs__srv__AudioRecord_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tidybot_msgs__srv__AudioRecord_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tidybot_msgs__srv__AudioRecord_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tidybot_msgs__srv__AudioRecord_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
