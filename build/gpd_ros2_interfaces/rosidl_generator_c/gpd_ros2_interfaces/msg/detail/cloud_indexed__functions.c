// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from gpd_ros2_interfaces:msg/CloudIndexed.idl
// generated code does not contain a copyright notice
#include "gpd_ros2_interfaces/msg/detail/cloud_indexed__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `cloud_sources`
#include "gpd_ros2_interfaces/msg/detail/cloud_sources__functions.h"
// Member `indices`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
gpd_ros2_interfaces__msg__CloudIndexed__init(gpd_ros2_interfaces__msg__CloudIndexed * msg)
{
  if (!msg) {
    return false;
  }
  // cloud_sources
  if (!gpd_ros2_interfaces__msg__CloudSources__init(&msg->cloud_sources)) {
    gpd_ros2_interfaces__msg__CloudIndexed__fini(msg);
    return false;
  }
  // indices
  if (!rosidl_runtime_c__int64__Sequence__init(&msg->indices, 0)) {
    gpd_ros2_interfaces__msg__CloudIndexed__fini(msg);
    return false;
  }
  return true;
}

void
gpd_ros2_interfaces__msg__CloudIndexed__fini(gpd_ros2_interfaces__msg__CloudIndexed * msg)
{
  if (!msg) {
    return;
  }
  // cloud_sources
  gpd_ros2_interfaces__msg__CloudSources__fini(&msg->cloud_sources);
  // indices
  rosidl_runtime_c__int64__Sequence__fini(&msg->indices);
}

bool
gpd_ros2_interfaces__msg__CloudIndexed__are_equal(const gpd_ros2_interfaces__msg__CloudIndexed * lhs, const gpd_ros2_interfaces__msg__CloudIndexed * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // cloud_sources
  if (!gpd_ros2_interfaces__msg__CloudSources__are_equal(
      &(lhs->cloud_sources), &(rhs->cloud_sources)))
  {
    return false;
  }
  // indices
  if (!rosidl_runtime_c__int64__Sequence__are_equal(
      &(lhs->indices), &(rhs->indices)))
  {
    return false;
  }
  return true;
}

bool
gpd_ros2_interfaces__msg__CloudIndexed__copy(
  const gpd_ros2_interfaces__msg__CloudIndexed * input,
  gpd_ros2_interfaces__msg__CloudIndexed * output)
{
  if (!input || !output) {
    return false;
  }
  // cloud_sources
  if (!gpd_ros2_interfaces__msg__CloudSources__copy(
      &(input->cloud_sources), &(output->cloud_sources)))
  {
    return false;
  }
  // indices
  if (!rosidl_runtime_c__int64__Sequence__copy(
      &(input->indices), &(output->indices)))
  {
    return false;
  }
  return true;
}

gpd_ros2_interfaces__msg__CloudIndexed *
gpd_ros2_interfaces__msg__CloudIndexed__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__CloudIndexed * msg = (gpd_ros2_interfaces__msg__CloudIndexed *)allocator.allocate(sizeof(gpd_ros2_interfaces__msg__CloudIndexed), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(gpd_ros2_interfaces__msg__CloudIndexed));
  bool success = gpd_ros2_interfaces__msg__CloudIndexed__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
gpd_ros2_interfaces__msg__CloudIndexed__destroy(gpd_ros2_interfaces__msg__CloudIndexed * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    gpd_ros2_interfaces__msg__CloudIndexed__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
gpd_ros2_interfaces__msg__CloudIndexed__Sequence__init(gpd_ros2_interfaces__msg__CloudIndexed__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__CloudIndexed * data = NULL;

  if (size) {
    data = (gpd_ros2_interfaces__msg__CloudIndexed *)allocator.zero_allocate(size, sizeof(gpd_ros2_interfaces__msg__CloudIndexed), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = gpd_ros2_interfaces__msg__CloudIndexed__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        gpd_ros2_interfaces__msg__CloudIndexed__fini(&data[i - 1]);
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
gpd_ros2_interfaces__msg__CloudIndexed__Sequence__fini(gpd_ros2_interfaces__msg__CloudIndexed__Sequence * array)
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
      gpd_ros2_interfaces__msg__CloudIndexed__fini(&array->data[i]);
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

gpd_ros2_interfaces__msg__CloudIndexed__Sequence *
gpd_ros2_interfaces__msg__CloudIndexed__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__CloudIndexed__Sequence * array = (gpd_ros2_interfaces__msg__CloudIndexed__Sequence *)allocator.allocate(sizeof(gpd_ros2_interfaces__msg__CloudIndexed__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = gpd_ros2_interfaces__msg__CloudIndexed__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
gpd_ros2_interfaces__msg__CloudIndexed__Sequence__destroy(gpd_ros2_interfaces__msg__CloudIndexed__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    gpd_ros2_interfaces__msg__CloudIndexed__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
gpd_ros2_interfaces__msg__CloudIndexed__Sequence__are_equal(const gpd_ros2_interfaces__msg__CloudIndexed__Sequence * lhs, const gpd_ros2_interfaces__msg__CloudIndexed__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!gpd_ros2_interfaces__msg__CloudIndexed__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
gpd_ros2_interfaces__msg__CloudIndexed__Sequence__copy(
  const gpd_ros2_interfaces__msg__CloudIndexed__Sequence * input,
  gpd_ros2_interfaces__msg__CloudIndexed__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(gpd_ros2_interfaces__msg__CloudIndexed);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    gpd_ros2_interfaces__msg__CloudIndexed * data =
      (gpd_ros2_interfaces__msg__CloudIndexed *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!gpd_ros2_interfaces__msg__CloudIndexed__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          gpd_ros2_interfaces__msg__CloudIndexed__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!gpd_ros2_interfaces__msg__CloudIndexed__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
