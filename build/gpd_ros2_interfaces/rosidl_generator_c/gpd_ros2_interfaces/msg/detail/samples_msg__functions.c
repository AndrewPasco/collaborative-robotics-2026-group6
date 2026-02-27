// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from gpd_ros2_interfaces:msg/SamplesMsg.idl
// generated code does not contain a copyright notice
#include "gpd_ros2_interfaces/msg/detail/samples_msg__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `samples`
#include "geometry_msgs/msg/detail/point__functions.h"

bool
gpd_ros2_interfaces__msg__SamplesMsg__init(gpd_ros2_interfaces__msg__SamplesMsg * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    gpd_ros2_interfaces__msg__SamplesMsg__fini(msg);
    return false;
  }
  // samples
  if (!geometry_msgs__msg__Point__Sequence__init(&msg->samples, 0)) {
    gpd_ros2_interfaces__msg__SamplesMsg__fini(msg);
    return false;
  }
  return true;
}

void
gpd_ros2_interfaces__msg__SamplesMsg__fini(gpd_ros2_interfaces__msg__SamplesMsg * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // samples
  geometry_msgs__msg__Point__Sequence__fini(&msg->samples);
}

bool
gpd_ros2_interfaces__msg__SamplesMsg__are_equal(const gpd_ros2_interfaces__msg__SamplesMsg * lhs, const gpd_ros2_interfaces__msg__SamplesMsg * rhs)
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
  // samples
  if (!geometry_msgs__msg__Point__Sequence__are_equal(
      &(lhs->samples), &(rhs->samples)))
  {
    return false;
  }
  return true;
}

bool
gpd_ros2_interfaces__msg__SamplesMsg__copy(
  const gpd_ros2_interfaces__msg__SamplesMsg * input,
  gpd_ros2_interfaces__msg__SamplesMsg * output)
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
  // samples
  if (!geometry_msgs__msg__Point__Sequence__copy(
      &(input->samples), &(output->samples)))
  {
    return false;
  }
  return true;
}

gpd_ros2_interfaces__msg__SamplesMsg *
gpd_ros2_interfaces__msg__SamplesMsg__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__SamplesMsg * msg = (gpd_ros2_interfaces__msg__SamplesMsg *)allocator.allocate(sizeof(gpd_ros2_interfaces__msg__SamplesMsg), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(gpd_ros2_interfaces__msg__SamplesMsg));
  bool success = gpd_ros2_interfaces__msg__SamplesMsg__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
gpd_ros2_interfaces__msg__SamplesMsg__destroy(gpd_ros2_interfaces__msg__SamplesMsg * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    gpd_ros2_interfaces__msg__SamplesMsg__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
gpd_ros2_interfaces__msg__SamplesMsg__Sequence__init(gpd_ros2_interfaces__msg__SamplesMsg__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__SamplesMsg * data = NULL;

  if (size) {
    data = (gpd_ros2_interfaces__msg__SamplesMsg *)allocator.zero_allocate(size, sizeof(gpd_ros2_interfaces__msg__SamplesMsg), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = gpd_ros2_interfaces__msg__SamplesMsg__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        gpd_ros2_interfaces__msg__SamplesMsg__fini(&data[i - 1]);
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
gpd_ros2_interfaces__msg__SamplesMsg__Sequence__fini(gpd_ros2_interfaces__msg__SamplesMsg__Sequence * array)
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
      gpd_ros2_interfaces__msg__SamplesMsg__fini(&array->data[i]);
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

gpd_ros2_interfaces__msg__SamplesMsg__Sequence *
gpd_ros2_interfaces__msg__SamplesMsg__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__SamplesMsg__Sequence * array = (gpd_ros2_interfaces__msg__SamplesMsg__Sequence *)allocator.allocate(sizeof(gpd_ros2_interfaces__msg__SamplesMsg__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = gpd_ros2_interfaces__msg__SamplesMsg__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
gpd_ros2_interfaces__msg__SamplesMsg__Sequence__destroy(gpd_ros2_interfaces__msg__SamplesMsg__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    gpd_ros2_interfaces__msg__SamplesMsg__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
gpd_ros2_interfaces__msg__SamplesMsg__Sequence__are_equal(const gpd_ros2_interfaces__msg__SamplesMsg__Sequence * lhs, const gpd_ros2_interfaces__msg__SamplesMsg__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!gpd_ros2_interfaces__msg__SamplesMsg__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
gpd_ros2_interfaces__msg__SamplesMsg__Sequence__copy(
  const gpd_ros2_interfaces__msg__SamplesMsg__Sequence * input,
  gpd_ros2_interfaces__msg__SamplesMsg__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(gpd_ros2_interfaces__msg__SamplesMsg);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    gpd_ros2_interfaces__msg__SamplesMsg * data =
      (gpd_ros2_interfaces__msg__SamplesMsg *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!gpd_ros2_interfaces__msg__SamplesMsg__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          gpd_ros2_interfaces__msg__SamplesMsg__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!gpd_ros2_interfaces__msg__SamplesMsg__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
