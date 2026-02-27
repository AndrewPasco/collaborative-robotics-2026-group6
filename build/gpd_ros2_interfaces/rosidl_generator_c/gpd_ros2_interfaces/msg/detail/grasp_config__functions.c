// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from gpd_ros2_interfaces:msg/GraspConfig.idl
// generated code does not contain a copyright notice
#include "gpd_ros2_interfaces/msg/detail/grasp_config__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `position`
// Member `sample`
#include "geometry_msgs/msg/detail/point__functions.h"
// Member `approach`
// Member `binormal`
// Member `axis`
#include "geometry_msgs/msg/detail/vector3__functions.h"

bool
gpd_ros2_interfaces__msg__GraspConfig__init(gpd_ros2_interfaces__msg__GraspConfig * msg)
{
  if (!msg) {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__init(&msg->position)) {
    gpd_ros2_interfaces__msg__GraspConfig__fini(msg);
    return false;
  }
  // approach
  if (!geometry_msgs__msg__Vector3__init(&msg->approach)) {
    gpd_ros2_interfaces__msg__GraspConfig__fini(msg);
    return false;
  }
  // binormal
  if (!geometry_msgs__msg__Vector3__init(&msg->binormal)) {
    gpd_ros2_interfaces__msg__GraspConfig__fini(msg);
    return false;
  }
  // axis
  if (!geometry_msgs__msg__Vector3__init(&msg->axis)) {
    gpd_ros2_interfaces__msg__GraspConfig__fini(msg);
    return false;
  }
  // width
  // score
  // sample
  if (!geometry_msgs__msg__Point__init(&msg->sample)) {
    gpd_ros2_interfaces__msg__GraspConfig__fini(msg);
    return false;
  }
  return true;
}

void
gpd_ros2_interfaces__msg__GraspConfig__fini(gpd_ros2_interfaces__msg__GraspConfig * msg)
{
  if (!msg) {
    return;
  }
  // position
  geometry_msgs__msg__Point__fini(&msg->position);
  // approach
  geometry_msgs__msg__Vector3__fini(&msg->approach);
  // binormal
  geometry_msgs__msg__Vector3__fini(&msg->binormal);
  // axis
  geometry_msgs__msg__Vector3__fini(&msg->axis);
  // width
  // score
  // sample
  geometry_msgs__msg__Point__fini(&msg->sample);
}

bool
gpd_ros2_interfaces__msg__GraspConfig__are_equal(const gpd_ros2_interfaces__msg__GraspConfig * lhs, const gpd_ros2_interfaces__msg__GraspConfig * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->position), &(rhs->position)))
  {
    return false;
  }
  // approach
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->approach), &(rhs->approach)))
  {
    return false;
  }
  // binormal
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->binormal), &(rhs->binormal)))
  {
    return false;
  }
  // axis
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->axis), &(rhs->axis)))
  {
    return false;
  }
  // width
  if (lhs->width != rhs->width) {
    return false;
  }
  // score
  if (lhs->score != rhs->score) {
    return false;
  }
  // sample
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->sample), &(rhs->sample)))
  {
    return false;
  }
  return true;
}

bool
gpd_ros2_interfaces__msg__GraspConfig__copy(
  const gpd_ros2_interfaces__msg__GraspConfig * input,
  gpd_ros2_interfaces__msg__GraspConfig * output)
{
  if (!input || !output) {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__copy(
      &(input->position), &(output->position)))
  {
    return false;
  }
  // approach
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->approach), &(output->approach)))
  {
    return false;
  }
  // binormal
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->binormal), &(output->binormal)))
  {
    return false;
  }
  // axis
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->axis), &(output->axis)))
  {
    return false;
  }
  // width
  output->width = input->width;
  // score
  output->score = input->score;
  // sample
  if (!geometry_msgs__msg__Point__copy(
      &(input->sample), &(output->sample)))
  {
    return false;
  }
  return true;
}

gpd_ros2_interfaces__msg__GraspConfig *
gpd_ros2_interfaces__msg__GraspConfig__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__GraspConfig * msg = (gpd_ros2_interfaces__msg__GraspConfig *)allocator.allocate(sizeof(gpd_ros2_interfaces__msg__GraspConfig), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(gpd_ros2_interfaces__msg__GraspConfig));
  bool success = gpd_ros2_interfaces__msg__GraspConfig__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
gpd_ros2_interfaces__msg__GraspConfig__destroy(gpd_ros2_interfaces__msg__GraspConfig * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    gpd_ros2_interfaces__msg__GraspConfig__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
gpd_ros2_interfaces__msg__GraspConfig__Sequence__init(gpd_ros2_interfaces__msg__GraspConfig__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__GraspConfig * data = NULL;

  if (size) {
    data = (gpd_ros2_interfaces__msg__GraspConfig *)allocator.zero_allocate(size, sizeof(gpd_ros2_interfaces__msg__GraspConfig), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = gpd_ros2_interfaces__msg__GraspConfig__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        gpd_ros2_interfaces__msg__GraspConfig__fini(&data[i - 1]);
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
gpd_ros2_interfaces__msg__GraspConfig__Sequence__fini(gpd_ros2_interfaces__msg__GraspConfig__Sequence * array)
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
      gpd_ros2_interfaces__msg__GraspConfig__fini(&array->data[i]);
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

gpd_ros2_interfaces__msg__GraspConfig__Sequence *
gpd_ros2_interfaces__msg__GraspConfig__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__GraspConfig__Sequence * array = (gpd_ros2_interfaces__msg__GraspConfig__Sequence *)allocator.allocate(sizeof(gpd_ros2_interfaces__msg__GraspConfig__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = gpd_ros2_interfaces__msg__GraspConfig__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
gpd_ros2_interfaces__msg__GraspConfig__Sequence__destroy(gpd_ros2_interfaces__msg__GraspConfig__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    gpd_ros2_interfaces__msg__GraspConfig__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
gpd_ros2_interfaces__msg__GraspConfig__Sequence__are_equal(const gpd_ros2_interfaces__msg__GraspConfig__Sequence * lhs, const gpd_ros2_interfaces__msg__GraspConfig__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!gpd_ros2_interfaces__msg__GraspConfig__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
gpd_ros2_interfaces__msg__GraspConfig__Sequence__copy(
  const gpd_ros2_interfaces__msg__GraspConfig__Sequence * input,
  gpd_ros2_interfaces__msg__GraspConfig__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(gpd_ros2_interfaces__msg__GraspConfig);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    gpd_ros2_interfaces__msg__GraspConfig * data =
      (gpd_ros2_interfaces__msg__GraspConfig *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!gpd_ros2_interfaces__msg__GraspConfig__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          gpd_ros2_interfaces__msg__GraspConfig__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!gpd_ros2_interfaces__msg__GraspConfig__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
