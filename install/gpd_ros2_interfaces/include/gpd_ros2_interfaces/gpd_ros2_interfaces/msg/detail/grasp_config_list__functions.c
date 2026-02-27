// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from gpd_ros2_interfaces:msg/GraspConfigList.idl
// generated code does not contain a copyright notice
#include "gpd_ros2_interfaces/msg/detail/grasp_config_list__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `grasps`
#include "gpd_ros2_interfaces/msg/detail/grasp_config__functions.h"

bool
gpd_ros2_interfaces__msg__GraspConfigList__init(gpd_ros2_interfaces__msg__GraspConfigList * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    gpd_ros2_interfaces__msg__GraspConfigList__fini(msg);
    return false;
  }
  // grasps
  if (!gpd_ros2_interfaces__msg__GraspConfig__Sequence__init(&msg->grasps, 0)) {
    gpd_ros2_interfaces__msg__GraspConfigList__fini(msg);
    return false;
  }
  return true;
}

void
gpd_ros2_interfaces__msg__GraspConfigList__fini(gpd_ros2_interfaces__msg__GraspConfigList * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // grasps
  gpd_ros2_interfaces__msg__GraspConfig__Sequence__fini(&msg->grasps);
}

bool
gpd_ros2_interfaces__msg__GraspConfigList__are_equal(const gpd_ros2_interfaces__msg__GraspConfigList * lhs, const gpd_ros2_interfaces__msg__GraspConfigList * rhs)
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
  // grasps
  if (!gpd_ros2_interfaces__msg__GraspConfig__Sequence__are_equal(
      &(lhs->grasps), &(rhs->grasps)))
  {
    return false;
  }
  return true;
}

bool
gpd_ros2_interfaces__msg__GraspConfigList__copy(
  const gpd_ros2_interfaces__msg__GraspConfigList * input,
  gpd_ros2_interfaces__msg__GraspConfigList * output)
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
  // grasps
  if (!gpd_ros2_interfaces__msg__GraspConfig__Sequence__copy(
      &(input->grasps), &(output->grasps)))
  {
    return false;
  }
  return true;
}

gpd_ros2_interfaces__msg__GraspConfigList *
gpd_ros2_interfaces__msg__GraspConfigList__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__GraspConfigList * msg = (gpd_ros2_interfaces__msg__GraspConfigList *)allocator.allocate(sizeof(gpd_ros2_interfaces__msg__GraspConfigList), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(gpd_ros2_interfaces__msg__GraspConfigList));
  bool success = gpd_ros2_interfaces__msg__GraspConfigList__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
gpd_ros2_interfaces__msg__GraspConfigList__destroy(gpd_ros2_interfaces__msg__GraspConfigList * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    gpd_ros2_interfaces__msg__GraspConfigList__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
gpd_ros2_interfaces__msg__GraspConfigList__Sequence__init(gpd_ros2_interfaces__msg__GraspConfigList__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__GraspConfigList * data = NULL;

  if (size) {
    data = (gpd_ros2_interfaces__msg__GraspConfigList *)allocator.zero_allocate(size, sizeof(gpd_ros2_interfaces__msg__GraspConfigList), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = gpd_ros2_interfaces__msg__GraspConfigList__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        gpd_ros2_interfaces__msg__GraspConfigList__fini(&data[i - 1]);
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
gpd_ros2_interfaces__msg__GraspConfigList__Sequence__fini(gpd_ros2_interfaces__msg__GraspConfigList__Sequence * array)
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
      gpd_ros2_interfaces__msg__GraspConfigList__fini(&array->data[i]);
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

gpd_ros2_interfaces__msg__GraspConfigList__Sequence *
gpd_ros2_interfaces__msg__GraspConfigList__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__GraspConfigList__Sequence * array = (gpd_ros2_interfaces__msg__GraspConfigList__Sequence *)allocator.allocate(sizeof(gpd_ros2_interfaces__msg__GraspConfigList__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = gpd_ros2_interfaces__msg__GraspConfigList__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
gpd_ros2_interfaces__msg__GraspConfigList__Sequence__destroy(gpd_ros2_interfaces__msg__GraspConfigList__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    gpd_ros2_interfaces__msg__GraspConfigList__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
gpd_ros2_interfaces__msg__GraspConfigList__Sequence__are_equal(const gpd_ros2_interfaces__msg__GraspConfigList__Sequence * lhs, const gpd_ros2_interfaces__msg__GraspConfigList__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!gpd_ros2_interfaces__msg__GraspConfigList__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
gpd_ros2_interfaces__msg__GraspConfigList__Sequence__copy(
  const gpd_ros2_interfaces__msg__GraspConfigList__Sequence * input,
  gpd_ros2_interfaces__msg__GraspConfigList__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(gpd_ros2_interfaces__msg__GraspConfigList);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    gpd_ros2_interfaces__msg__GraspConfigList * data =
      (gpd_ros2_interfaces__msg__GraspConfigList *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!gpd_ros2_interfaces__msg__GraspConfigList__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          gpd_ros2_interfaces__msg__GraspConfigList__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!gpd_ros2_interfaces__msg__GraspConfigList__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
