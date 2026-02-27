// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from gpd_ros2_interfaces:srv/DetectGrasps.idl
// generated code does not contain a copyright notice
#include "gpd_ros2_interfaces/srv/detail/detect_grasps__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `cloud_indexed`
#include "gpd_ros2_interfaces/msg/detail/cloud_indexed__functions.h"

bool
gpd_ros2_interfaces__srv__DetectGrasps_Request__init(gpd_ros2_interfaces__srv__DetectGrasps_Request * msg)
{
  if (!msg) {
    return false;
  }
  // cloud_indexed
  if (!gpd_ros2_interfaces__msg__CloudIndexed__init(&msg->cloud_indexed)) {
    gpd_ros2_interfaces__srv__DetectGrasps_Request__fini(msg);
    return false;
  }
  return true;
}

void
gpd_ros2_interfaces__srv__DetectGrasps_Request__fini(gpd_ros2_interfaces__srv__DetectGrasps_Request * msg)
{
  if (!msg) {
    return;
  }
  // cloud_indexed
  gpd_ros2_interfaces__msg__CloudIndexed__fini(&msg->cloud_indexed);
}

bool
gpd_ros2_interfaces__srv__DetectGrasps_Request__are_equal(const gpd_ros2_interfaces__srv__DetectGrasps_Request * lhs, const gpd_ros2_interfaces__srv__DetectGrasps_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // cloud_indexed
  if (!gpd_ros2_interfaces__msg__CloudIndexed__are_equal(
      &(lhs->cloud_indexed), &(rhs->cloud_indexed)))
  {
    return false;
  }
  return true;
}

bool
gpd_ros2_interfaces__srv__DetectGrasps_Request__copy(
  const gpd_ros2_interfaces__srv__DetectGrasps_Request * input,
  gpd_ros2_interfaces__srv__DetectGrasps_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // cloud_indexed
  if (!gpd_ros2_interfaces__msg__CloudIndexed__copy(
      &(input->cloud_indexed), &(output->cloud_indexed)))
  {
    return false;
  }
  return true;
}

gpd_ros2_interfaces__srv__DetectGrasps_Request *
gpd_ros2_interfaces__srv__DetectGrasps_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__srv__DetectGrasps_Request * msg = (gpd_ros2_interfaces__srv__DetectGrasps_Request *)allocator.allocate(sizeof(gpd_ros2_interfaces__srv__DetectGrasps_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(gpd_ros2_interfaces__srv__DetectGrasps_Request));
  bool success = gpd_ros2_interfaces__srv__DetectGrasps_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
gpd_ros2_interfaces__srv__DetectGrasps_Request__destroy(gpd_ros2_interfaces__srv__DetectGrasps_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    gpd_ros2_interfaces__srv__DetectGrasps_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence__init(gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__srv__DetectGrasps_Request * data = NULL;

  if (size) {
    data = (gpd_ros2_interfaces__srv__DetectGrasps_Request *)allocator.zero_allocate(size, sizeof(gpd_ros2_interfaces__srv__DetectGrasps_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = gpd_ros2_interfaces__srv__DetectGrasps_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        gpd_ros2_interfaces__srv__DetectGrasps_Request__fini(&data[i - 1]);
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
gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence__fini(gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence * array)
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
      gpd_ros2_interfaces__srv__DetectGrasps_Request__fini(&array->data[i]);
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

gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence *
gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence * array = (gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence *)allocator.allocate(sizeof(gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence__destroy(gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence__are_equal(const gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence * lhs, const gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!gpd_ros2_interfaces__srv__DetectGrasps_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence__copy(
  const gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence * input,
  gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(gpd_ros2_interfaces__srv__DetectGrasps_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    gpd_ros2_interfaces__srv__DetectGrasps_Request * data =
      (gpd_ros2_interfaces__srv__DetectGrasps_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!gpd_ros2_interfaces__srv__DetectGrasps_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          gpd_ros2_interfaces__srv__DetectGrasps_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!gpd_ros2_interfaces__srv__DetectGrasps_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `grasp_configs`
#include "gpd_ros2_interfaces/msg/detail/grasp_config_list__functions.h"

bool
gpd_ros2_interfaces__srv__DetectGrasps_Response__init(gpd_ros2_interfaces__srv__DetectGrasps_Response * msg)
{
  if (!msg) {
    return false;
  }
  // grasp_configs
  if (!gpd_ros2_interfaces__msg__GraspConfigList__init(&msg->grasp_configs)) {
    gpd_ros2_interfaces__srv__DetectGrasps_Response__fini(msg);
    return false;
  }
  return true;
}

void
gpd_ros2_interfaces__srv__DetectGrasps_Response__fini(gpd_ros2_interfaces__srv__DetectGrasps_Response * msg)
{
  if (!msg) {
    return;
  }
  // grasp_configs
  gpd_ros2_interfaces__msg__GraspConfigList__fini(&msg->grasp_configs);
}

bool
gpd_ros2_interfaces__srv__DetectGrasps_Response__are_equal(const gpd_ros2_interfaces__srv__DetectGrasps_Response * lhs, const gpd_ros2_interfaces__srv__DetectGrasps_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // grasp_configs
  if (!gpd_ros2_interfaces__msg__GraspConfigList__are_equal(
      &(lhs->grasp_configs), &(rhs->grasp_configs)))
  {
    return false;
  }
  return true;
}

bool
gpd_ros2_interfaces__srv__DetectGrasps_Response__copy(
  const gpd_ros2_interfaces__srv__DetectGrasps_Response * input,
  gpd_ros2_interfaces__srv__DetectGrasps_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // grasp_configs
  if (!gpd_ros2_interfaces__msg__GraspConfigList__copy(
      &(input->grasp_configs), &(output->grasp_configs)))
  {
    return false;
  }
  return true;
}

gpd_ros2_interfaces__srv__DetectGrasps_Response *
gpd_ros2_interfaces__srv__DetectGrasps_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__srv__DetectGrasps_Response * msg = (gpd_ros2_interfaces__srv__DetectGrasps_Response *)allocator.allocate(sizeof(gpd_ros2_interfaces__srv__DetectGrasps_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(gpd_ros2_interfaces__srv__DetectGrasps_Response));
  bool success = gpd_ros2_interfaces__srv__DetectGrasps_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
gpd_ros2_interfaces__srv__DetectGrasps_Response__destroy(gpd_ros2_interfaces__srv__DetectGrasps_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    gpd_ros2_interfaces__srv__DetectGrasps_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence__init(gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__srv__DetectGrasps_Response * data = NULL;

  if (size) {
    data = (gpd_ros2_interfaces__srv__DetectGrasps_Response *)allocator.zero_allocate(size, sizeof(gpd_ros2_interfaces__srv__DetectGrasps_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = gpd_ros2_interfaces__srv__DetectGrasps_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        gpd_ros2_interfaces__srv__DetectGrasps_Response__fini(&data[i - 1]);
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
gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence__fini(gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence * array)
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
      gpd_ros2_interfaces__srv__DetectGrasps_Response__fini(&array->data[i]);
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

gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence *
gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence * array = (gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence *)allocator.allocate(sizeof(gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence__destroy(gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence__are_equal(const gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence * lhs, const gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!gpd_ros2_interfaces__srv__DetectGrasps_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence__copy(
  const gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence * input,
  gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(gpd_ros2_interfaces__srv__DetectGrasps_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    gpd_ros2_interfaces__srv__DetectGrasps_Response * data =
      (gpd_ros2_interfaces__srv__DetectGrasps_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!gpd_ros2_interfaces__srv__DetectGrasps_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          gpd_ros2_interfaces__srv__DetectGrasps_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!gpd_ros2_interfaces__srv__DetectGrasps_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
