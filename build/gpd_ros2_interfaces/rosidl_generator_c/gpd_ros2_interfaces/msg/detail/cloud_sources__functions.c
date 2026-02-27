// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from gpd_ros2_interfaces:msg/CloudSources.idl
// generated code does not contain a copyright notice
#include "gpd_ros2_interfaces/msg/detail/cloud_sources__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `cloud`
#include "sensor_msgs/msg/detail/point_cloud2__functions.h"
// Member `camera_source`
#include "rosidl_runtime_c/primitives_sequence_functions.h"
// Member `view_points`
#include "geometry_msgs/msg/detail/point__functions.h"

bool
gpd_ros2_interfaces__msg__CloudSources__init(gpd_ros2_interfaces__msg__CloudSources * msg)
{
  if (!msg) {
    return false;
  }
  // cloud
  if (!sensor_msgs__msg__PointCloud2__init(&msg->cloud)) {
    gpd_ros2_interfaces__msg__CloudSources__fini(msg);
    return false;
  }
  // camera_source
  if (!rosidl_runtime_c__int64__Sequence__init(&msg->camera_source, 0)) {
    gpd_ros2_interfaces__msg__CloudSources__fini(msg);
    return false;
  }
  // view_points
  if (!geometry_msgs__msg__Point__Sequence__init(&msg->view_points, 0)) {
    gpd_ros2_interfaces__msg__CloudSources__fini(msg);
    return false;
  }
  return true;
}

void
gpd_ros2_interfaces__msg__CloudSources__fini(gpd_ros2_interfaces__msg__CloudSources * msg)
{
  if (!msg) {
    return;
  }
  // cloud
  sensor_msgs__msg__PointCloud2__fini(&msg->cloud);
  // camera_source
  rosidl_runtime_c__int64__Sequence__fini(&msg->camera_source);
  // view_points
  geometry_msgs__msg__Point__Sequence__fini(&msg->view_points);
}

bool
gpd_ros2_interfaces__msg__CloudSources__are_equal(const gpd_ros2_interfaces__msg__CloudSources * lhs, const gpd_ros2_interfaces__msg__CloudSources * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // cloud
  if (!sensor_msgs__msg__PointCloud2__are_equal(
      &(lhs->cloud), &(rhs->cloud)))
  {
    return false;
  }
  // camera_source
  if (!rosidl_runtime_c__int64__Sequence__are_equal(
      &(lhs->camera_source), &(rhs->camera_source)))
  {
    return false;
  }
  // view_points
  if (!geometry_msgs__msg__Point__Sequence__are_equal(
      &(lhs->view_points), &(rhs->view_points)))
  {
    return false;
  }
  return true;
}

bool
gpd_ros2_interfaces__msg__CloudSources__copy(
  const gpd_ros2_interfaces__msg__CloudSources * input,
  gpd_ros2_interfaces__msg__CloudSources * output)
{
  if (!input || !output) {
    return false;
  }
  // cloud
  if (!sensor_msgs__msg__PointCloud2__copy(
      &(input->cloud), &(output->cloud)))
  {
    return false;
  }
  // camera_source
  if (!rosidl_runtime_c__int64__Sequence__copy(
      &(input->camera_source), &(output->camera_source)))
  {
    return false;
  }
  // view_points
  if (!geometry_msgs__msg__Point__Sequence__copy(
      &(input->view_points), &(output->view_points)))
  {
    return false;
  }
  return true;
}

gpd_ros2_interfaces__msg__CloudSources *
gpd_ros2_interfaces__msg__CloudSources__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__CloudSources * msg = (gpd_ros2_interfaces__msg__CloudSources *)allocator.allocate(sizeof(gpd_ros2_interfaces__msg__CloudSources), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(gpd_ros2_interfaces__msg__CloudSources));
  bool success = gpd_ros2_interfaces__msg__CloudSources__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
gpd_ros2_interfaces__msg__CloudSources__destroy(gpd_ros2_interfaces__msg__CloudSources * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    gpd_ros2_interfaces__msg__CloudSources__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
gpd_ros2_interfaces__msg__CloudSources__Sequence__init(gpd_ros2_interfaces__msg__CloudSources__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__CloudSources * data = NULL;

  if (size) {
    data = (gpd_ros2_interfaces__msg__CloudSources *)allocator.zero_allocate(size, sizeof(gpd_ros2_interfaces__msg__CloudSources), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = gpd_ros2_interfaces__msg__CloudSources__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        gpd_ros2_interfaces__msg__CloudSources__fini(&data[i - 1]);
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
gpd_ros2_interfaces__msg__CloudSources__Sequence__fini(gpd_ros2_interfaces__msg__CloudSources__Sequence * array)
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
      gpd_ros2_interfaces__msg__CloudSources__fini(&array->data[i]);
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

gpd_ros2_interfaces__msg__CloudSources__Sequence *
gpd_ros2_interfaces__msg__CloudSources__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gpd_ros2_interfaces__msg__CloudSources__Sequence * array = (gpd_ros2_interfaces__msg__CloudSources__Sequence *)allocator.allocate(sizeof(gpd_ros2_interfaces__msg__CloudSources__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = gpd_ros2_interfaces__msg__CloudSources__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
gpd_ros2_interfaces__msg__CloudSources__Sequence__destroy(gpd_ros2_interfaces__msg__CloudSources__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    gpd_ros2_interfaces__msg__CloudSources__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
gpd_ros2_interfaces__msg__CloudSources__Sequence__are_equal(const gpd_ros2_interfaces__msg__CloudSources__Sequence * lhs, const gpd_ros2_interfaces__msg__CloudSources__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!gpd_ros2_interfaces__msg__CloudSources__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
gpd_ros2_interfaces__msg__CloudSources__Sequence__copy(
  const gpd_ros2_interfaces__msg__CloudSources__Sequence * input,
  gpd_ros2_interfaces__msg__CloudSources__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(gpd_ros2_interfaces__msg__CloudSources);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    gpd_ros2_interfaces__msg__CloudSources * data =
      (gpd_ros2_interfaces__msg__CloudSources *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!gpd_ros2_interfaces__msg__CloudSources__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          gpd_ros2_interfaces__msg__CloudSources__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!gpd_ros2_interfaces__msg__CloudSources__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
