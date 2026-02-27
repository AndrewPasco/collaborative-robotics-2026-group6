// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from gpd_ros2_interfaces:srv/DetectGrasps.idl
// generated code does not contain a copyright notice
#include "gpd_ros2_interfaces/srv/detail/detect_grasps__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "gpd_ros2_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "gpd_ros2_interfaces/srv/detail/detect_grasps__struct.h"
#include "gpd_ros2_interfaces/srv/detail/detect_grasps__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "gpd_ros2_interfaces/msg/detail/cloud_indexed__functions.h"  // cloud_indexed

// forward declare type support functions
size_t get_serialized_size_gpd_ros2_interfaces__msg__CloudIndexed(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_gpd_ros2_interfaces__msg__CloudIndexed(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, gpd_ros2_interfaces, msg, CloudIndexed)();


using _DetectGrasps_Request__ros_msg_type = gpd_ros2_interfaces__srv__DetectGrasps_Request;

static bool _DetectGrasps_Request__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _DetectGrasps_Request__ros_msg_type * ros_message = static_cast<const _DetectGrasps_Request__ros_msg_type *>(untyped_ros_message);
  // Field name: cloud_indexed
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, gpd_ros2_interfaces, msg, CloudIndexed
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->cloud_indexed, cdr))
    {
      return false;
    }
  }

  return true;
}

static bool _DetectGrasps_Request__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _DetectGrasps_Request__ros_msg_type * ros_message = static_cast<_DetectGrasps_Request__ros_msg_type *>(untyped_ros_message);
  // Field name: cloud_indexed
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, gpd_ros2_interfaces, msg, CloudIndexed
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->cloud_indexed))
    {
      return false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_gpd_ros2_interfaces
size_t get_serialized_size_gpd_ros2_interfaces__srv__DetectGrasps_Request(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _DetectGrasps_Request__ros_msg_type * ros_message = static_cast<const _DetectGrasps_Request__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name cloud_indexed

  current_alignment += get_serialized_size_gpd_ros2_interfaces__msg__CloudIndexed(
    &(ros_message->cloud_indexed), current_alignment);

  return current_alignment - initial_alignment;
}

static uint32_t _DetectGrasps_Request__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_gpd_ros2_interfaces__srv__DetectGrasps_Request(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_gpd_ros2_interfaces
size_t max_serialized_size_gpd_ros2_interfaces__srv__DetectGrasps_Request(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // member: cloud_indexed
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_gpd_ros2_interfaces__msg__CloudIndexed(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = gpd_ros2_interfaces__srv__DetectGrasps_Request;
    is_plain =
      (
      offsetof(DataType, cloud_indexed) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _DetectGrasps_Request__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_gpd_ros2_interfaces__srv__DetectGrasps_Request(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_DetectGrasps_Request = {
  "gpd_ros2_interfaces::srv",
  "DetectGrasps_Request",
  _DetectGrasps_Request__cdr_serialize,
  _DetectGrasps_Request__cdr_deserialize,
  _DetectGrasps_Request__get_serialized_size,
  _DetectGrasps_Request__max_serialized_size
};

static rosidl_message_type_support_t _DetectGrasps_Request__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_DetectGrasps_Request,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, gpd_ros2_interfaces, srv, DetectGrasps_Request)() {
  return &_DetectGrasps_Request__type_support;
}

#if defined(__cplusplus)
}
#endif

// already included above
// #include <cassert>
// already included above
// #include <limits>
// already included above
// #include <string>
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
// already included above
// #include "gpd_ros2_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
// already included above
// #include "gpd_ros2_interfaces/srv/detail/detect_grasps__struct.h"
// already included above
// #include "gpd_ros2_interfaces/srv/detail/detect_grasps__functions.h"
// already included above
// #include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "gpd_ros2_interfaces/msg/detail/grasp_config_list__functions.h"  // grasp_configs

// forward declare type support functions
size_t get_serialized_size_gpd_ros2_interfaces__msg__GraspConfigList(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_gpd_ros2_interfaces__msg__GraspConfigList(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, gpd_ros2_interfaces, msg, GraspConfigList)();


using _DetectGrasps_Response__ros_msg_type = gpd_ros2_interfaces__srv__DetectGrasps_Response;

static bool _DetectGrasps_Response__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _DetectGrasps_Response__ros_msg_type * ros_message = static_cast<const _DetectGrasps_Response__ros_msg_type *>(untyped_ros_message);
  // Field name: grasp_configs
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, gpd_ros2_interfaces, msg, GraspConfigList
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->grasp_configs, cdr))
    {
      return false;
    }
  }

  return true;
}

static bool _DetectGrasps_Response__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _DetectGrasps_Response__ros_msg_type * ros_message = static_cast<_DetectGrasps_Response__ros_msg_type *>(untyped_ros_message);
  // Field name: grasp_configs
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, gpd_ros2_interfaces, msg, GraspConfigList
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->grasp_configs))
    {
      return false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_gpd_ros2_interfaces
size_t get_serialized_size_gpd_ros2_interfaces__srv__DetectGrasps_Response(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _DetectGrasps_Response__ros_msg_type * ros_message = static_cast<const _DetectGrasps_Response__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name grasp_configs

  current_alignment += get_serialized_size_gpd_ros2_interfaces__msg__GraspConfigList(
    &(ros_message->grasp_configs), current_alignment);

  return current_alignment - initial_alignment;
}

static uint32_t _DetectGrasps_Response__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_gpd_ros2_interfaces__srv__DetectGrasps_Response(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_gpd_ros2_interfaces
size_t max_serialized_size_gpd_ros2_interfaces__srv__DetectGrasps_Response(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // member: grasp_configs
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_gpd_ros2_interfaces__msg__GraspConfigList(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = gpd_ros2_interfaces__srv__DetectGrasps_Response;
    is_plain =
      (
      offsetof(DataType, grasp_configs) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _DetectGrasps_Response__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_gpd_ros2_interfaces__srv__DetectGrasps_Response(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_DetectGrasps_Response = {
  "gpd_ros2_interfaces::srv",
  "DetectGrasps_Response",
  _DetectGrasps_Response__cdr_serialize,
  _DetectGrasps_Response__cdr_deserialize,
  _DetectGrasps_Response__get_serialized_size,
  _DetectGrasps_Response__max_serialized_size
};

static rosidl_message_type_support_t _DetectGrasps_Response__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_DetectGrasps_Response,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, gpd_ros2_interfaces, srv, DetectGrasps_Response)() {
  return &_DetectGrasps_Response__type_support;
}

#if defined(__cplusplus)
}
#endif

#include "rosidl_typesupport_fastrtps_cpp/service_type_support.h"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "gpd_ros2_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "gpd_ros2_interfaces/srv/detect_grasps.h"

#if defined(__cplusplus)
extern "C"
{
#endif

static service_type_support_callbacks_t DetectGrasps__callbacks = {
  "gpd_ros2_interfaces::srv",
  "DetectGrasps",
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, gpd_ros2_interfaces, srv, DetectGrasps_Request)(),
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, gpd_ros2_interfaces, srv, DetectGrasps_Response)(),
};

static rosidl_service_type_support_t DetectGrasps__handle = {
  rosidl_typesupport_fastrtps_c__identifier,
  &DetectGrasps__callbacks,
  get_service_typesupport_handle_function,
};

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, gpd_ros2_interfaces, srv, DetectGrasps)() {
  return &DetectGrasps__handle;
}

#if defined(__cplusplus)
}
#endif
