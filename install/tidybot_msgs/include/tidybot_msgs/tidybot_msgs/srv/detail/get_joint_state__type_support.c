// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from tidybot_msgs:srv/GetJointState.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "tidybot_msgs/srv/detail/get_joint_state__rosidl_typesupport_introspection_c.h"
#include "tidybot_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "tidybot_msgs/srv/detail/get_joint_state__functions.h"
#include "tidybot_msgs/srv/detail/get_joint_state__struct.h"


// Include directives for member types
// Member `joint_names`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__GetJointState_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tidybot_msgs__srv__GetJointState_Request__init(message_memory);
}

void tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__GetJointState_Request_fini_function(void * message_memory)
{
  tidybot_msgs__srv__GetJointState_Request__fini(message_memory);
}

size_t tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__size_function__GetJointState_Request__joint_names(
  const void * untyped_member)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return member->size;
}

const void * tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__get_const_function__GetJointState_Request__joint_names(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__get_function__GetJointState_Request__joint_names(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__fetch_function__GetJointState_Request__joint_names(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const rosidl_runtime_c__String * item =
    ((const rosidl_runtime_c__String *)
    tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__get_const_function__GetJointState_Request__joint_names(untyped_member, index));
  rosidl_runtime_c__String * value =
    (rosidl_runtime_c__String *)(untyped_value);
  *value = *item;
}

void tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__assign_function__GetJointState_Request__joint_names(
  void * untyped_member, size_t index, const void * untyped_value)
{
  rosidl_runtime_c__String * item =
    ((rosidl_runtime_c__String *)
    tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__get_function__GetJointState_Request__joint_names(untyped_member, index));
  const rosidl_runtime_c__String * value =
    (const rosidl_runtime_c__String *)(untyped_value);
  *item = *value;
}

bool tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__resize_function__GetJointState_Request__joint_names(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  rosidl_runtime_c__String__Sequence__fini(member);
  return rosidl_runtime_c__String__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__GetJointState_Request_message_member_array[1] = {
  {
    "joint_names",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__GetJointState_Request, joint_names),  // bytes offset in struct
    NULL,  // default value
    tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__size_function__GetJointState_Request__joint_names,  // size() function pointer
    tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__get_const_function__GetJointState_Request__joint_names,  // get_const(index) function pointer
    tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__get_function__GetJointState_Request__joint_names,  // get(index) function pointer
    tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__fetch_function__GetJointState_Request__joint_names,  // fetch(index, &value) function pointer
    tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__assign_function__GetJointState_Request__joint_names,  // assign(index, value) function pointer
    tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__resize_function__GetJointState_Request__joint_names  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__GetJointState_Request_message_members = {
  "tidybot_msgs__srv",  // message namespace
  "GetJointState_Request",  // message name
  1,  // number of fields
  sizeof(tidybot_msgs__srv__GetJointState_Request),
  tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__GetJointState_Request_message_member_array,  // message members
  tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__GetJointState_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__GetJointState_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__GetJointState_Request_message_type_support_handle = {
  0,
  &tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__GetJointState_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tidybot_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, srv, GetJointState_Request)() {
  if (!tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__GetJointState_Request_message_type_support_handle.typesupport_identifier) {
    tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__GetJointState_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tidybot_msgs__srv__GetJointState_Request__rosidl_typesupport_introspection_c__GetJointState_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "tidybot_msgs/srv/detail/get_joint_state__rosidl_typesupport_introspection_c.h"
// already included above
// #include "tidybot_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "tidybot_msgs/srv/detail/get_joint_state__functions.h"
// already included above
// #include "tidybot_msgs/srv/detail/get_joint_state__struct.h"


// Include directives for member types
// Member `state`
#include "sensor_msgs/msg/joint_state.h"
// Member `state`
#include "sensor_msgs/msg/detail/joint_state__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tidybot_msgs__srv__GetJointState_Response__rosidl_typesupport_introspection_c__GetJointState_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tidybot_msgs__srv__GetJointState_Response__init(message_memory);
}

void tidybot_msgs__srv__GetJointState_Response__rosidl_typesupport_introspection_c__GetJointState_Response_fini_function(void * message_memory)
{
  tidybot_msgs__srv__GetJointState_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember tidybot_msgs__srv__GetJointState_Response__rosidl_typesupport_introspection_c__GetJointState_Response_message_member_array[1] = {
  {
    "state",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__GetJointState_Response, state),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tidybot_msgs__srv__GetJointState_Response__rosidl_typesupport_introspection_c__GetJointState_Response_message_members = {
  "tidybot_msgs__srv",  // message namespace
  "GetJointState_Response",  // message name
  1,  // number of fields
  sizeof(tidybot_msgs__srv__GetJointState_Response),
  tidybot_msgs__srv__GetJointState_Response__rosidl_typesupport_introspection_c__GetJointState_Response_message_member_array,  // message members
  tidybot_msgs__srv__GetJointState_Response__rosidl_typesupport_introspection_c__GetJointState_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  tidybot_msgs__srv__GetJointState_Response__rosidl_typesupport_introspection_c__GetJointState_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tidybot_msgs__srv__GetJointState_Response__rosidl_typesupport_introspection_c__GetJointState_Response_message_type_support_handle = {
  0,
  &tidybot_msgs__srv__GetJointState_Response__rosidl_typesupport_introspection_c__GetJointState_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tidybot_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, srv, GetJointState_Response)() {
  tidybot_msgs__srv__GetJointState_Response__rosidl_typesupport_introspection_c__GetJointState_Response_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sensor_msgs, msg, JointState)();
  if (!tidybot_msgs__srv__GetJointState_Response__rosidl_typesupport_introspection_c__GetJointState_Response_message_type_support_handle.typesupport_identifier) {
    tidybot_msgs__srv__GetJointState_Response__rosidl_typesupport_introspection_c__GetJointState_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tidybot_msgs__srv__GetJointState_Response__rosidl_typesupport_introspection_c__GetJointState_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "tidybot_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "tidybot_msgs/srv/detail/get_joint_state__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers tidybot_msgs__srv__detail__get_joint_state__rosidl_typesupport_introspection_c__GetJointState_service_members = {
  "tidybot_msgs__srv",  // service namespace
  "GetJointState",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // tidybot_msgs__srv__detail__get_joint_state__rosidl_typesupport_introspection_c__GetJointState_Request_message_type_support_handle,
  NULL  // response message
  // tidybot_msgs__srv__detail__get_joint_state__rosidl_typesupport_introspection_c__GetJointState_Response_message_type_support_handle
};

static rosidl_service_type_support_t tidybot_msgs__srv__detail__get_joint_state__rosidl_typesupport_introspection_c__GetJointState_service_type_support_handle = {
  0,
  &tidybot_msgs__srv__detail__get_joint_state__rosidl_typesupport_introspection_c__GetJointState_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, srv, GetJointState_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, srv, GetJointState_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tidybot_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, srv, GetJointState)() {
  if (!tidybot_msgs__srv__detail__get_joint_state__rosidl_typesupport_introspection_c__GetJointState_service_type_support_handle.typesupport_identifier) {
    tidybot_msgs__srv__detail__get_joint_state__rosidl_typesupport_introspection_c__GetJointState_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)tidybot_msgs__srv__detail__get_joint_state__rosidl_typesupport_introspection_c__GetJointState_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, srv, GetJointState_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, srv, GetJointState_Response)()->data;
  }

  return &tidybot_msgs__srv__detail__get_joint_state__rosidl_typesupport_introspection_c__GetJointState_service_type_support_handle;
}
