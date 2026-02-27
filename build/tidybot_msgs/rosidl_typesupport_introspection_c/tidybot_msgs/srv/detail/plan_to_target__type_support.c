// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from tidybot_msgs:srv/PlanToTarget.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "tidybot_msgs/srv/detail/plan_to_target__rosidl_typesupport_introspection_c.h"
#include "tidybot_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "tidybot_msgs/srv/detail/plan_to_target__functions.h"
#include "tidybot_msgs/srv/detail/plan_to_target__struct.h"


// Include directives for member types
// Member `arm_name`
#include "rosidl_runtime_c/string_functions.h"
// Member `target_pose`
#include "geometry_msgs/msg/pose.h"
// Member `target_pose`
#include "geometry_msgs/msg/detail/pose__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tidybot_msgs__srv__PlanToTarget_Request__rosidl_typesupport_introspection_c__PlanToTarget_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tidybot_msgs__srv__PlanToTarget_Request__init(message_memory);
}

void tidybot_msgs__srv__PlanToTarget_Request__rosidl_typesupport_introspection_c__PlanToTarget_Request_fini_function(void * message_memory)
{
  tidybot_msgs__srv__PlanToTarget_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember tidybot_msgs__srv__PlanToTarget_Request__rosidl_typesupport_introspection_c__PlanToTarget_Request_message_member_array[6] = {
  {
    "arm_name",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__PlanToTarget_Request, arm_name),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_pose",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__PlanToTarget_Request, target_pose),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "use_orientation",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__PlanToTarget_Request, use_orientation),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "execute",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__PlanToTarget_Request, execute),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "duration",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__PlanToTarget_Request, duration),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "max_condition_number",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__PlanToTarget_Request, max_condition_number),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tidybot_msgs__srv__PlanToTarget_Request__rosidl_typesupport_introspection_c__PlanToTarget_Request_message_members = {
  "tidybot_msgs__srv",  // message namespace
  "PlanToTarget_Request",  // message name
  6,  // number of fields
  sizeof(tidybot_msgs__srv__PlanToTarget_Request),
  tidybot_msgs__srv__PlanToTarget_Request__rosidl_typesupport_introspection_c__PlanToTarget_Request_message_member_array,  // message members
  tidybot_msgs__srv__PlanToTarget_Request__rosidl_typesupport_introspection_c__PlanToTarget_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  tidybot_msgs__srv__PlanToTarget_Request__rosidl_typesupport_introspection_c__PlanToTarget_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tidybot_msgs__srv__PlanToTarget_Request__rosidl_typesupport_introspection_c__PlanToTarget_Request_message_type_support_handle = {
  0,
  &tidybot_msgs__srv__PlanToTarget_Request__rosidl_typesupport_introspection_c__PlanToTarget_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tidybot_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, srv, PlanToTarget_Request)() {
  tidybot_msgs__srv__PlanToTarget_Request__rosidl_typesupport_introspection_c__PlanToTarget_Request_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Pose)();
  if (!tidybot_msgs__srv__PlanToTarget_Request__rosidl_typesupport_introspection_c__PlanToTarget_Request_message_type_support_handle.typesupport_identifier) {
    tidybot_msgs__srv__PlanToTarget_Request__rosidl_typesupport_introspection_c__PlanToTarget_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tidybot_msgs__srv__PlanToTarget_Request__rosidl_typesupport_introspection_c__PlanToTarget_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "tidybot_msgs/srv/detail/plan_to_target__rosidl_typesupport_introspection_c.h"
// already included above
// #include "tidybot_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "tidybot_msgs/srv/detail/plan_to_target__functions.h"
// already included above
// #include "tidybot_msgs/srv/detail/plan_to_target__struct.h"


// Include directives for member types
// Member `joint_positions`
#include "rosidl_runtime_c/primitives_sequence_functions.h"
// Member `message`
// already included above
// #include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__PlanToTarget_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tidybot_msgs__srv__PlanToTarget_Response__init(message_memory);
}

void tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__PlanToTarget_Response_fini_function(void * message_memory)
{
  tidybot_msgs__srv__PlanToTarget_Response__fini(message_memory);
}

size_t tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__size_function__PlanToTarget_Response__joint_positions(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__get_const_function__PlanToTarget_Response__joint_positions(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__get_function__PlanToTarget_Response__joint_positions(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__fetch_function__PlanToTarget_Response__joint_positions(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__get_const_function__PlanToTarget_Response__joint_positions(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__assign_function__PlanToTarget_Response__joint_positions(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__get_function__PlanToTarget_Response__joint_positions(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__resize_function__PlanToTarget_Response__joint_positions(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__PlanToTarget_Response_message_member_array[7] = {
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__PlanToTarget_Response, success),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "joint_positions",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__PlanToTarget_Response, joint_positions),  // bytes offset in struct
    NULL,  // default value
    tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__size_function__PlanToTarget_Response__joint_positions,  // size() function pointer
    tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__get_const_function__PlanToTarget_Response__joint_positions,  // get_const(index) function pointer
    tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__get_function__PlanToTarget_Response__joint_positions,  // get(index) function pointer
    tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__fetch_function__PlanToTarget_Response__joint_positions,  // fetch(index, &value) function pointer
    tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__assign_function__PlanToTarget_Response__joint_positions,  // assign(index, value) function pointer
    tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__resize_function__PlanToTarget_Response__joint_positions  // resize(index) function pointer
  },
  {
    "executed",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__PlanToTarget_Response, executed),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "position_error",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__PlanToTarget_Response, position_error),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "orientation_error",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__PlanToTarget_Response, orientation_error),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "condition_number",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__PlanToTarget_Response, condition_number),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "message",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__srv__PlanToTarget_Response, message),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__PlanToTarget_Response_message_members = {
  "tidybot_msgs__srv",  // message namespace
  "PlanToTarget_Response",  // message name
  7,  // number of fields
  sizeof(tidybot_msgs__srv__PlanToTarget_Response),
  tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__PlanToTarget_Response_message_member_array,  // message members
  tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__PlanToTarget_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__PlanToTarget_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__PlanToTarget_Response_message_type_support_handle = {
  0,
  &tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__PlanToTarget_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tidybot_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, srv, PlanToTarget_Response)() {
  if (!tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__PlanToTarget_Response_message_type_support_handle.typesupport_identifier) {
    tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__PlanToTarget_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tidybot_msgs__srv__PlanToTarget_Response__rosidl_typesupport_introspection_c__PlanToTarget_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "tidybot_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "tidybot_msgs/srv/detail/plan_to_target__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers tidybot_msgs__srv__detail__plan_to_target__rosidl_typesupport_introspection_c__PlanToTarget_service_members = {
  "tidybot_msgs__srv",  // service namespace
  "PlanToTarget",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // tidybot_msgs__srv__detail__plan_to_target__rosidl_typesupport_introspection_c__PlanToTarget_Request_message_type_support_handle,
  NULL  // response message
  // tidybot_msgs__srv__detail__plan_to_target__rosidl_typesupport_introspection_c__PlanToTarget_Response_message_type_support_handle
};

static rosidl_service_type_support_t tidybot_msgs__srv__detail__plan_to_target__rosidl_typesupport_introspection_c__PlanToTarget_service_type_support_handle = {
  0,
  &tidybot_msgs__srv__detail__plan_to_target__rosidl_typesupport_introspection_c__PlanToTarget_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, srv, PlanToTarget_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, srv, PlanToTarget_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tidybot_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, srv, PlanToTarget)() {
  if (!tidybot_msgs__srv__detail__plan_to_target__rosidl_typesupport_introspection_c__PlanToTarget_service_type_support_handle.typesupport_identifier) {
    tidybot_msgs__srv__detail__plan_to_target__rosidl_typesupport_introspection_c__PlanToTarget_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)tidybot_msgs__srv__detail__plan_to_target__rosidl_typesupport_introspection_c__PlanToTarget_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, srv, PlanToTarget_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, srv, PlanToTarget_Response)()->data;
  }

  return &tidybot_msgs__srv__detail__plan_to_target__rosidl_typesupport_introspection_c__PlanToTarget_service_type_support_handle;
}
