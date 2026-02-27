// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from gpd_ros2_interfaces:srv/DetectGrasps.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "gpd_ros2_interfaces/srv/detail/detect_grasps__rosidl_typesupport_introspection_c.h"
#include "gpd_ros2_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "gpd_ros2_interfaces/srv/detail/detect_grasps__functions.h"
#include "gpd_ros2_interfaces/srv/detail/detect_grasps__struct.h"


// Include directives for member types
// Member `cloud_indexed`
#include "gpd_ros2_interfaces/msg/cloud_indexed.h"
// Member `cloud_indexed`
#include "gpd_ros2_interfaces/msg/detail/cloud_indexed__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void gpd_ros2_interfaces__srv__DetectGrasps_Request__rosidl_typesupport_introspection_c__DetectGrasps_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  gpd_ros2_interfaces__srv__DetectGrasps_Request__init(message_memory);
}

void gpd_ros2_interfaces__srv__DetectGrasps_Request__rosidl_typesupport_introspection_c__DetectGrasps_Request_fini_function(void * message_memory)
{
  gpd_ros2_interfaces__srv__DetectGrasps_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember gpd_ros2_interfaces__srv__DetectGrasps_Request__rosidl_typesupport_introspection_c__DetectGrasps_Request_message_member_array[1] = {
  {
    "cloud_indexed",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gpd_ros2_interfaces__srv__DetectGrasps_Request, cloud_indexed),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers gpd_ros2_interfaces__srv__DetectGrasps_Request__rosidl_typesupport_introspection_c__DetectGrasps_Request_message_members = {
  "gpd_ros2_interfaces__srv",  // message namespace
  "DetectGrasps_Request",  // message name
  1,  // number of fields
  sizeof(gpd_ros2_interfaces__srv__DetectGrasps_Request),
  gpd_ros2_interfaces__srv__DetectGrasps_Request__rosidl_typesupport_introspection_c__DetectGrasps_Request_message_member_array,  // message members
  gpd_ros2_interfaces__srv__DetectGrasps_Request__rosidl_typesupport_introspection_c__DetectGrasps_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  gpd_ros2_interfaces__srv__DetectGrasps_Request__rosidl_typesupport_introspection_c__DetectGrasps_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t gpd_ros2_interfaces__srv__DetectGrasps_Request__rosidl_typesupport_introspection_c__DetectGrasps_Request_message_type_support_handle = {
  0,
  &gpd_ros2_interfaces__srv__DetectGrasps_Request__rosidl_typesupport_introspection_c__DetectGrasps_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_gpd_ros2_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gpd_ros2_interfaces, srv, DetectGrasps_Request)() {
  gpd_ros2_interfaces__srv__DetectGrasps_Request__rosidl_typesupport_introspection_c__DetectGrasps_Request_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gpd_ros2_interfaces, msg, CloudIndexed)();
  if (!gpd_ros2_interfaces__srv__DetectGrasps_Request__rosidl_typesupport_introspection_c__DetectGrasps_Request_message_type_support_handle.typesupport_identifier) {
    gpd_ros2_interfaces__srv__DetectGrasps_Request__rosidl_typesupport_introspection_c__DetectGrasps_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &gpd_ros2_interfaces__srv__DetectGrasps_Request__rosidl_typesupport_introspection_c__DetectGrasps_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "gpd_ros2_interfaces/srv/detail/detect_grasps__rosidl_typesupport_introspection_c.h"
// already included above
// #include "gpd_ros2_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "gpd_ros2_interfaces/srv/detail/detect_grasps__functions.h"
// already included above
// #include "gpd_ros2_interfaces/srv/detail/detect_grasps__struct.h"


// Include directives for member types
// Member `grasp_configs`
#include "gpd_ros2_interfaces/msg/grasp_config_list.h"
// Member `grasp_configs`
#include "gpd_ros2_interfaces/msg/detail/grasp_config_list__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void gpd_ros2_interfaces__srv__DetectGrasps_Response__rosidl_typesupport_introspection_c__DetectGrasps_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  gpd_ros2_interfaces__srv__DetectGrasps_Response__init(message_memory);
}

void gpd_ros2_interfaces__srv__DetectGrasps_Response__rosidl_typesupport_introspection_c__DetectGrasps_Response_fini_function(void * message_memory)
{
  gpd_ros2_interfaces__srv__DetectGrasps_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember gpd_ros2_interfaces__srv__DetectGrasps_Response__rosidl_typesupport_introspection_c__DetectGrasps_Response_message_member_array[1] = {
  {
    "grasp_configs",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gpd_ros2_interfaces__srv__DetectGrasps_Response, grasp_configs),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers gpd_ros2_interfaces__srv__DetectGrasps_Response__rosidl_typesupport_introspection_c__DetectGrasps_Response_message_members = {
  "gpd_ros2_interfaces__srv",  // message namespace
  "DetectGrasps_Response",  // message name
  1,  // number of fields
  sizeof(gpd_ros2_interfaces__srv__DetectGrasps_Response),
  gpd_ros2_interfaces__srv__DetectGrasps_Response__rosidl_typesupport_introspection_c__DetectGrasps_Response_message_member_array,  // message members
  gpd_ros2_interfaces__srv__DetectGrasps_Response__rosidl_typesupport_introspection_c__DetectGrasps_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  gpd_ros2_interfaces__srv__DetectGrasps_Response__rosidl_typesupport_introspection_c__DetectGrasps_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t gpd_ros2_interfaces__srv__DetectGrasps_Response__rosidl_typesupport_introspection_c__DetectGrasps_Response_message_type_support_handle = {
  0,
  &gpd_ros2_interfaces__srv__DetectGrasps_Response__rosidl_typesupport_introspection_c__DetectGrasps_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_gpd_ros2_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gpd_ros2_interfaces, srv, DetectGrasps_Response)() {
  gpd_ros2_interfaces__srv__DetectGrasps_Response__rosidl_typesupport_introspection_c__DetectGrasps_Response_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gpd_ros2_interfaces, msg, GraspConfigList)();
  if (!gpd_ros2_interfaces__srv__DetectGrasps_Response__rosidl_typesupport_introspection_c__DetectGrasps_Response_message_type_support_handle.typesupport_identifier) {
    gpd_ros2_interfaces__srv__DetectGrasps_Response__rosidl_typesupport_introspection_c__DetectGrasps_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &gpd_ros2_interfaces__srv__DetectGrasps_Response__rosidl_typesupport_introspection_c__DetectGrasps_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "gpd_ros2_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "gpd_ros2_interfaces/srv/detail/detect_grasps__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers gpd_ros2_interfaces__srv__detail__detect_grasps__rosidl_typesupport_introspection_c__DetectGrasps_service_members = {
  "gpd_ros2_interfaces__srv",  // service namespace
  "DetectGrasps",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // gpd_ros2_interfaces__srv__detail__detect_grasps__rosidl_typesupport_introspection_c__DetectGrasps_Request_message_type_support_handle,
  NULL  // response message
  // gpd_ros2_interfaces__srv__detail__detect_grasps__rosidl_typesupport_introspection_c__DetectGrasps_Response_message_type_support_handle
};

static rosidl_service_type_support_t gpd_ros2_interfaces__srv__detail__detect_grasps__rosidl_typesupport_introspection_c__DetectGrasps_service_type_support_handle = {
  0,
  &gpd_ros2_interfaces__srv__detail__detect_grasps__rosidl_typesupport_introspection_c__DetectGrasps_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gpd_ros2_interfaces, srv, DetectGrasps_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gpd_ros2_interfaces, srv, DetectGrasps_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_gpd_ros2_interfaces
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gpd_ros2_interfaces, srv, DetectGrasps)() {
  if (!gpd_ros2_interfaces__srv__detail__detect_grasps__rosidl_typesupport_introspection_c__DetectGrasps_service_type_support_handle.typesupport_identifier) {
    gpd_ros2_interfaces__srv__detail__detect_grasps__rosidl_typesupport_introspection_c__DetectGrasps_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)gpd_ros2_interfaces__srv__detail__detect_grasps__rosidl_typesupport_introspection_c__DetectGrasps_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gpd_ros2_interfaces, srv, DetectGrasps_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gpd_ros2_interfaces, srv, DetectGrasps_Response)()->data;
  }

  return &gpd_ros2_interfaces__srv__detail__detect_grasps__rosidl_typesupport_introspection_c__DetectGrasps_service_type_support_handle;
}
