// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from gpd_ros2_interfaces:srv/DetectGrasps.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "gpd_ros2_interfaces/srv/detail/detect_grasps__struct.h"
#include "gpd_ros2_interfaces/srv/detail/detect_grasps__functions.h"

bool gpd_ros2_interfaces__msg__cloud_indexed__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * gpd_ros2_interfaces__msg__cloud_indexed__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool gpd_ros2_interfaces__srv__detect_grasps__request__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[60];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("gpd_ros2_interfaces.srv._detect_grasps.DetectGrasps_Request", full_classname_dest, 59) == 0);
  }
  gpd_ros2_interfaces__srv__DetectGrasps_Request * ros_message = _ros_message;
  {  // cloud_indexed
    PyObject * field = PyObject_GetAttrString(_pymsg, "cloud_indexed");
    if (!field) {
      return false;
    }
    if (!gpd_ros2_interfaces__msg__cloud_indexed__convert_from_py(field, &ros_message->cloud_indexed)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * gpd_ros2_interfaces__srv__detect_grasps__request__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of DetectGrasps_Request */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("gpd_ros2_interfaces.srv._detect_grasps");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "DetectGrasps_Request");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  gpd_ros2_interfaces__srv__DetectGrasps_Request * ros_message = (gpd_ros2_interfaces__srv__DetectGrasps_Request *)raw_ros_message;
  {  // cloud_indexed
    PyObject * field = NULL;
    field = gpd_ros2_interfaces__msg__cloud_indexed__convert_to_py(&ros_message->cloud_indexed);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "cloud_indexed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// already included above
// #include <Python.h>
// already included above
// #include <stdbool.h>
// already included above
// #include "numpy/ndarrayobject.h"
// already included above
// #include "rosidl_runtime_c/visibility_control.h"
// already included above
// #include "gpd_ros2_interfaces/srv/detail/detect_grasps__struct.h"
// already included above
// #include "gpd_ros2_interfaces/srv/detail/detect_grasps__functions.h"

bool gpd_ros2_interfaces__msg__grasp_config_list__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * gpd_ros2_interfaces__msg__grasp_config_list__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool gpd_ros2_interfaces__srv__detect_grasps__response__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[61];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("gpd_ros2_interfaces.srv._detect_grasps.DetectGrasps_Response", full_classname_dest, 60) == 0);
  }
  gpd_ros2_interfaces__srv__DetectGrasps_Response * ros_message = _ros_message;
  {  // grasp_configs
    PyObject * field = PyObject_GetAttrString(_pymsg, "grasp_configs");
    if (!field) {
      return false;
    }
    if (!gpd_ros2_interfaces__msg__grasp_config_list__convert_from_py(field, &ros_message->grasp_configs)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * gpd_ros2_interfaces__srv__detect_grasps__response__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of DetectGrasps_Response */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("gpd_ros2_interfaces.srv._detect_grasps");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "DetectGrasps_Response");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  gpd_ros2_interfaces__srv__DetectGrasps_Response * ros_message = (gpd_ros2_interfaces__srv__DetectGrasps_Response *)raw_ros_message;
  {  // grasp_configs
    PyObject * field = NULL;
    field = gpd_ros2_interfaces__msg__grasp_config_list__convert_to_py(&ros_message->grasp_configs);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "grasp_configs", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
