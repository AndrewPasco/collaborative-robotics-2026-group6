# generated from rosidl_generator_py/resource/_idl.py.em
# with input from gpd_ros2_interfaces:srv/DetectGrasps.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_DetectGrasps_Request(type):
    """Metaclass of message 'DetectGrasps_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('gpd_ros2_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'gpd_ros2_interfaces.srv.DetectGrasps_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__detect_grasps__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__detect_grasps__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__detect_grasps__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__detect_grasps__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__detect_grasps__request

            from gpd_ros2_interfaces.msg import CloudIndexed
            if CloudIndexed.__class__._TYPE_SUPPORT is None:
                CloudIndexed.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class DetectGrasps_Request(metaclass=Metaclass_DetectGrasps_Request):
    """Message class 'DetectGrasps_Request'."""

    __slots__ = [
        '_cloud_indexed',
    ]

    _fields_and_field_types = {
        'cloud_indexed': 'gpd_ros2_interfaces/CloudIndexed',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['gpd_ros2_interfaces', 'msg'], 'CloudIndexed'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from gpd_ros2_interfaces.msg import CloudIndexed
        self.cloud_indexed = kwargs.get('cloud_indexed', CloudIndexed())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.cloud_indexed != other.cloud_indexed:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def cloud_indexed(self):
        """Message field 'cloud_indexed'."""
        return self._cloud_indexed

    @cloud_indexed.setter
    def cloud_indexed(self, value):
        if __debug__:
            from gpd_ros2_interfaces.msg import CloudIndexed
            assert \
                isinstance(value, CloudIndexed), \
                "The 'cloud_indexed' field must be a sub message of type 'CloudIndexed'"
        self._cloud_indexed = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_DetectGrasps_Response(type):
    """Metaclass of message 'DetectGrasps_Response'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('gpd_ros2_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'gpd_ros2_interfaces.srv.DetectGrasps_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__detect_grasps__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__detect_grasps__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__detect_grasps__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__detect_grasps__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__detect_grasps__response

            from gpd_ros2_interfaces.msg import GraspConfigList
            if GraspConfigList.__class__._TYPE_SUPPORT is None:
                GraspConfigList.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class DetectGrasps_Response(metaclass=Metaclass_DetectGrasps_Response):
    """Message class 'DetectGrasps_Response'."""

    __slots__ = [
        '_grasp_configs',
    ]

    _fields_and_field_types = {
        'grasp_configs': 'gpd_ros2_interfaces/GraspConfigList',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['gpd_ros2_interfaces', 'msg'], 'GraspConfigList'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from gpd_ros2_interfaces.msg import GraspConfigList
        self.grasp_configs = kwargs.get('grasp_configs', GraspConfigList())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.grasp_configs != other.grasp_configs:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def grasp_configs(self):
        """Message field 'grasp_configs'."""
        return self._grasp_configs

    @grasp_configs.setter
    def grasp_configs(self, value):
        if __debug__:
            from gpd_ros2_interfaces.msg import GraspConfigList
            assert \
                isinstance(value, GraspConfigList), \
                "The 'grasp_configs' field must be a sub message of type 'GraspConfigList'"
        self._grasp_configs = value


class Metaclass_DetectGrasps(type):
    """Metaclass of service 'DetectGrasps'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('gpd_ros2_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'gpd_ros2_interfaces.srv.DetectGrasps')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__detect_grasps

            from gpd_ros2_interfaces.srv import _detect_grasps
            if _detect_grasps.Metaclass_DetectGrasps_Request._TYPE_SUPPORT is None:
                _detect_grasps.Metaclass_DetectGrasps_Request.__import_type_support__()
            if _detect_grasps.Metaclass_DetectGrasps_Response._TYPE_SUPPORT is None:
                _detect_grasps.Metaclass_DetectGrasps_Response.__import_type_support__()


class DetectGrasps(metaclass=Metaclass_DetectGrasps):
    from gpd_ros2_interfaces.srv._detect_grasps import DetectGrasps_Request as Request
    from gpd_ros2_interfaces.srv._detect_grasps import DetectGrasps_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
