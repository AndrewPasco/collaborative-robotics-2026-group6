# generated from rosidl_generator_py/resource/_idl.py.em
# with input from gpd_ros2_interfaces:msg/CloudIndexed.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'indices'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_CloudIndexed(type):
    """Metaclass of message 'CloudIndexed'."""

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
                'gpd_ros2_interfaces.msg.CloudIndexed')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__cloud_indexed
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__cloud_indexed
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__cloud_indexed
            cls._TYPE_SUPPORT = module.type_support_msg__msg__cloud_indexed
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__cloud_indexed

            from gpd_ros2_interfaces.msg import CloudSources
            if CloudSources.__class__._TYPE_SUPPORT is None:
                CloudSources.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CloudIndexed(metaclass=Metaclass_CloudIndexed):
    """Message class 'CloudIndexed'."""

    __slots__ = [
        '_cloud_sources',
        '_indices',
    ]

    _fields_and_field_types = {
        'cloud_sources': 'gpd_ros2_interfaces/CloudSources',
        'indices': 'sequence<int64>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['gpd_ros2_interfaces', 'msg'], 'CloudSources'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('int64')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from gpd_ros2_interfaces.msg import CloudSources
        self.cloud_sources = kwargs.get('cloud_sources', CloudSources())
        self.indices = array.array('q', kwargs.get('indices', []))

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
        if self.cloud_sources != other.cloud_sources:
            return False
        if self.indices != other.indices:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def cloud_sources(self):
        """Message field 'cloud_sources'."""
        return self._cloud_sources

    @cloud_sources.setter
    def cloud_sources(self, value):
        if __debug__:
            from gpd_ros2_interfaces.msg import CloudSources
            assert \
                isinstance(value, CloudSources), \
                "The 'cloud_sources' field must be a sub message of type 'CloudSources'"
        self._cloud_sources = value

    @builtins.property
    def indices(self):
        """Message field 'indices'."""
        return self._indices

    @indices.setter
    def indices(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'q', \
                "The 'indices' array.array() must have the type code of 'q'"
            self._indices = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, int) for v in value) and
                 all(val >= -9223372036854775808 and val < 9223372036854775808 for val in value)), \
                "The 'indices' field must be a set or sequence and each value of type 'int' and each integer in [-9223372036854775808, 9223372036854775807]"
        self._indices = array.array('q', value)
