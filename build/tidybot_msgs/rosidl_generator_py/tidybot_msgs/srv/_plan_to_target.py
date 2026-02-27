# generated from rosidl_generator_py/resource/_idl.py.em
# with input from tidybot_msgs:srv/PlanToTarget.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PlanToTarget_Request(type):
    """Metaclass of message 'PlanToTarget_Request'."""

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
            module = import_type_support('tidybot_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'tidybot_msgs.srv.PlanToTarget_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__plan_to_target__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__plan_to_target__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__plan_to_target__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__plan_to_target__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__plan_to_target__request

            from geometry_msgs.msg import Pose
            if Pose.__class__._TYPE_SUPPORT is None:
                Pose.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'USE_ORIENTATION__DEFAULT': True,
            'EXECUTE__DEFAULT': True,
            'DURATION__DEFAULT': 2.0,
            'MAX_CONDITION_NUMBER__DEFAULT': 100.0,
        }

    @property
    def USE_ORIENTATION__DEFAULT(cls):
        """Return default value for message field 'use_orientation'."""
        return True

    @property
    def EXECUTE__DEFAULT(cls):
        """Return default value for message field 'execute'."""
        return True

    @property
    def DURATION__DEFAULT(cls):
        """Return default value for message field 'duration'."""
        return 2.0

    @property
    def MAX_CONDITION_NUMBER__DEFAULT(cls):
        """Return default value for message field 'max_condition_number'."""
        return 100.0


class PlanToTarget_Request(metaclass=Metaclass_PlanToTarget_Request):
    """Message class 'PlanToTarget_Request'."""

    __slots__ = [
        '_arm_name',
        '_target_pose',
        '_use_orientation',
        '_execute',
        '_duration',
        '_max_condition_number',
    ]

    _fields_and_field_types = {
        'arm_name': 'string',
        'target_pose': 'geometry_msgs/Pose',
        'use_orientation': 'boolean',
        'execute': 'boolean',
        'duration': 'double',
        'max_condition_number': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Pose'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.arm_name = kwargs.get('arm_name', str())
        from geometry_msgs.msg import Pose
        self.target_pose = kwargs.get('target_pose', Pose())
        self.use_orientation = kwargs.get(
            'use_orientation', PlanToTarget_Request.USE_ORIENTATION__DEFAULT)
        self.execute = kwargs.get(
            'execute', PlanToTarget_Request.EXECUTE__DEFAULT)
        self.duration = kwargs.get(
            'duration', PlanToTarget_Request.DURATION__DEFAULT)
        self.max_condition_number = kwargs.get(
            'max_condition_number', PlanToTarget_Request.MAX_CONDITION_NUMBER__DEFAULT)

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
        if self.arm_name != other.arm_name:
            return False
        if self.target_pose != other.target_pose:
            return False
        if self.use_orientation != other.use_orientation:
            return False
        if self.execute != other.execute:
            return False
        if self.duration != other.duration:
            return False
        if self.max_condition_number != other.max_condition_number:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def arm_name(self):
        """Message field 'arm_name'."""
        return self._arm_name

    @arm_name.setter
    def arm_name(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'arm_name' field must be of type 'str'"
        self._arm_name = value

    @builtins.property
    def target_pose(self):
        """Message field 'target_pose'."""
        return self._target_pose

    @target_pose.setter
    def target_pose(self, value):
        if __debug__:
            from geometry_msgs.msg import Pose
            assert \
                isinstance(value, Pose), \
                "The 'target_pose' field must be a sub message of type 'Pose'"
        self._target_pose = value

    @builtins.property
    def use_orientation(self):
        """Message field 'use_orientation'."""
        return self._use_orientation

    @use_orientation.setter
    def use_orientation(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'use_orientation' field must be of type 'bool'"
        self._use_orientation = value

    @builtins.property
    def execute(self):
        """Message field 'execute'."""
        return self._execute

    @execute.setter
    def execute(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'execute' field must be of type 'bool'"
        self._execute = value

    @builtins.property
    def duration(self):
        """Message field 'duration'."""
        return self._duration

    @duration.setter
    def duration(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'duration' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'duration' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._duration = value

    @builtins.property
    def max_condition_number(self):
        """Message field 'max_condition_number'."""
        return self._max_condition_number

    @max_condition_number.setter
    def max_condition_number(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'max_condition_number' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'max_condition_number' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._max_condition_number = value


# Import statements for member types

# Member 'joint_positions'
import array  # noqa: E402, I100

# already imported above
# import builtins

# already imported above
# import math

# already imported above
# import rosidl_parser.definition


class Metaclass_PlanToTarget_Response(type):
    """Metaclass of message 'PlanToTarget_Response'."""

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
            module = import_type_support('tidybot_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'tidybot_msgs.srv.PlanToTarget_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__plan_to_target__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__plan_to_target__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__plan_to_target__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__plan_to_target__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__plan_to_target__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class PlanToTarget_Response(metaclass=Metaclass_PlanToTarget_Response):
    """Message class 'PlanToTarget_Response'."""

    __slots__ = [
        '_success',
        '_joint_positions',
        '_executed',
        '_position_error',
        '_orientation_error',
        '_condition_number',
        '_message',
    ]

    _fields_and_field_types = {
        'success': 'boolean',
        'joint_positions': 'sequence<double>',
        'executed': 'boolean',
        'position_error': 'double',
        'orientation_error': 'double',
        'condition_number': 'double',
        'message': 'string',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.success = kwargs.get('success', bool())
        self.joint_positions = array.array('d', kwargs.get('joint_positions', []))
        self.executed = kwargs.get('executed', bool())
        self.position_error = kwargs.get('position_error', float())
        self.orientation_error = kwargs.get('orientation_error', float())
        self.condition_number = kwargs.get('condition_number', float())
        self.message = kwargs.get('message', str())

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
        if self.success != other.success:
            return False
        if self.joint_positions != other.joint_positions:
            return False
        if self.executed != other.executed:
            return False
        if self.position_error != other.position_error:
            return False
        if self.orientation_error != other.orientation_error:
            return False
        if self.condition_number != other.condition_number:
            return False
        if self.message != other.message:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def success(self):
        """Message field 'success'."""
        return self._success

    @success.setter
    def success(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'success' field must be of type 'bool'"
        self._success = value

    @builtins.property
    def joint_positions(self):
        """Message field 'joint_positions'."""
        return self._joint_positions

    @joint_positions.setter
    def joint_positions(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'joint_positions' array.array() must have the type code of 'd'"
            self._joint_positions = value
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'joint_positions' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._joint_positions = array.array('d', value)

    @builtins.property
    def executed(self):
        """Message field 'executed'."""
        return self._executed

    @executed.setter
    def executed(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'executed' field must be of type 'bool'"
        self._executed = value

    @builtins.property
    def position_error(self):
        """Message field 'position_error'."""
        return self._position_error

    @position_error.setter
    def position_error(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'position_error' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'position_error' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._position_error = value

    @builtins.property
    def orientation_error(self):
        """Message field 'orientation_error'."""
        return self._orientation_error

    @orientation_error.setter
    def orientation_error(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'orientation_error' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'orientation_error' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._orientation_error = value

    @builtins.property
    def condition_number(self):
        """Message field 'condition_number'."""
        return self._condition_number

    @condition_number.setter
    def condition_number(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'condition_number' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'condition_number' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._condition_number = value

    @builtins.property
    def message(self):
        """Message field 'message'."""
        return self._message

    @message.setter
    def message(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'message' field must be of type 'str'"
        self._message = value


class Metaclass_PlanToTarget(type):
    """Metaclass of service 'PlanToTarget'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('tidybot_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'tidybot_msgs.srv.PlanToTarget')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__plan_to_target

            from tidybot_msgs.srv import _plan_to_target
            if _plan_to_target.Metaclass_PlanToTarget_Request._TYPE_SUPPORT is None:
                _plan_to_target.Metaclass_PlanToTarget_Request.__import_type_support__()
            if _plan_to_target.Metaclass_PlanToTarget_Response._TYPE_SUPPORT is None:
                _plan_to_target.Metaclass_PlanToTarget_Response.__import_type_support__()


class PlanToTarget(metaclass=Metaclass_PlanToTarget):
    from tidybot_msgs.srv._plan_to_target import PlanToTarget_Request as Request
    from tidybot_msgs.srv._plan_to_target import PlanToTarget_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
