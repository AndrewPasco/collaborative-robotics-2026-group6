// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tidybot_msgs:srv/AudioRecord.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__AUDIO_RECORD__STRUCT_H_
#define TIDYBOT_MSGS__SRV__DETAIL__AUDIO_RECORD__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/AudioRecord in the package tidybot_msgs.
typedef struct tidybot_msgs__srv__AudioRecord_Request
{
  /// true = start recording, false = stop recording and return audio
  bool start;
} tidybot_msgs__srv__AudioRecord_Request;

// Struct for a sequence of tidybot_msgs__srv__AudioRecord_Request.
typedef struct tidybot_msgs__srv__AudioRecord_Request__Sequence
{
  tidybot_msgs__srv__AudioRecord_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tidybot_msgs__srv__AudioRecord_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'message'
#include "rosidl_runtime_c/string.h"
// Member 'audio_data'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in srv/AudioRecord in the package tidybot_msgs.
typedef struct tidybot_msgs__srv__AudioRecord_Response
{
  /// Whether the operation succeeded
  bool success;
  /// Status message
  rosidl_runtime_c__String message;
  /// Recorded audio samples (PCM float32, mono)
  /// Only populated when start=false (stop)
  rosidl_runtime_c__float__Sequence audio_data;
  /// Sample rate in Hz (e.g., 16000)
  int32_t sample_rate;
  /// Duration of recording in seconds
  double duration;
} tidybot_msgs__srv__AudioRecord_Response;

// Struct for a sequence of tidybot_msgs__srv__AudioRecord_Response.
typedef struct tidybot_msgs__srv__AudioRecord_Response__Sequence
{
  tidybot_msgs__srv__AudioRecord_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tidybot_msgs__srv__AudioRecord_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TIDYBOT_MSGS__SRV__DETAIL__AUDIO_RECORD__STRUCT_H_
