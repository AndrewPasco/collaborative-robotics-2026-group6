# ME 326 SEAL Team 6 - Group Chosen Task

For our third task, the Tidybot will transport a “liquid” from one vessel to another, using voice
commands.
**Scope breakdown:**

- Baseline task: after verbal prompt, pick up a known container and pour a liquid (or
    stand-in material, i.e. sand) into a known second container
- Potential extension 1: unscrew a lid off the container before pouring liquid
- Potential extension 2: pour a previously decided amount of liquid into a beaker based on
    language input
**Necessary system components:**
- Perception:
- Container detection, grasp selection: (combination of image recognition model
and traditional image processing, affordance map for handle grasp?)
- Fill level “measurement”: detect level in stationary container which we pour into
(mostly edge detection based?)
- Voice-to-text RNN + text semantics extraction using LLM
- Navigation:
- RRT* / A* / relevant planning algorithm given our map of the space, detected
location of object
- Arm control:
- Motion/trajectory primitives for lid opening
- Position control for “stable” pouring
- Control mode dependent on portion of task (ie velocity control for
following trajectory to grasp pose, position control while pouring, etc)
**Real life applications:**
- Fire fighting (robot pours water on fires)
- Barista (mixing liquids)
- Chemical laboratories: pouring precise amounts of liquid or automating chemical tests
- Assistive Robot: Help elderly with routine tasks.
**Potential pitfalls:**
- Not enough torque available to open a lid
- Two-arm coordination for successful bottle opening / simultaneous gripping
- Measurement error in amount of liquid poured (liquid level detection, pouring alignment)
- Weight of cup/vessel causes slip
- Not enough torque to control to maintain controlled “pour”
