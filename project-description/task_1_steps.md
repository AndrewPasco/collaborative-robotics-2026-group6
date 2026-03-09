
**Legend:**  $\star$ = Extra Tasks

## **Task 1: State Machine Overview**

**A. INTERPRET VERBAL COMMAND** 
*   **Implementation:** Speech-to-Text (S2T) + Large Language Model (LLM) API call.

**B. Navigate to Object** 

  **1. SCAN SCENE (SPIN)** 
  *   **Expansion:** [+ EXPLORATION (build global map)   $\star$]

  **2. LOCATE OBJECT (BANANA)** 
  *   **Implementation:** Object detection (YOLO).

  **3. PLAN PATH** 
  *   **Control Logic:** Differential drive to keep Object centered in the frame (using the same detector from Step C).

  **4. NAVIGATE** 
  *   **Condition:** Move until "sufficiently close."

**C. Grabbing Object**
  **1. PLAN GRAB [OBTAIN OBJECT POSE IN LOCAL FRAME]** 
  *   **Method 1 (Easy):** AprilTag.
  *   **Method 2 (Fun):** YOLO + GraspAnything.

  **2. EXECUTE GRAB** 
  *   **Implementation:** MoveIt / or similar motion planning library.

  **3. HOLD AT “SAFE” GRAB POSE** 
  *   **Implementation:** MoveIt / or similar

**D. NAVIGATE TO START** 
*   **Prerequisite:** Mark start location with "something" (AprilTag / Object).
*   **Action:** Repeat steps **B** through **E** to return.