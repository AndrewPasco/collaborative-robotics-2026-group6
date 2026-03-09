**Legend:** $\star$ = Extra Tasks

## **Task 3: State Machine Overview**

### **BOTTLE**

**A. INTERPRET VERBAL COMMAND**
*   **Implementation:** Speech-to-Text (S2T) + Large Language Model (LLM) API call.

**B. Navigate to Bottle**
  **1. SCAN SCENE (SPIN)**
  *   **Expansion:** [+ EXPLORATION (build global map) $\star$]

  **2. LOCATE OBJECT**
  *   **Implementation:** Object detection (YOLO).

  **3. PLAN PATH**
  *   **Control Logic:** Differential drive to keep Object centered in frame, same detector.

  **4. NAVIGATE**
  *   **Condition:** Move until "sufficiently close".

**C. Grabbing Bottle**
  **1. PLAN GRAB [OBTAIN OBJECT POSE IN LOCAL FRAME]**
  *   **Method 1 (Easy):** AprilTag.
  *   **Method 2 (Fun):** YOLO + GraspAnything.

  **2. EXECUTE GRAB**
  *   **Implementation:** MoveIt / similar.

**D. Handling Lid**
  **1. PLAN GRASP FOR LID $\star$**
  *   **Method:** AprilTag/GraspAnything.
  *   **Alternative:** Possible semantic segmentation.

  **2. EXECUTE LID GRASP $\star$**
  *   **Implementation:** MoveIt.

  **3. UNSCREW LID $\star$**
  *   **Implementation:** MoveIt.

  **4. HOLD AT "STOW" GRAB POSE (NO SPILLAGE)**
  *   **Implementation:** MoveIt.

---

### **POURING SEQUENCE**

**E. Navigate to Target Bowl**

  **1. SCAN SCENE (SPIN)**
  *   **Expansion:** [+ EXPLORATION (build global map) $\star$]

  **2. LOCATE OBJECT**
  *   **Implementation:** Object detection (YOLO).

  **3. PLAN PATH**
  *   **Control Logic:** Differential drive to keep Object centered in frame, same detector.

  **4. NAVIGATE**
  *   **Condition:** Move until "sufficiently close".

**F. Execute Pour**

  **1. PLAN POUR START POSE [OBTAIN BOWL POSE IN LOCAL FRAME]**
  *   **Method:** Similar method as 2a.

  **2. EXECUTE POUR**
  *   **Condition 1:** UNTIL LEVEL? $\star$
      *   **Detail:** Main indication: encoder/bottle fiducials.
  *   **Condition 2:** FROM PROMPT? $\star$