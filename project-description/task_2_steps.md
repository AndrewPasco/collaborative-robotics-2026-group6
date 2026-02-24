**Legend:** $\star$ = Extra Tasks

## **Task 2: State Machine Overview**

### **PAYLOAD**

**A. INTERPRET VERBAL COMMAND**
*   **Implementation:** Speech-to-Text (S2T) + Large Language Model (LLM) API call.

**B. Navigate to Payload Object**
  **1. SCAN SCENE (SPIN)**
  *   **Expansion:** [+ EXPLORATION (build global map) $\star$]
  *   **Note:**  Optionality to store target as well if encountered.

  **2. LOCATE OBJECT (BANANA)**
  *   **Implementation:** Object detection (YOLO).

  **3. PLAN PATH**
  *   **Control Logic:** Differential drive to keep Object centered in frame, same detector.

  **4. NAVIGATE**
  *   **Condition:** Move until "sufficiently close".

**C. Grabbing Object**
  **1. PLAN GRAB [OBTAIN OBJECT POSE IN LOCAL FRAME]**
  *   **Method 1 (Easy):** AprilTag.
  *   **Method 2 (Fun):** YOLO + GraspAnything.

  **2. EXECUTE GRAB**
  *   **Implementation:** MoveIt / or similar.

  **3. HOLD AT "SAFE" GRAB POSE**
  *   **Implementation:** MoveIt / or similar.

---

### **BIN**

**D. Navigate to Bin**

  **1. SCAN SCENE**

  **2. LOCATE OBJECT (BIN)**
  *   **Expansion:** [+ EXPLORATION (build global map) $\star$]

  **3. PLAN PATH**

  **4. NAVIGATE**

**E. Depositing Object**
  **1. PLAN DEPOSIT [OBTAIN BIN POSE IN LOCAL FRAME] $\star$**
  *   **Method 1 (Small Bin $\star$):** Segmentation (DINO/CLIP/...) 
      *   **Detail:** Find "center" to get depth (semantic model).
  *   **Method 2:** OR big bin, Known shape.

  **2. EXECUTE DEPOSIT (DROP) $\star$**
  *   **Implementation:** MoveIt.