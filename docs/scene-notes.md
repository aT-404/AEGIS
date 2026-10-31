# Scene Notes & Observations

Reference observations for sample industrial and surveillance footage used during system tuning:

### 1. `factory_floor_sample.mp4`
- **Resolution:** 1280×720 @ 30 FPS
- **Camera Mount:** Elevated corner view (~3.5m height, 40° downward tilt)
- **Occupancy:** 2–5 workers walking between workstations
- **Zones:**
  - `R-01` (Machinery Exclusion Zone): `[[120, 250], [480, 250], [480, 600], [120, 600]]`
  - `R-02` (High-Voltage Area): `[[310, 200], [640, 200], [640, 580], [310, 580]]`
- **Tuning Note:** Workers occasionally stop near the perimeter of R-02. Dwell timer hysteresis prevents false triggers from slight centroid jitter.

### 2. `entrance_gate_sample.mp4`
- **Resolution:** 1920×1080 @ 25 FPS
- **Camera Mount:** Gatehouse eye-level mount facing turnstiles / barrier
- **Occupancy:** Bursts of pedestrians during shift changes (up to 8 people concurrently)
- **Zones:**
  - `GATE-01` (Entrance Monitoring): `[[550, 300], [850, 300], [850, 720], [550, 720]]`
- **Tuning Note:** Crowd threshold set to 5 with 1-second cooldown suppression.

### 3. `warehouse_sample.mp4`
- **Resolution:** 640×480 @ 15 FPS
- **Camera Mount:** Ceiling aisle view
- **Lighting:** Artificial fluorescent, moderate contrast
- **Occupancy:** 1–3 forklift and foot workers

### 4. `outdoor_perimeter_sample.mp4`
- **Resolution:** 1280×720 @ 30 FPS
- **Camera Mount:** Fence-line perimeter post
- **Lighting:** Daylight / outdoor shadow shifts
- **Zones:** Boundary security line
