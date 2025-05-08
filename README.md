# **AMCO Navigation System**

The **AMCO Navigation System** is an advanced autonomous navigation framework designed for dynamic environments. It leverages pre-captured sensor data, real-time obstacle detection, and a GUI for efficient path planning.

---

## **Features**

* **Pre-captured Data:**
  The system comes with pre-collected images, point clouds, and segmentation masks to facilitate quick deployment and testing.

* **GUI Integration:**
  An intuitive graphical interface that supports dynamic obstacle detection and real-time cost map generation.

* **Sensor Data Ready:**
  Pre-configured to work with readily available sensor data, eliminating the need for additional configurations.

---

## **Setup Instructions**

1. **Extract the Environment and Data:**
   * Download the following in the AMCO directory:
     * amco virtual environment zip file: https://drive.google.com/file/d/1-QMhpc3Qon9gTuGlkElMi5EpEDypxH1I/view?usp=sharing
     * data zip  file: https://drive.google.com/file/d/1km8LpZVdTbi53efmQYmIcoX9FcEc9H-U/view?usp=sharing
     
   * Unzip the amco virtual environment and data zip files into the main `AMCO` directory.

2. **Activate the Virtual Environment:**

   ```bash
   source amco/bin/activate
   ```
3. **Modification of Paths:**
    Navigate to the follwoing files and modify the paths accordingly:
   * `src/planner/dwa`
   * `src/preprocessing/map.py`
  
5. **Run the Navigation Script:**
   Navigate to the `src/planner` directory and execute the DWA planner script:

   ```bash
   python src/planner/dwa.py
   ```

---

## **Usage Notes**

* Ensure that the virtual environment is activated before running the script to avoid dependency issues.
* The GUI will display the navigation path, dynamic obstacles, and the generated cost map in real-time.

---
