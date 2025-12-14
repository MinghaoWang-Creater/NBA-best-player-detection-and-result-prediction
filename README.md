# NBA-best-player-detection-and-result-prediction
NBA best player detection and result prediction
Please refer to the project report for detailed experimental principles and results.

# Contributors：

1) **Name: Jiahang Zhang**, NYU Student ID: jz7581 
2) **Name: Minghao Wang**, NYU Student ID: mw5945

---

# NBA Statistics ML Project

This repository contains a machine-learning project (Project: **NBA statistics data**) with two goals:

1) **Outstanding player mining** via outlier detection + clustering  
2) **Game outcome prediction** via supervised classification  


---

## 1. Project Overview

NBA provides rich historical statistics for players and teams. This project applies machine learning to:
- detect **outlier players** (both exceptional and poor performers) and then identify the **truly outstanding** group, and
- predict **win/loss outcomes** from match-level and team-level features.  


---

## 2. Task A — Outstanding Player Detection (Unsupervised)

### Problem Formulation
Selecting “outstanding players” is treated as an **unsupervised clustering problem** because the dataset has **no labels** for “best player.”  
The pipeline first detects **outliers**, then clusters them to separate **excellent** vs **poor** players.  


### Data & Features
The project uses **career-level** player statistics (regular season and playoffs) because multi-year averages better reflect true ability than a single season. 

Engineered features include:
- minutes per game, points per game, turnovers per game, rebounds per game, assists per game,
- blocks per game, field-goal %, free-throw %  


Important data handling:
- remove players with very low playing time (e.g., minutes/game < 3) to reduce noise, 
- account for missing historical stats: before ~1970, steals/blocks/turnovers were not recorded, so the dataset is split based on whether these fields are all zero to improve robustness. 

### Methods
**Outlier detection**
- LOF (Local Outlier Factor) is used to detect outliers. 

**Outlier clustering**
- Outliers contain both “very good” and “very bad” players, so **K-Means** is applied to outliers.
- To represent overall excellence (position-agnostic), clustering focuses on **points/game, minutes/game, and shooting %**. 
- The number of clusters is chosen by the **elbow method** (K=5). 
- The “outstanding” cluster is selected as the cluster with the **highest average points/game**, and results from regular season & playoffs are merged; the intersection identifies players who are exceptional in both contexts.

### Notes on Results
- The outlier ratio parameter was tested at 0.1 and 0.2 to adjust strictness. 

---
### Outputs:
<img width="865" height="211" alt="image" src="https://github.com/user-attachments/assets/ba23d94c-0635-429e-b877-a3f424216a43" />

<img width="636" height="658" alt="聚类1" src="https://github.com/user-attachments/assets/7ddc1146-b3e9-4590-a56b-edc52a99da97" />

<img width="844" height="547" alt="聚类2" src="https://github.com/user-attachments/assets/62454332-ae3d-4054-a790-836582bfa28b" />



## 3. Task B — Game Outcome Prediction (Supervised)

### Data Source
Match data (preseason + regular season + playoffs) for **2015–2017** seasons is collected from basketball-reference.com, along with team capability/ranking metrics. 

### Feature Engineering (high level)
From match results and schedules, features include:
- home/away info and final scores → win/loss label (HW = home win), net score margin, and points, 
- recent form features: each team’s **last 3 games** win/loss indicators (hm1–hm3, vm1–vm3), 
- weekly aggregated averages from cumulative net score / weekly stats to reflect team state. 
- team ranking features: MOV/A, ORtg/A, DRtg/A, NRtg/A are added into the match dataset. 

Data processing:
- normalize/standardize features to improve convergence and accuracy, :contentReference
- remove highly correlated features (> 0.9) to reduce redundant dimensions (examples include (vMOV,vNRtg), (hMOV,hNRtg), (hSco_avew,vSco_avew)). 

### Models
- **SVM** and **Logistic Regression** are trained and compared. 

### Notes on Results (from the report)
- SVM and Logistic Regression achieve similar accuracy; Logistic Regression runs faster. 
- Reported prediction accuracy reaches about **70%**. 

---

### Outputs:

<img width="1411" height="581" alt="image" src="https://github.com/user-attachments/assets/18e83259-71ab-4947-845c-2c6e288a77ee" />

<img width="570" height="482" alt="image" src="https://github.com/user-attachments/assets/5cb954b6-230c-4244-be32-a7b440dd8186" />

<img width="568" height="528" alt="image" src="https://github.com/user-attachments/assets/6013d16b-a0bf-4fc5-bb4a-0897431dc27d" />

<img width="569" height="528" alt="image" src="https://github.com/user-attachments/assets/d46711ac-fc47-4d05-94cf-90cb78950d1b" />

<img width="1400" height="469" alt="image" src="https://github.com/user-attachments/assets/cbbdf559-2d6e-43d6-9c4d-34e2af330025" />

<img width="1399" height="459" alt="image" src="https://github.com/user-attachments/assets/fce59820-0f29-42a7-a1a9-0244b2c49001" />



## 4. How to Run (example)

### Environment
- Python 3.x
- Recommended packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- Google Colab

```bash
pip install numpy pandas scikit-learn matplotlib





