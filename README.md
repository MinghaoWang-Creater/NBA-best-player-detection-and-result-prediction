# NBA-best-player-detection-and-result-prediction
NBA best player detection and result prediction

# Contributors：

1) **Name: Jiahang Zhang**, NYU Student ID: jz7581 
2) **Name: Minghao Wang**, NYU Student ID: 

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
- LOF (Local Outlier Factor) and Isolation Forest are both used to detect outliers. 

**Outlier clustering**
- Outliers contain both “very good” and “very bad” players, so **K-Means** is applied to outliers.
- To represent overall excellence (position-agnostic), clustering focuses on **points/game, minutes/game, and shooting %**. 
- The number of clusters is chosen by the **elbow method** (K=5). 
- The “outstanding” cluster is selected as the cluster with the **highest average points/game**, and results from regular season & playoffs are merged; the intersection identifies players who are exceptional in both contexts.

### Notes on Results
- With the same outlier ratio, Isolation Forest finds **more** outliers than LOF (stricter definition) and its outstanding-player set includes LOF’s results. 
- LOF is **faster**, while Isolation Forest is **more reasonable** for this project according to the report. 
- The outlier ratio parameter was tested at 0.1 and 0.2 to adjust strictness. 

---
### Outputs (example):

<img width="1403" height="139" alt="image" src="https://github.com/user-attachments/assets/aa88d1e6-a0d4-4d6e-a113-ee58a5199962" />

<img width="644" height="665" alt="image" src="https://github.com/user-attachments/assets/3fc70c13-658f-4d76-a596-7b53415dd471" />

<img width="858" height="554" alt="image" src="https://github.com/user-attachments/assets/11463d15-3eb0-46d2-86a8-eea6b04f7ede" />



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

## 4. How to Run (example)

### Environment
- Python 3.x
- Recommended packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- Google Colab

```bash
pip install numpy pandas scikit-learn matplotlib





