# NBA-best-player-detection-and-result-prediction
NBA best player detection and result prediction.

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
Match data (preseason + regular season + playoffs) for **2022-2024** seasons is collected from basketball-reference.com, along with team capability/ranking metrics. 

### Feature Engineering (high level)
From match results and schedules, features include:
- home/away info and final scores → win/loss label (HW = home win), net score margin, and points, 
- recent form features: each team’s **last 3 games** win/loss indicators (hm1–hm3, vm1–vm3), 
- weekly aggregated averages from cumulative net score / weekly stats to reflect team state. 
- team ranking features: MOV/A, ORtg/A, DRtg/A, NRtg/A are added into the match dataset. 

Data processing:
- normalize/standardize features to improve convergence and accuracy, :contentReference
- Adopt three feature dimensionality processing schemes:
  - No dimensionality processing is performed; the original 14 features are used.
  - Dimensionality processing is performed using PCA, selecting the first 8 principal components.
  - Draw a Pearson correlation heatmap, and remove highly correlated features (> 0.9) to reduce redundant dimensions (examples include (vMOV, vNRtg), (hMOV, hNRtg), (hSco_avew, vSco_avew)).

### Models
- **Logistic Regression** is trained and compared. 

### Notes on Results (from the report)
- Reported prediction accuracy reaches about **70%**.
- Using PCA for feature dimensionality processing offers good runtime efficiency.

---

### Outputs:

<img width="865" height="356" alt="image" src="https://github.com/user-attachments/assets/bc95d365-cb43-40d7-ad83-a4a1169aaeba" />

#### No dimensionality processing：
Accuracy_train:0.699077

Accuracy_test:0.690382

Time cost: :	0.28s

#### PCA:

Original feature count: 14

Feature examples: ['hm1', 'hm2', 'hm3', 'vm1', 'vm2']...


Feature count after PCA dimensionality reduction: 8

Cumulative explained variance ratio: 0.9598

<img width="1390" height="490" alt="PCA" src="https://github.com/user-attachments/assets/3604bc20-0fcb-45f0-ac78-cdb66adc0782" />

Training set accuracy: 0.6889

Test set accuracy: 0.7207

Time Cost： :	0.01s

#### Select features using Pearson correlation heatmap

<img width="1533" height="1453" alt="pearson1" src="https://github.com/user-attachments/assets/34191cce-ccee-4144-93ae-ea928c5147e4" />

Accuracy_train:0.690508

Accuracy_test:0.714097

Time cost： :	0.03s

## 4. How to Run (example)

### Environment
- Python 3.x
- Recommended packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- Google Colab

```bash
pip install numpy pandas scikit-learn matplotlib

- Game outcomes prediction results will be stored in the `prediction_results` folder, saved in CSV format.



