# intro-ml-nycu-2025

好玩共編區：
https://hackmd.io/@ZxrF4aU8SHaTG2JAJ5ZasQ/SJjDMBzaxl
期中報告：
https://hackmd.io/@s1ash/SJNrV6w0le
Course repo for **CSCS20024 — Introduction to Machine Learning (Fall 2025, NYCU)**  

> This repository contains homework solutions, notes, and the final project for the course.

## Structure
```
intro-ml-nycu-2025/
├── hw1_dataset_split/
├── hw2_linear_regression/
├── final_project/
│   ├── proposal/
│   ├── code/
│   └── presentation_slides/
├── requirements.txt
├── .gitignore
└── README.md
```

## Homework Checklist
- [ ] HW1 — Dataset Splitting & Data Leakage
- [ ] HW2 — Linear Regression
- [ ] HW3 — (TBD)
- [ ] HW4 — (TBD)

## Final Project
- 4 students per team
- Proposal due: Week 9
- Presentations: Weeks 15–16
- Deliverables: 5–8pp report (PDF) + 10-min presentation

## Environment
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dev Notes
- Use feature branches and pull requests.
- Keep notebooks and large data out of Git (see `.gitignore` and the `data/` note below).
- Add minimal, reproducible examples for grading and future reference.

## Data Handling
- Create a local `data/` directory for raw datasets (not tracked by Git).
- For reproducibility, include small samples in `data_samples/` when needed.
