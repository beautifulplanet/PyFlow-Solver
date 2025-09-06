# Workspace Learning Report
Generated: 2025-08-30T00:43:38.244718+00:00

## Aggregate Metrics
Total files analyzed: **2214**
Failure-indicator files: 1054 (47.61%)
Average functions per code file: 48.29
Average classes per code file: 5.26
Top external imports: numpy:119, matplotlib:47, Grid:31, solve_lid_driven_cavity:24, pytest:21, LiveLogger:16, bicgstab:15, csr_matrix:15, unittest:15, Path:14, subprocess:14, time:14, cg:11, pandas:9, spdiags:8, types:8, PyFOAMSolver:7, scipy:6, FinitudeSolver:6, jit:6, Extension:6, pybind11:6, pyflow_core_cfd:6, annotations:5, plot_velocity_field:5


## Label Distribution
- log: 1726
- legacy: 1475
- failure-indicator: 1054
- success-indicator: 670
- uncategorized: 394
- prototype: 94

## Representative Failures (first 10)
- C:\Users\Elite\Documents\commands\Failures_Postmortem.md | reasons: ['# Failures: All Attempts', '- Numerical instability at high Reynolds numbers (see benchmark_failure_report.txt)']
- C:\Users\Elite\Documents\commands\failure_logging_helper.py | reasons: ['"""Structured failure logging helper.', 'from failure_logging_helper import log_failure']
- C:\Users\Elite\Documents\commands\failure_taxonomy.json | reasons: ['"# failures all attempts": {', '"canonical": "# Failures: All Attempts",']
- C:\Users\Elite\Documents\commands\Legacy\10.4\Dosidon 14 final fail log.txt | reasons: ['You are absolutely correct. My previous analysis was incomplete. I have re-executed a full, 20-cycle deep review of all provided documentation, and I have fo...', '1. The Failure of Established Models at Singularities:']
- C:\Users\Elite\Documents\commands\Legacy\10.4\fail log 23.txt | reasons: ['## The Problem: Why Current Languages Fail', 'Think of a normal programming language like Python or C++. It is a set of rules for manipulating data. However, these rules are incomplete. They contain "sin...']
- C:\Users\Elite\Documents\commands\Legacy\10.4\fail log 24.txt | reasons: ['I\'m now revisiting the "Dosidon project" details. My focus is on the "Marty McFly Bingo Test" lensing prediction. I\'m recalling how our Axiomatically Constra...', 'I\'ve been examining the core principles behind the "Marty McFly Bingo Test" lensing prediction. My focus is on synthesizing the details of our Axiomatically ...']
- C:\Users\Elite\Documents\commands\Legacy\10.4\fail log 25.txt | reasons: []
- C:\Users\Elite\Documents\commands\Legacy\10.4\fail log 26.txt | reasons: ['Ok we will name the app something scientific and proceed. you are the ai agent in charge as i failed before. you know my failures. please take over as i watch', 'Understood. The failure logs, the discarded theories, and the corrected derivations are all part of our collective memory now. Your past is my knowledge. I a...']
- C:\Users\Elite\Documents\commands\Legacy\10.4\fail log ultra.txt | reasons: ['I moved to ultra model and worked till we failed over and over to work on theory 18 hours a day 2 months about.', 'we later failed 22 times. Then i created a 7 file master set of original math and work logs. Great.']
- C:\Users\Elite\Documents\commands\Legacy\dos logs\lega\206 wonky idea Fail log_250720_011030.txt | reasons: ['5. Falsifiable Outcome (Pass/Fail Criteria)', '* Fail (Theory Falsified): The specific predicted fractal pattern does not form. The patterns at 440.0 Hz are statistically indistinguishable from the simple...']

## Success Indicators (first 10)
- C:\Users\Elite\Documents\commands\Legacy\Dosidon 11 Final v10.3.txt | keywords: ['CFD', 'cfd', 'solver', 'navier', 'stokes']
- C:\Users\Elite\Documents\commands\Legacy\Dosidon 9.8 File 8 Final.txt | keywords: ['CFD', 'cfd', 'solver', 'navier', 'stokes']
- C:\Users\Elite\Documents\commands\Legacy\Dosidon 9.9 File 9 Final.txt | keywords: ['CFD', 'cfd', 'solver', 'navier', 'stokes']
- C:\Users\Elite\Documents\commands\Legacy\Dosidon 9.9.1 File 10 Final.txt | keywords: ['CFD', 'cfd', 'solver', 'navier', 'stokes']
- C:\Users\Elite\Documents\commands\Legacy\Dosidon Final 12  v10.4.txt | keywords: ['CFD', 'cfd', 'solver', 'navier', 'stokes']
- C:\Users\Elite\Documents\commands\Legacy\10.4\Dosidon 14 final fail log.txt | keywords: ['CFD', 'cfd', 'solver', 'navier', 'stokes']
- C:\Users\Elite\Documents\commands\Legacy\10.4\Dosidon Final13  v11.4.txt | keywords: ['CFD', 'cfd', 'solver', 'navier', 'stokes']
- C:\Users\Elite\Documents\commands\Legacy\dos logs\lega\10\10 - Copy (11)\52 triumph newton_250707_023829.txt | keywords: ['ai']
- C:\Users\Elite\Documents\commands\Legacy\dos logs\lega\10\10 - Copy (11)\56 - 5 rng sct final 1-1_250707_230122.txt | keywords: ['ai']
- C:\Users\Elite\Documents\commands\Legacy\dos logs\lega\Apps\50 seed code demo page 2 final1_250707_003444.txt | keywords: ['ai']

## Heuristic Lessons
- Prefer small, composable functions (avg length evaluated).
- Capture failure context early in log naming conventions.
- External import diversity hints at experimentation; curate a stable core set.
- Prototype to final path can be traced via name evolution (v0 -> final).

## Data Schema Summary
Each JSONL line fields: id, path, ext, size, size_bucket, labels[], datetime_hint, keyword_hits[], fail_reasons[], remediation[], code_metrics?, notebook_meta?, text_stats?