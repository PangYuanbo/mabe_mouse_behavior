# MABe Challenge - Social Action Recognition in Mice - Competition Rules

## Competition Overview

**Competition Name:** MABe Consortium - Social Action Recognition in Mice
**Sponsor:** Cornell University
**Prize Pool:** $50,000 USD
**Competition Website:** https://www.kaggle.com/competitions/MABe-mouse-behavior-detection

## Competition Description

Animal social behavior is complex. This competition challenges you to build models to **identify over 30 different social and non-social behaviors** in pairs and groups of co-housed mice, based on markerless motion capture of their movements in top-down video recordings.

### Dataset
- Over **400 hours** of footage from 20+ behavioral recording systems
- All carefully labeled **frame-by-frame** by experts
- Goal: Recognize behaviors as accurately as a trained human observer
- Challenge: Overcome variability from different data collection equipment and motion capture pipelines

### Applications
- Automate behavior analysis in neuroscience, computational biology, ethology, and ecology
- Create a foundation for future ML and behavior research
- Deploy across numerous labs

## Timeline

- **September 18, 2025** - Start Date
- **December 8, 2025** - Entry Deadline (must accept rules before this date)
- **December 8, 2025** - Team Merger Deadline
- **December 15, 2025** - Final Submission Deadline

All deadlines are at **11:59 PM UTC**

## Evaluation

### Evaluation Metric

**F-Score variant** - The F scores are **averaged across each lab, each video**, and score **only the specific behaviors and mice that were annotated for a specific video**.

**Key Points:**
- Only behaviors that were annotated in a video are evaluated
- Different labs annotated different behaviors
- Must predict ALL behaviors that appear in annotations (both social AND non-social)
- NOT just social behaviors!

Full implementation available in competition evaluation code.

### Submission File Format

You must create a row in the submission file for **each discrete action interval**:

```csv
row_id,video_id,agent_id,target_id,action,start_frame,stop_frame
0,101686631,mouse1,mouse2,sniff,0,10
1,101686631,mouse2,mouse1,sniff,15,16
2,101686631,mouse1,mouse2,sniff,30,40
3,101686631,mouse2,mouse1,sniff,55,65
```

**Columns:**
- `row_id` - Unique row ID (simple enumeration)
- `video_id` - Video identifier
- `agent_id` - ID of mouse performing behavior (mouse1-mouse4)
- `target_id` - ID of mouse targeted by behavior
- `action` - Behavior type (e.g., sniff, attack, mount)
- `start_frame` - First frame of the action
- `stop_frame` - Last frame of the action

## Competition Rules

### Team Limits
- Maximum team size: **5 members**
- Team mergers allowed (subject to submission count limits)

### Submission Limits
- **5 submissions per day**
- **2 Final Submissions** for judging
- Must submit via Notebooks (Code Competition)

### Code Requirements

Submissions must be made through **Notebooks** with:
- CPU Notebook ≤ 9 hours run-time
- GPU Notebook ≤ 9 hours run-time
- Internet access **disabled**
- Freely & publicly available external data allowed (including pre-trained models)
- Submission file must be named `submission.csv`

### External Data and Tools

**Allowed:**
- External data that is publicly available and equally accessible to all participants at no cost
- Pre-trained models (if reasonably accessible)
- Automated Machine Learning Tools (AMLT) like Google AutoML, H2O Driverless AI

**Reasonableness Standard:**
- Must not exclude participants due to excessive costs
- Small subscription charges acceptable (e.g., Gemini Advanced)
- Proprietary datasets exceeding prize costs NOT reasonable

### Winner's Obligations

Prize winners must:
1. **Deliver final model code** including:
   - Training code
   - Inference code
   - Documentation with detailed methodology
   - Description of computational environment required
   - Must be capable of generating winning submission

2. **Grant Open Source License:**
   - CC BY 4.0 License for winning submission
   - Does not limit commercial use
   - Exception for generally available commercial software and incompatible pretrained models

3. **Provide detailed description:**
   - Architecture, preprocessing, loss function
   - Training details, hyper-parameters
   - Link to code repository with complete instructions
   - Must be reproducible

## Prizes

- **1st Place:** $20,000
- **2nd Place:** $10,000
- **3rd Place:** $8,000
- **4th Place:** $7,000
- **5th Place:** $5,000

## Dataset Description

### Files

**[train/test].csv** - Metadata about mice and recording setups:
- `lab_id` - Lab pseudonym (CalMS21, CRIM13, MABe22 are public datasets)
- `video_id` - Unique video identifier
- `mouse[1-4]_[strain/color/sex/id/age/condition]` - Mouse information
- `frames_per_second`
- `video_duration_sec`
- `pix_per_cm_approx`
- `video_[width/height]_pix`
- `arena_[width/height]_cm`
- `arena_shape`
- `arena_type`
- `body_parts_tracked` - Varies by lab (different tracking technology)
- `behaviors_labeled` - Which behaviors were annotated (sparse annotations)
- `tracking_method` - Pose tracking model used

**[train/test]_tracking/** - Feature data (parquet files):
- `video_frame` - Frame number
- `mouse_id` - Mouse identifier (1-4)
- `bodypart` - Tracked body part (varies by lab)
- `x` - X coordinate in pixels
- `y` - Y coordinate in pixels

**train_annotation/** - Training labels (parquet files):
- `agent_id` - Mouse performing behavior
- `target_id` - Mouse targeted by behavior (can be same as agent for self-directed behaviors)
- `action` - Behavior name
- `start_frame` - First frame
- `stop_frame` - Last frame

**sample_submission.csv** - Example submission format

### Important Notes

- **Hidden test set** - Actual test data provided during scoring (~200 videos)
- **Sparse annotations** - Not all behaviors annotated for all videos
- **Different labs** - Different body parts tracked, different behaviors annotated
- **Multiple annotators** - Some CalMS21 videos duplicated with different behavior annotations

## Behaviors in Training Data

### All Unique Actions (37 total)

Based on analysis of training annotations:

**High-frequency behaviors (>1000 intervals):**
1. sniff - 37,837 intervals
2. sniffgenital - 7,862 intervals
3. attack - 7,462 intervals
4. rear - 4,408 intervals
5. sniffbody - 3,518 intervals
6. approach - 3,270 intervals
7. sniffface - 2,811 intervals
8. mount - 2,747 intervals
9. escape - 2,071 intervals
10. reciprocalsniff - 1,492 intervals
11. defend - 1,409 intervals
12. selfgroom - 1,356 intervals
13. dig - 1,127 intervals
14. climb - 1,010 intervals

**Medium-frequency behaviors (100-1000 intervals):**
15. chase - 826 intervals
16. intromit - 691 intervals
17. avoid - 530 intervals
18. dominancemount - 410 intervals
19. dominance - 329 intervals
20. huddle - 299 intervals
21. disengage - 279 intervals
22. rest - 233 intervals
23. follow - 233 intervals
24. attemptmount - 223 intervals
25. shepherd - 201 intervals
26. flinch - 184 intervals
27. chaseattack - 124 intervals
28. tussle - 122 intervals
29. freeze - 105 intervals
30. exploreobject - 105 intervals

**Low-frequency behaviors (<100 intervals):**
31. submit - 86 intervals
32. run - 76 intervals
33. dominancegroom - 53 intervals
34. genitalgroom - 50 intervals
35. allogroom - 45 intervals
36. biteobject - 33 intervals
37. ejaculate - 3 intervals

### Behavior Categories

**Social investigation (7 behaviors):**
- sniff, sniffgenital, sniffface, sniffbody
- reciprocalsniff, approach, follow

**Mating behaviors (4 behaviors):**
- mount, intromit, attemptmount, ejaculate

**Aggressive behaviors (7 behaviors):**
- attack, chase, chaseattack, bite (NOT in data)
- dominance, defend, flinch

**Social interaction (11 behaviors):**
- avoid, escape, freeze, allogroom
- shepherd, disengage, run, dominancegroom, huddle
- dominancemount, tussle, submit, genitalgroom

**Non-social/Individual behaviors (8 behaviors):**
- rear, selfgroom, rest, dig, climb
- exploreobject, biteobject

## Critical Requirements for Model Design

### ⚠️ IMPORTANT: Must Predict ALL 37 Behaviors

The competition explicitly states: **"identify over 30 different social and non-social behaviors"**

This means:
- ✅ Include social behaviors (sniff, attack, mount, etc.)
- ✅ Include non-social behaviors (rear, selfgroom, rest, dig, climb, etc.)
- ✅ All 37 behaviors found in training data should have separate class IDs
- ❌ Do NOT map non-social behaviors to background class

### Evaluation Considerations

1. **Sparse annotations:** Only behaviors labeled for a specific video are evaluated
2. **Lab variability:** Different labs annotated different subsets of behaviors
3. **Per-video scoring:** F1 scores computed per video, then averaged
4. **Class imbalance:** Background frames ~72%, labeled behaviors ~28%
5. **Long-tail distribution:** Most behaviors appear in <1% of frames

## Previous Related Work

This competition builds on:
- **MABe 2021 Workshop** - CalMS21 at NeurIPS (https://arxiv.org/pdf/2104.02710.pdf)
- **MABe 2022 Competition** - MABe22 at ICML 2023 (https://arxiv.org/pdf/2207.10553.pdf)
- **CRIM13 Dataset** - CVPR (doi: 10.1109/CVPR.2012.6247817)

All pose and annotation files from CalMS21, MABe22, and CRIM13 are included as additional training data.

## Data Licenses

- **Data Access and Use:** CC BY 4.0
- **Winner License:** CC BY 4.0 (Open Source)
- May use data for any purpose (commercial or non-commercial)
- Must not transmit data to non-participants

## Eligibility

- Must be 18+ or age of majority in jurisdiction
- Residents worldwide except: Crimea, DNR, LNR, Cuba, Iran, Syria, North Korea
- Not subject to U.S. export controls or sanctions
- Competition Entity employees can participate but cannot win prizes

## Contact

For issues or questions, refer to:
- Competition Website: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection
- Kaggle Forums: Discussion forum on competition page
- Code Competition FAQ: Available on Kaggle

## Citation

Jennifer J. Sun, Markus Marks, Sam Golden, Talmo Pereira, Ann Kennedy, Sohier Dane, Addison Howard, and Ashley Chow. MABe Challenge - Social Action Recognition in Mice. https://kaggle.com/competitions/MABe-mouse-behavior-detection, 2025. Kaggle.
