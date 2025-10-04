# V8 Kaggle Submission Guide

## Quick Steps

1. **Upload Model to Kaggle Dataset**
   - Go to https://www.kaggle.com/datasets → "New Dataset"
   - Upload: `checkpoints/v8_5090/best_model.pth`
   - Name: `mabe-v8-model`

2. **Create Kaggle Notebook**
   - Upload `kaggle_submission_v8.ipynb` to competition
   - Add datasets: `mabe-v8-model` + `MABe-mouse-behavior-detection`
   - Enable GPU (T4/P100)

3. **Run & Submit**
   - Click "Run All"
   - Wait ~1-2 hours for inference
   - Submit to competition

## Files Created

- `kaggle_submission_v8.ipynb` - Kaggle notebook
- `inference_v8.py` - Local inference script
- `test_interval_f1.py` - Interval F1 tests

## V8 Model Performance

- Validation Action Acc: 86.31%
- Agent/Target Acc: 98%+
- 28 fine-grained behavior classes
- Model size: 18.3 MB
