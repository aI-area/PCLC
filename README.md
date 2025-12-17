# PCLC
**PCLC: Pattern-Conditioned Lifelong Consolidation for EEG emotion recognition. Implements DDE, Pattern Identifier with PCR loss, MoE routing, and PSCR contrastive replay for cross-dataset continual learning on SEED-IV→SEED-V.**

## Datasets 

### SEED_IV: https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
### SEED_V:

⚙️
# The code automatically:
# 1. Loads SEED-IV & SEED-V datasets
# 2. Preprocesses EEG features (DE extraction, robust scaling)
# 3. Trains PCLC on SEED-IV sessions 1→3
# 4. Evaluates cross-dataset generalization (IV→V)
# 5. Performs continual learning on SEED-V
# 6. Generates all figures (PDF) and statistical analyses

📝 Citation
If using this code, please cite our paper on Pattern-Conditioned Lifelong Consolidation for EEG emotion recognition.
