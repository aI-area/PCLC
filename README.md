# PCLC
**PCLC: Pattern-Conditioned Lifelong Consolidation for EEG emotion recognition.**

**PCLC is a novel framework for EEG-based emotion recognition that addresses catastrophic forgetting in continual learning scenarios. By discovering latent neural patterns and conditioning both classification and memory consolidation on these patterns, PCLC achieves robust cross-dataset adaptation from SEED-IV to SEED-V with minimal forgetting.

## Datasets 

>  SEED_IV: https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
>  SEED_V: https://bcmi.sjtu.edu.cn/home/seed/seed-v.html 

#⚙️ #The code automatically:
- Loads SEED-IV & SEED-V datasets
- Preprocesses EEG features (DE extraction, robust scaling)
- Trains PCLC on SEED-IV sessions 1→3
- Evaluates cross-dataset generalization (IV→V)
- Performs continual learning on SEED-V
- Generates all figures (PDF) and statistical analyses

📝 # Citation

If using this code, please cite our incoming paper on PCLC that is currently underreview in a IEEE Transaction journal.
