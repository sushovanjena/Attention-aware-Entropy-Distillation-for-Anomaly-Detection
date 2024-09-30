# Attend, Distill, Detect: Attention-aware-Entropy-Distillation-for-Anomaly-Detection
# Accepted in International Conference on Pattern Recognition 2024 (h5 index- 56)

# Environment
requirements.txt

# Training
Train a model:

Dataset has to be downloaded data and has to be passed in the below argument.

parser.add_argument("--mvtec-ad", type=str, default='', help="MvTec-AD dataset path")
``` 
python main.py train --epochs 400
```
After running this command, a directory `snapshots/` should be created, inside which checkpoint will be saved.

# Testing
Evaluate a model:
```
python main.py test --category carpet --checkpoint snapshots/best.pth.tar
```
This command will evaluate the model specified by --checkpoint argument. 


# Citation

If you find the work useful in your research, please cite our papar.
```
@misc{jena2024attenddistilldetectattentionaware,
      title={Attend, Distill, Detect: Attention-aware Entropy Distillation for Anomaly Detection}, 
      author={Sushovan Jena and Vishwas Saini and Ujjwal Shaw and Pavitra Jain and Abhay Singh Raihal and Anoushka Banerjee and Sharad Joshi and Ananth Ganesh and Arnav Bhavsar},
      year={2024},
      eprint={2405.06467},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.06467}, 
}
```
