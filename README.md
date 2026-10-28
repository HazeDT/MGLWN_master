# MGLWN_master
This is the code library for the implementation of Multi-hop Graph Lifting Wavelet Network[MGLWN: An Interpretable Multi-hop Graph Lifting Wavelet Network with Mixture-of-Experts Fusion for Robust Fault Diagnosis](https://ieeexplore.ieee.org/document/11333912).

![MGLWN-based-fault diagnosis](https://github.com/HazeDT/MGLWN_master/blob/main/MGLWN.png)


# Note
At the core of MGLWN is a novel multi-hop graph lifting convolution, designed to selectively fuse multi-scale features through a mixture-of-experts strategy. To power this convolution, two key mechanisms are incorporated: First, a multi-hop graph lifting decomposition scheme is proposed to construct computational trees for each node, effectively alleviating the structural information loss introduced by conventional graph lifting schemes. Second, a graph attention-based lifting operator is developed to parameterize the predictor and updater. This embedded attention mechanism enables the model to capture intrinsic topological dependencies by dynamically weighting neighboring nodes during the decomposition process.


# Implementation
python ./MGLWN_master/train_graph.py --model_name MGLWN_master  --data_name THUGearboxRadius --data_dir r"./datasets/THUGearboxRadius.pkl"


# Citation

MGLWN:
@ARTICLE{MGLWN,
  author={Li, Tianfu and Li, Wenjun and Wang, Fujin and Sun, Chuang and Zhao, Zhibin and Yan, Ruqiang},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={MGLWN: An Interpretable Multi-hop Graph Lifting Wavelet Network with Mixture-of-Experts Fusion for Robust Fault Diagnosis}, 
  year={2026},
  volume={},
  number={},
  pages={},
  doi={}}


