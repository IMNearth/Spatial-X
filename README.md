# Spatial-VLN: Zero-Shot Vision-and-Language Navigation with Spatial Scene Priors

This repository contains the code and data for our series of work on **zero-shot Vision-and-Language Navigation (VLN)** using **global spatial scene priors**. We are the first to close-loop the pre-exploration to physically grounded 3D scene reconstruction for VLN agents and investigate how pre-explored 3D scene representations can provide a robust reasoning basis in multiple ways.

## Our Works

### SpatialNav: Leveraging Spatial Scene Graphs for Zero-Shot Vision-and-Language Navigation [[arXiv]](https://arxiv.org/abs/2601.06806)

> We propose a zero-shot VLN setting that allows agents to pre-explore the environment, and construct the **Spatial Scene Graph (SSG)** to capture global spatial structure and semantics. Based on SSG, **SpatialNav** integrates an agent-centric spatial map, compass-aligned visual representation, and remote object localization for efficient navigation. SpatialNav significantly outperforms existing zero-shot agents and narrows the gap with state-of-the-art learning-based methods.

### SpatialAnt: Autonomous Zero-Shot Robot Navigation via Active Scene Reconstruction and Visual Anticipation [[arXiv]](#)

> Building on SpatialNav, **SpatialAnt** addresses the reality gap when deploying pre-exploration-based agents on real robots. We introduce a **physical grounding strategy** to recover metric scale from monocular RGB-based reconstructed scene point clouds. We further design a **visual anticipation mechanism** that renders future observations from noisy point clouds for counterfactual reasoning. SpatialAnt achieves state-of-the-art zero-shot performance in both simulation and real-world deployment on the Hello Robot.

### How the Two Works Relate

```
----------------------------------------------------------------------------------------------------
SpatialNav                                          SpatialAnt
(Foundation)                                        (Real-World Extension)
----------------------------------------------------------------------------------------------------
                             Pre-exploration Assumption
High-quality point clouds              -->          Real deployment with monocular RGB camera only
are available after exploration                     (self-reconstructed noisy scenes)
----------------------------------------------------------------------------------------------------
                                  Core Mechanism
Spatial Scene Graph (SSG)              -->          Visual Anticipation
for global spatial reasoning                        for counterfactual reasoning
----------------------------------------------------------------------------------------------------
                                Experiment Settings   
Discrete (sim) + Continuous (sim)      -->          Continuous (sim) + Hello-Robot (real)
----------------------------------------------------------------------------------------------------
```

## Results

### SpatialNav

| Dataset | Setting | SR | SPL |
|:---|:---|:---:|:---:|
| R2R (val-unseen) | Discrete | 57.7 | 47.6 |
| REVERIE (val-unseen) | Discrete | 49.6 | 40.6 |
| R2R-CE (val-unseen) | Continuous | 64.0 | 52.2 |
| RxR-CE (val-unseen) | Continuous | 32.4 | 19.6 |

### SpatialAnt

| Dataset | Setting | SR | SPL |
|:---|:---|:---:|:---:|
| R2R-CE (val-unseen) | Continuous | 66.0 | 54.4 |
| RxR-CE (val-unseen) | Continuous | 50.8 | 35.6 |


## TODO

- [ ] Release the scene pre-exploration and reconstruction code
- [ ] Release spatial scene graph construction code
- [ ] Release the SpatialAnt and SpatialNav agent
- [ ] Release real-world deployment code for Hello Robot

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{zhang2026spatialnav,
  title={SpatialNav: Leveraging Spatial Scene Graphs for Zero-Shot Vision-and-Language Navigation},
  author={Zhang, Jiwen and Li, Zejun and Wang, Siyuan and Shi, Xiangyu and Wei, Zhongyu and Wu, Qi},
  journal={arXiv preprint arXiv:2601.06806},
  year={2026}
}
```


