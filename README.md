# MC-HDSet-hypergraph-matching

This is the code of the MC-HDSet matching algorithm proposed in

Jian Hou, Marcello Pelillo, Huaqiang Yuan. Hypergraph matching via game-theoretic hypergraph clustering. Pattern Recognition, vol. 125, 2022.

Dependency

1. You may need to compile the codes in the "ann_mwrapper" and "pwmetric" folders.  These two third-party libraries are used in approximate knn and similarity/distance calculation.

2. You need to download the vlfeat library by yourself, or use other SIFT detection implementations and make revisions in img2feat.m accordingly. This code is tested with vlfeat-0.9.19.

Usage

Simply run demo_mchdset.m for a demonstration of the matching process.

Citation

If you use this code in your research, we appreciate it if you cite the following paper:

Jian Hou, Marcello Pelillo, Huaqiang Yuan. Hypergraph matching via game-theoretic hypergraph clustering. Pattern Recognition, vol. 125, 2022.

