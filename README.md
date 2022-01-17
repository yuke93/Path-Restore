## Path-Restore [[DOI](https://doi.org/10.1109/TPAMI.2021.3096255)][[arXiv](https://arxiv.org/abs/1904.10343)][[Project Page](https://www.mmlab-ntu.com/project/pathrestore/index.html)]

### Overview
- Path-Restore selects a specific network path for each image region.
<img src='imgs/framework.png' align="center" width="700">

### Test
The code was implemented with TensorFlow 1.9
- Put your noisy test images in `data/test/mine/` and run the following command.
```
cd codes/
python main.py --is_train=False --load_dir_test=../model/path-restore
```

### Citation
    @article{yu2021path,
     title={Path-Restore: Learning Network Path Selection for Image Restoration},
     author={Yu, Ke and Wang, Xintao and Dong, Chao and Tang, Xiaoou and Loy, Chen Change},
     journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
     year={2021},
     publisher={IEEE}
    }
