# Hindsight is 20/20: Leveraging Past Traversals to Aid 3D Perception

This is the official code release for paper **Hindsight is 20/20: Leveraging Past Traversals to Aid 3D Perception** (ICLR 2022).

### Citation
```
@inproceedings{you2022hindsight,
  title = {Hindsight is 20/20: Leveraging Past Traversals to Aid 3D Perception},
  author = {You, Yurong and Luo, Katie Z and Chen, Xiangyu and Chen, Junan and Chao, Wei-Lun and Sun, Wen and Hariharan, Bharath and Campbell, Mark and Weinberger, Kilian Q.},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year = {2022},
  month = apr,
  url = {https://openreview.net/forum?id=qsZoGvFiJn1}
}
```

## Environment
```bash
conda create --name hindsight python=3.8
conda activate hindsight
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install opencv-python matplotlib wandb scipy tqdm easydict scikit-learn

# ME
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
git checkout c854f0c # 0.5.4
python setup.py install
```
for OpenPCDet, follow [`downstream/OpenPCDet/docs/INSTALL.md`](downstream/OpenPCDet/docs/INSTALL.md) to install except
**you should install the spconv with the code in [`third_party/spconv`](third_party/spconv)**.

## Data Pre-processing
Please refer to [`data_preprocessing/lyft/LYFT_PREPROCESSING.md`](data_preprocessing/lyft/LYFT_PREPROCESSING.md) and
[`data_preprocessing/nuscenes/NUSCENES_PREPROCESSING.md`](data_preprocessing/nuscenes/NUSCENES_PREPROCESSING.md).

## Training and Evaluation
We implement the computation of SQuaSH as a submodule in OpenPCDet (as [sparse_query](downstream/OpenPCDet/pcdet/models/history_query/sparse_query.py)) and modify the
[KITTI dataloader](downstream/OpenPCDet/pcdet/datasets/kitti/kitti_dataset.py) / [augmentor](downstream/OpenPCDet/pcdet/datasets/augmentor/data_augmentor.py) to load the history traversals.

We include the corresponding configs of four detection models in
[downstream/OpenPCDet/tools/cfgs/lyft_models](downstream/OpenPCDet/tools/cfgs/lyft_models)
and [downstream/OpenPCDet/tools/cfgs/nuscenes_boston_models](downstream/OpenPCDet/tools/cfgs/nuscenes_boston_models).

Training:
```bash
cd downstream/OpenPCDet/tools
OMP_NUM_THREADS=6 bash scripts/dist_train.sh <num_of_gpus> --cfg_file <cfg> --merge_all_iters_to_one_epoch --fix_random_seed
```

Evaluation:
```bash
cd downstream/OpenPCDet/tools
OMP_NUM_THREADS=6 bash scripts/dist_test.sh <num_of_gpus> --cfg_file <cfg> --ckpt <ckpt_path>
```

### Checkpoints

## License


## Acknowledgement
This work uses [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [MinkowskiEngine
](https://github.com/NVIDIA/MinkowskiEngine) and [spconv](https://github.com/traveller59/spconv).
We thank them for open-sourcing excellent libraries for 3D understanding tasks.