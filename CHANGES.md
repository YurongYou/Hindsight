## Changes to OpenPCDet
* `downstream/OpenPCDet/pcdet/datasets/kitti/kitti_object_eval_python/eval.py`
to support evaluation by range.
* To support spatial quantization for Hindsight
    ```
    downstream/OpenPCDet/pcdet/datasets/augmentor/data_augmentor.py
    downstream/OpenPCDet/pcdet/datasets/dataset.py
    ```
* Data loading
    ```
    downstream/OpenPCDet/pcdet/datasets/kitti/kitti_dataset.py
    ```
* To build the spatial featurizer
    ```
    downstream/OpenPCDet/pcdet/models/detectors/detector3d_template.py
    ```
* Minor changes to `downstream/OpenPCDet/tools/train.py`

## Changes to spconv
To support reverse index of points in the voxel (`points_to_voxel_3d_np_yurong`)
```
third_party/spconv/spconv/utils/__init__.py
third_party/spconv/include/spconv/point2voxel.h
third_party/spconv/src/utils/all.cc
```