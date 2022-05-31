# Sparse-RCNN inplementation 

This is an unofficial pytorch implementation of SparseRCNN object detection as described in [Sparse R-CNN: End-to-End Object Detection with Learnable Proposals](https://arxiv.org/abs/2011.12450) by Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, Ping Luo

## Result
I only train this model with resnet50 backbone on the coco dataset for 6 epochs, and the default max epochs is 30.
```
AP : 35.747, AP50 : 53.073, AP75 : 38.319
```

## Roadmap
- [x] Use albumentations instead of the basic transforms
- [x] Add eval script and demo
- [ ] fp16 mixed precision training
- [ ] MAE
- [ ] Voc dataset support 
- [ ] Support for multiple GPUs

## Example 
```
python train.py --set BASE_ROOT /home/input/coco-2017-dataset/coco2017 SOLVER.IMS_PER_BATCH 4 MODEL.BACKBONE "resnet50"
```

## Reference
[original official implement](https://github.com/PeizeSun/SparseR-CNN) based on detectron2 and DETR
```text
@article{peize2020sparse,
  title   =  {{SparseR-CNN}: End-to-End Object Detection with Learnable Proposals},
  author  =  {Peize Sun and Rufeng Zhang and Yi Jiang and Tao Kong and Chenfeng Xu and Wei Zhan and Masayoshi Tomizuka and Lei Li and Zehuan Yuan and Changhu Wang and Ping Luo},
  journal =  {arXiv preprint arXiv:2011.12450},
  year    =  {2020}
}
```