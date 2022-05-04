# PyTorch Classification
PyTorch 기반 Classification 모델 구조 및 학습 기법을 테스트하기 위한 프로젝트

## Implementations
 * Darknet19
 * VGG16

## TODOs
- [x] ~~Darknet19~~
- [x] ~~VGG16~~
- [ ] Daknet53
- [ ] ResNet

## Requirements
* `PyTorch >= 1.8.1`
* `PyTorch Lightning`
* `Albumentations`
* `PyYaml`

## Train Detector
```python
python train_classifier.py --cfg configs/darknet19_tiny-imagenet.yaml
```

## Test Detector
```python
python test_classifier.py --cfg configs/darknet19_tiny-imagenet.yaml
```
