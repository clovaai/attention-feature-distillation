## Show, Attend and Distill: Knowledge Distillation via Attention-based Feature Matching
Official pytorch implementation of ["Show, Attend and Distill: Knowledge Distillation via Attention-based Feature Matching" (AAAI-2021)](http://34.94.61.102/paper_AAAI-9785.html)

## Requirements
- Python3
- PyTorch (> 1.2.0)
- torchvision
- numpy
- Pillow

## Training
We include a trained WRN-40-2 parameters at ```/trained/wrn40x2/model.pth```. \
Run ```main.py``` with student network as WRN-16-2 and teacher as WRN-40-2 to reproduce experiment result on CIFAR100.
```
python main.py --data_dir PATH_TO_DATA --data CIFAR100 --trained_dir /trained/wrn40x2/model.pth\
 --model wrn16x2 --model_t wrn40x2 --beta 200
```

## License

```
Copyright 2021-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
