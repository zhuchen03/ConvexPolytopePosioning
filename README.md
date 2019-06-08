# ConvexPolytopePosioning
This repository provides codes to reproduce the major experiments in the paper [Transferable Clean-Label Poisoning Attacks on Deep Neural Nets](https://arxiv.org/abs/1905.05897), (ICML 2019).

<div  align="center">
<img src="https://user-images.githubusercontent.com/18202259/59151206-f1462b80-89e3-11e9-8ea7-86d6da27fb9b.png" width = "60%" />
</div>

If you find this code useful for your research you could cite
```
@inproceedings{zhu2019transferable,
  title={Transferable Clean-Label Poisoning Attacks on Deep Neural Nets},
  author={Zhu, Chen and Huang, W Ronny and Shafahi, Ali and, Li, Hengduo and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={International Conference on Machine Learning},
  pages={7614--7623},
  year={2019}
}
```

## Prerequisites
The experiments can be reproduced with PyTorch 1.0.1 and CUDA 9.0 on Ubuntu 16.04. 

Before running any experiments, please download our split of the CIFAR10 dataset [here](https://www.dropbox.com/s/451maqtq716ggr4/CIFAR10_TRAIN_Split.pth), create a directory datasets/ and move the file into datasets/.
One example for doing that is executing the following under this project directory
```bash
mkdir datasets && cd datasets && wget https://www.dropbox.com/s/raw/451maqtq716ggr4/CIFAR10_TRAIN_Split.pth
```

We also provide most of the substitute and victim models that we used for our experiments via 
[Dropbox](https://www.dropbox.com/s/7dorf2grr3vdgqt/model-chks.tgz?dl=0). 
You could also train any substitute model as used in the paper with train_cifar10_models.py, where we have tweaked the code from [kuangliu](https://github.com/kuangliu/pytorch-cifar.git) to add Dropout operations into the networks and choosing different subsets of the training data.
One example for running the training is:
```bash
python train_cifar10_models.py --gpu 0 --net ResNet50 --train-dp 0.25  --sidx 0 --eidx 4800
```
Feel free to contact us if you have any question or find anything missing.


## Launch
We will add more examples as well as the poisons we used in the paper soon, 
but here are some simple examples we have cleared up.

To attack the transfer learning setting:
```
bash launch/attack-transfer.sh
``` 

To attack the end-to-end setting:
```
bash launch/attack-end2end.sh
```

