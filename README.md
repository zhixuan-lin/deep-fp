# Deep Webstie Fingerprinting

This is a Pytorch implementation of [Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning. Website fingerprinting based on CNN](https://arxiv.org/abs/1801.02265). This is for one of my course projects.

Run training:

```
python tools/train_net.py --config ./config/openworld-nodef.yaml
```

Run testing:

```
python tools/test_net.py --config ./config/openworld-nodef.yaml
