for i in {0..5}
    python train_brainage.py resnet 1e-4
    python train_brainage.py resnet 1e-4 imagenet $i
    python train_brainage.py resnet 1e-4 brats $i
    python train_brainage.py resnet 1e-5 imagenet+brats $i
    python train_brainage.py unet 1e-3
    python train_brainage.py unet 1e-4 brats $i
done