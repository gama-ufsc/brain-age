for LR in 1e-3 1e-4 1e-5
do
    for i in {0..4}
    do
        python train_brainage.py resnet $LR
        python train_brainage.py resnet $LR imagenet $i
        python train_brainage.py resnet $LR brats $i
        python train_brainage.py resnet $LR imagenet+brats $i
        python train_brainage.py unet $LR
        python train_brainage.py unet $LR brats $i
    done
done