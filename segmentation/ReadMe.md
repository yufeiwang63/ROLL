### Code usage script:

Step 1: Generate training, validation and testing data using the command below.
    python training_data_generation.py

Step 2: Train UNet model using the example command below.
    python train_script.py --train_data_dir data/train --val_data_dir data/val --model_save_dir saved_models

Step 3: Test and visualise the model using the command below.
    python test_script.py --test_data_dir data/test/ --model saved_models/pytorchmodel_epoch*

### File structure/descriptions  
This folder include all code files for doing the segmentation, training background subtractor, training UNet.  
- unet/: implementation of UNet in pytorch.           
- DataLoader.py: pytorch dataloader for training UNet  
- segment_image.py: the key file that implements unknown object segmentation. 
- test_script.py: test a trained UNet model.
- train_script.py: train a UNet model.  
- train_data_generation.py: generating training data for UNet.    
- ReadMe.md: instructions for how to use the code to train the UNet.



