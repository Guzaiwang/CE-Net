# CE-Net
The manuscript has been accepted in TMI.

Please start up the "visdom" before running the main.py.
Then, run the main.py file.

We have uploaded the DRIVE dataset to run the retinal vessel detection. The other medical datasets will be
uploaded in the next submission.

The submission mainly contains:
1. architecture (called CE-Net) in networks/cenet.py
2. multi-class dice loss in loss.py
3. data augmentation in data.py

Update:
We have modified the loss function. 
The cuda error (or warning) will not occur. 

Update:
The test code has been uploaded. 
Besides, we release a pretrained model, which achieves 0.9819 in the AUC scor in the DRIVE dataset. 
