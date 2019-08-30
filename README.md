# Segmentation-Toolbox-for-Medical-Imaging
A PyTorch codebase template for medical imaging segmentation developed by Jesse Sun from Dr. Bo Wang's lab. 

This template codebase was derived from the codebase used for the work `HAUNet: Hyper-Dense Attention U-Nets for Interpretable 
Medical Imaging Segmentation` (preprint coming soon) by Jesse Sun and Bo Wang from UHN.

We made this codebase public to speed up the development of machine learning research in healthcare.

Currently, the codebase is linked to the database AC17. You may override the `data/ac17_dataloader.py` file by writing your own
custom dataloader class for your dataset of choice.

TODO:
1. Comment all the code.
2. Add a file on how to navigate the codebase.

There may be some instabilities currently. Notably, directory locations are left blank or have `'./'` prefixing it, please
fill in the locations depending on where you are storing the files on your environment.

If you do find our work useful, please consider citing the following (not published yet):

```
@inproceedings{sun2019haunet,
    title={HAUNet: Hyper-Dense Attention U-Nets for Interpretable Medical Imaging Segmentation},
    author={Sun, Jesse and Wang, Bo},
    booktitle={},
    year={2019}
}
```
