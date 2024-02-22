# PIP-Net: Patch-Based Intuitive Prototypes for Interpretable Image Classification
This repository presents the PyTorch code for PIP-Net (Patch-based Intuitive Prototypes Network). 

**Main Paper at CVPR**: ["PIP-Net: Patch-Based Intuitive Prototypes for Interpretable Image Classification"](https://openaccess.thecvf.com/content/CVPR2023/papers/Nauta_PIP-Net_Patch-Based_Intuitive_Prototypes_for_Interpretable_Image_Classification_CVPR_2023_paper.pdf) introduces PIP-Net for natural images.\
**Medical applications, data quality inspection and manual corrections**: [Interpreting and Correcting Medical Image Classification with PIP-Net](https://link.springer.com/chapter/10.1007/978-3-031-50396-2_11), applies PIP-Net to X-rays and skin lesion images where biases can be fixed by (manually) disabling prototypes. \
**Evaluation of part-prototype models like PIP-Net**: [The Co-12 Recipe for Evaluating Interpretable Part-Prototype Image Classifiers](https://arxiv.org/abs/2307.14517), presented at the [XAI World Conference](https://xaiworldconference.com/) in July 2023. 


PIP-Net is an interpretable and intuitive deep learning method for image classification. PIP-Net learns prototypical parts: interpretable concepts visualized as image patches. PIP-Net classifies an image with a sparse scoring sheet where the presence of a prototypical part in an image adds evidence for a class. PIP-Net is globally interpretable since the set of learned prototypes shows the entire reasoning of the model. A smaller local explanation locates the relevant prototypes in a test image. The model can also abstain from a decision for out-of-distribution data by saying “I haven’t seen this before”. The model only uses image-level labels and does not rely on any part annotations. 

![Overview of PIP-Net](https://github.com/M-Nauta/PIPNet/blob/main/nauta_pipnet_cpvr.png)

### Required Python Packages:
* [PyTorch](https://pytorch.org/get-started/locally/) (incl torchvision, tested with PyTorch 1.13)
* [tqdm](https://tqdm.github.io/)
* scikit-learn
* openCV (optional, used to generate heatmaps)
* pandas
* matplotlib

### Training PIP-Net
PIP-Net can be trained by running `main.py` with arguments. Run `main.py --help` to see all the argument options. Recommended parameters per dataset are present in the `used_arguments.txt` file (usually corresponds to the default options). 

#### Training PIP-Net on your own data
Want to train PIP-Net on another dataset? Add your dataset in ``util/data.py`` by creating a function ``get_yourdata`` with the desired data augmentation (that captures human perception of similarity), add it to the existing ``get_data`` function in ``util/data.py`` and give your dataset a name. Use ``--dataset your_dataset_name`` as argument to run PIP-Net on your dataset. 

Other relevant arguments are for example  ``--weighted_loss`` which is useful when your data is imbalanced. In case of a 2-class task with presence/absence reasoning, you could consider using ``--bias`` to include a traininable bias term in the linear classification layer (which could decrease the OoD abilities) such that PIP-Net does not necessarily need to find evidence for the absence-class.

Check your `--log_dir` to keep track of the training progress. This directory contains `log_epoch_overview.csv` which prints statistics per epoch. File `tqdm.txt` prints updates per iteration and potential errors. File `out.txt` includes all print statements such as additional info. See the **Interpreting the Results** section for further details. 

Visualizations of prototypes are included in your `--log_dir` / `--dir_for_saving_images`. 

#### Trained checkpoints
Various trained versions of PIP-Net are made available:

- PIP-Net with the ConvNext backbone (recommended) trained on the birds CUB-200-2011 dataset is available [for download here](https://drive.google.com/file/d/1G8iiXgZ5gENYicwS8nLIg2Gf43A49kKm/view)  (320MB). Download the CUB dataset (see instructions in this README) and run the following command to generate the prototypes and evaluate the model: 
``python3 main.py --dataset CUB-200-2011 --epochs_pretrain 0 --batch_size 64 --freeze_epochs 10 --epochs 0 --log_dir ./runs/pipnet_cub --state_dict_dir_net ./pipnet_cub_trained``. Update the path of ``--state_dict_dir_net`` to the checkpoint if needed.
- PIP-Net with the ResNet50 backbone trained on the birds CUB-200-2011 dataset is available [for download here](https://drive.google.com/file/d/1zI1bcEXDsp8eN20msSiySo6UHD9y_bgw/view)  (280MB). Use ``--net resnet50``.
- PIP-Net with the ConvNext backbone (recommended) trained on the CARS dataset is available [for download here](https://drive.google.com/file/d/1JQNbhzw6s7yJsd_3--hCAReGkbT9PRlP/view)  (320MB). Use ``--dataset CARS``.
- PIP-Net with the ResNet50 backbone trained on the CARS dataset is available [for download here](https://drive.google.com/file/d/15t_nIjqR6m-dRFljqi-ntyv-gr4TIF7m/view)  (280MB).

### Data
The code can be applied to any imaging classification data set, structured according to the [Imagefolder format](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder): 

>root/class1/xxx.png  <br /> root/class1/xxy.png  <br /> root/class2/xyy.png <br /> root/class2/yyy.png

Add or update the paths to your dataset in ``util/data.py``. 

For preparing [CUB-200-2011]([http://www.vision.caltech.edu/visipedia/CUB-200-2011.html](https://www.vision.caltech.edu/datasets/cub_200_2011/)) with 200 bird species, use `util/preprocess_cub.py`. For [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) with 196 car types, use the [Instructions of ProtoTree](https://github.com/M-Nauta/ProtoTree/blob/main/README.md#preprocessing-cub).

### Interpreting the Results
During training, various files will be created in your ``--log_dir``:

- **``log_epoch_overview.csv``** keeps track of the training progress per epoch. It contains accuracies, the number of prototypes, loss values etc. In case of a 2-class task, the third value is F1-score, otherwise this is top5-accuracy. 
- **``out.txt``** collects the standard output from print statements. Its most relevant content is:
    - More performance metrics are printed, such as sparsity ratio. In case of a 2-class task, it also shows the sensitivity, specificity, confusion matrix, etc.
    - At the end of the file, after training, the relevant prototypes per class are printed. E.g., ``Class 0 has 5 relevant prototypes: [(prototype_id, class weight), ...]''. This information thus shows the learned scoring sheet of PIP-Net.
- **``tqdm.txt``** contains the progress via progress bar package [tqdm](https://tqdm.github.io/). Useful to see how long one epoch will take, and how the losses evolve. Errors are also printed here.
- **``metadata``** folder logs the provided arguments.
- **``checkpoints``** folder contains state_dicts of the saved models. 
- **Prototype visualisations** After training, various folders are created to visualise the learned reasoning of PIP-Net.
    - ``visualised_pretrained_prototypes_topk`` visualises the top-10 most similar image patches per prototype after the pretraining phase. Each row in ``grid_topk_all`` corresponds to one prototype. The number corresponds with the index of the prototype node, starting at 0.
    - ``visualised_prototypes_topk`` visualises the top-10 most similar image patches after the full (first and second stage) training. Prototypes that are not relevant to any class (all weights are zero) are excluded.
    - ``visualised_prototypes`` is a more extensive visualisation of the prototypes learned after training PIP-Net. The ``grid_xxx.png`` images show all image patches that are similar to prototype with index ``xxx``. The number of image patches (or the size of the png file) already gives an indication how often this prototype is found in the training set. If you want to know where these image patches come from (to see some more context), you can open the corresponding folder ``prototype_xxx``. Each image contains a yellow square indicating where prototype ``xxx`` was found, coresponding with an image patch in ``grid_xxx.png``. The file name is ``pxxx_imageid_similarityscore_imagename_rect.png``.
    - ``visualization_results`` (or other ``--dir_for_saving_images``) contains predictions including local explanations for test images. A subfolder corresponding to a test image contains the test image itself, and folders with predicted classes: ``classname_outputscore``. In such a class folder, it is visualised where which prototypes are detected: ``muliplicationofsimilarityandweight_prototypeindex_similarityscore_classweight_rect_or_patch.png``.

### Hyperparameter FAQ
* **What is the best number of epochs for my dataset?**
The right number of epochs (`--epochs` and `--epochs_pretrain`) will depend on the data set size and difficulty of the classification task. Hence, tuning the parameters might require some trial-and-error. You can start with the default values. For datasets of different sizes, we recommend to set the number of epochs such that the number of iterations (i.e., weight updates) during the second training state is around 10,000 (rule of thumb). Hence, epochs = 10000 / (num_images_in_trainingset / batch_size). The number of iterations for one epoch is easily found in ``tqdm.txt``. Similarly, the number of pretraining epochs `--epochs_pretrain` can be set such that there are 2000 weight updates. 

* **I have CUDA memory issues, what can I do?** PIP-Net is designed to fit onto one GPU. If your GPU has less CUDA memory, you have the following options: 1) reduce your batch size `--batch_size` or `--batch_size_pretrain`. Set it as large as possible to still fit in CUDA memory. 2) freeze more layers of the CNN backbone. Rather than optimizing the whole CNN backbone from `--freeze_epochs` onwards, you could keep the first layers frozen during the whole training process. Adapt the code around line 200 in `util/args.py` as indicated in the comments there. Alternatively, set `--freeze_epochs` equal to `--epochs`. 3) Use ``--net convnext_tiny_13`` instead of the default ``convnext_tiny_26`` to make training faster and more efficient. The potential downside is that the latent output grid is less fine-grained and could therefore impact prototype localization, but the impact will depend on your data and classification task.  

### Reference and Citation
Please refer to our work when using or discussing PIP-Net:

```
Meike Nauta, Jörg Schlötterer, Maurice van Keulen, Christin Seifert (2023). “PIP-Net: Patch-Based Intuitive Prototypes for Interpretable Image Classification.” IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
```

BibTex citation:
```
@article{nauta2023pipnet,
  title={PIP-Net: Patch-Based Intuitive Prototypes for Interpretable Image Classification},
  author={Nauta, Meike and Schlötterer, Jörg and van Keulen, Maurice and Seifert, Christin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```



