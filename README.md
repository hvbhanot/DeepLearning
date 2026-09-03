# Deep Learning with PyTorch - A Complete Guide

106 annotated notebooks taking you from matrix multiplication to GANs and style
transfer. Every notebook opens with what it teaches and why it comes where it
does; every non-trivial cell carries a comment explaining what it does and what
to look for in the output.

**How to read this.** Follow the order below. Each notebook's header names its
prerequisite and the next step, so you can also just open one and follow the
chain. Notebooks marked *Code Challenge* are exercises that consolidate the
preceding ideas.

**Running it.** Everything is a Colab notebook. Most assume a GPU
(`device = 'cuda'`); a few hardcode it, so on a CPU machine change those to
`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. Datasets download
themselves on first run.

---

## The through-line

The course is one argument, repeated at increasing scale:

1. **A model is a function with tunable parameters.** Gradient descent tunes them.
2. **Architecture is a prior.** A fully connected layer assumes nothing; a
   convolution assumes locality; a recurrent layer assumes order. Matching the
   prior to the data is most of the work.
3. **A model handles the variation it was trained on, and nothing else.** This
   appears in `ShiftedMNIST`, the occlusion autoencoders, the missing-7 challenge
   and the RNN extrapolation - four different costumes, one lesson.
4. **Capacity has to be spent where it pays.** More parameters is not more
   accuracy; the parametric experiments throughout measure exactly where the
   returns stop.

---

## Curriculum

### Foundations
| # | Notebook | What it covers |
|---|---|---|
| 00 | `Math.ipynb` | Transpose, matmul, softmax, entropy, argmax, mean/variance - in NumPy and PyTorch |
| 01 | `GradientDesent.ipynb` | Gradient descent by hand in 1D and 2D; local minima, stationary points, adaptive learning rates |
| - | `Neural Network Backpropagation` | An MLP built from scratch in NumPy: forward pass, chain rule, weight updates, XOR |

### ANNs - the training loop
| # | Notebook | What it covers |
|---|---|---|
| 02 | `ANNs/ANNs.ipynb` | First models: regression (MSE) and binary classification (sigmoid + BCE) |
| 03 | `ANNs/MultilayerANNs.ipynb` | Depth, and the dead-unit failure a narrow bottleneck causes |
| 04 | `ANNs/NNClasses.ipynb` | `nn.Module` - the idiom every later model uses |
| 05 | `ANNs/NumNodes` | Counting parameters; deep vs wide |
| 06 | `ANNs/IrisANN` | Multiclass: raw logits + `CrossEntropyLoss` + `argmax` |
| 07 | `ANNs/DepthVsBreadth.ipynb` | The parametric-experiment pattern, on a heat map |
| 08 | `ANNs/CodeChallenge.ipynb` | Three-class classification and a width sweep |

### FNNs - real data
| # | Notebook | What it covers |
|---|---|---|
| 09 | `FNNs/MNIST.ipynb` | DataLoaders, train/test split, `.train()`/`.eval()`, the loss-pairing rule |
| 10 | `FNNs/BinaryMNIST.ipynb` | 1 bit per pixel is nearly enough - how much information the task needs |
| 11 | `FNNs/MNISTDepthVsBreadth.ipynb` | Depth x width on real data, scored on both splits |
| 12 | `FNNs/MNISToptimizers.ipynb` | SGD / RMSprop / Adam x six learning rates |
| 13 | `FNNs/MinMaxMNIST.ipynb` | Feature scaling, and reading weight histograms |
| 14 | `FNNs/ScrambledMNIST.ipynb` | Permute the pixels: accuracy unchanged. FNNs ignore geometry |
| 15 | `FNNs/ShiftedMNIST.ipynb` | Shift the test images: accuracy collapses. The case for CNNs |
| 16 | `FNNs/CodeChallengeMissing7.ipynb` | Confidently wrong on a class it never saw |

### Data, splits, regularization
| # | Notebook | What it covers |
|---|---|---|
| 17 | `More on Data/Anatomy_of_DL_TD.ipynb` | `Dataset` and `DataLoader`, slowly |
| 18 | `Cross Validation/CrossValidation.ipynb` | Train/test splits, and seeing overfitting for the first time |
| 19 | `Cross Validation/DataLoader.ipynb` | Split first, then batch |
| 20 | `More on Data/Importing_data&Saving_models.ipynb` | Four ways data arrives; `state_dict` |
| 21 | `More on Data/DepthVsBreadth.ipynb` | 80 layers, and why they will not train |
| 22 | `More on Data/FeatureAugmentation.ipynb` | Which new features can possibly help |
| 23 | `More on Data/Handeling_Unbalanced_Data_Exact_Replication.ipynb` | Duplicating data adds nothing; duplicate after splitting |
| 24 | `More on Data/Handeling_Unbalanced_Data_noise_augmentation.ipynb` | Copies with noise do help |
| 25 | `More on Data/CodeChallenge_UnbalancedData.ipynb` | The accuracy trap on imbalanced classes |
| 26 | `Regularization/Regularization_theory.ipynb` | `train()`/`eval()`, `no_grad()`, dropout scaling, L1 vs L2 |
| 27 | `Regularization/Mini_batch.ipynb` | What batch size actually controls |
| 28 | `Regularization/Dropuout.ipynb` | The inverted-U of dropout rate, and when it hurts |
| 29 | `Regularization/Weight_Regularization_(L2).ipynb` | `weight_decay` in one argument |
| 30 | `Regularization/Weight_Regularization_(L1).ipynb` | Sparsity, and adding a custom penalty by hand |

### Metaparameters
| # | Notebook | What it covers |
|---|---|---|
| 31 | `Metaparameters/Normalization.ipynb` | StandardScaler vs MinMaxScaler; fit on train only |
| 32 | `Metaparameters/ExploringWineDataSet.ipynb` | Parameters vs metaparameters; a batch-size sweep |
| 33 | `Metaparameters/Activation_Functions.ipynb` | Why ReLU won: saturation and vanishing gradients |
| 34 | `Metaparameters/ReLU_Varients.ipynb` | LeakyReLU, ReLU6, and dead units |
| 35 | `Metaparameters/LossFunctions.ipynb` | Every loss plotted; the pairing table; custom losses |
| 36 | `Metaparameters/BatchNorm.ipynb` | Normalising activations, not just inputs |
| 37 | `Metaparameters/SGDwithMomentum.ipynb` | EWMA from scratch, then momentum |
| 38 | `Metaparameters/RMSprop_and_Adam.ipynb` | 3 optimizers x 20 learning rates |
| 39 | `Metaparameters/AdamwithL2.ipynb` | Weight decay with Adam - and why AdamW exists |
| 40 | `Metaparameters/LearningRateDecay.ipynb` | StepLR and ExponentialLR; where `scheduler.step()` goes |
| 41 | `Metaparameters/CodeChallenge_PredictSugar.ipynb` | Regression, done properly - and a problem with no signal |

### Model evaluation
| # | Notebook | What it covers |
|---|---|---|
| 42 | `Model Evaluation/ARPF.ipynb` | Accuracy, recall, precision, F1 - and why accuracy alone lies |
| 43 | `Model Evaluation/ARPF_WineDataset.ipynb` | `classification_report`, confusion matrices |
| 44 | `Model Evaluation/Computation_Time.ipynb` | Timing training, and why GPU timers lie |
| 45 | `Model Evaluation/ARPF_MNIST.ipynb` | Multiclass metrics; reading a 10x10 confusion matrix |
| 46 | `Model Evaluation/CodeChallenge_UnbalncedMNIST.ipynb` | Row-normalised confusion matrices on rare classes |

### FNN milestone projects
| # | Notebook | What it covers |
|---|---|---|
| 47 | `FNN Milestone Projects/FNN_HeartDisease.ipynb` | Medical data, `pos_weight`, recall over accuracy |
| 48 | `FNN Milestone Projects/FNN_Adding Machine.ipynb` | Huber loss, R2, scaling the target |
| 49 | `FNN Milestone Projects/ProjectMissingDataInterPolationipynb` | Imputation as supervised learning |

### Weights and hardware
| # | Notebook | What it covers |
|---|---|---|
| 50 | `Running Models on GPUs/GPU_Implementation.ipynb` | Model and data on the same device |
| 51 | `Weight Initialization/WeightMatrixSizeExplanation.ipynb` | Weight shapes; reading and writing them |
| 52 | `Weight Initialization/BaseWeightInitailzation.ipynb` | Why zero init fails: the symmetry argument |
| 53 | `Weight Initialization/LearningRelatedWeightChanges.ipynb` | Stub - tracking weight movement per layer |
| 54 | `Weight Initialization/CodeChallenge_IndenticallyRandomWeights.ipynb` | Seeds: random but reproducible |
| 55 | `Weight Initialization/KiamingAndXaviarInitialization.ipynb` | Scaling initial variance by fan-in |
| 56 | `Weight Initialization/FreezingWeightsDuringTraining.ipynb` | `requires_grad` - the mechanism behind transfer learning |
| 57 | `Weight Initialization/CodeChallenge_Weight_Inits.ipynb` | Initial scale swept over five orders of magnitude |
| 58 | `Weight Initialization/CodeChallenge_KaimingVsXaivier.ipynb` | 100 runs each - comparing distributions, not single numbers |

### Convolution
| # | Notebook | What it covers |
|---|---|---|
| 59 | `Convolution and transformations/Convolutions_in_Code.ipynb` | Convolution by hand, in scipy, and in PyTorch |
| 60 | `Convolution and transformations/Conv2D_class.ipynb` | The five arguments of `nn.Conv2d` |
| 61 | `Convolution and transformations/CodeChallenge_Choosing_Parameters.ipynb` | The output-size formula |
| 62 | `Convolution and transformations/Max_Mean_Pooling.ipynb` | Max vs average pooling |
| 63 | `Convolution and transformations/Transpose_Conv.ipynb` | Upsampling, and checkerboard artifacts |
| 64 | `Convolution and transformations/Image_transformations.ipynb` | Preparation vs augmentation |
| 65 | `Convolution and transformations/Creating_Custom_DataLoaders.ipynb` | Writing a `Dataset` |

### CNNs
| # | Notebook | What it covers |
|---|---|---|
| 66 | `CNNs/CNN_MNIST.ipynb` | The CNN skeleton; global average pooling |
| 67 | `CNNs/CNN_MNIST_ROLLED.ipynb` | The shifted-MNIST rematch - and the CNN wins |
| 68 | `CNNs/EMNIST.ipynb` | Conv-BN-ReLU-Pool; affine augmentation; 26 classes |
| 69 | `CNNs/CNN_CIFAR_100.ipynb` | The full modern recipe: VGG blocks, AutoAugment, label smoothing, cosine annealing |
| 70 | `CNNs/Examining_Feature_Maps.ipynb` | What each layer actually responds to |
| 71 | `CNNs/CodeChallenge_Softcode_Internal_Parameters.ipynb` | Parameterising an architecture |
| 72 | `CNNs/CodeChallenge_VaryingNumber_of_Channels.ipynb` | How many channels are enough |
| 73 | `CNNs/CodeChallenge_How_Wide_FC_Layer.ipynb` | The classifier head barely matters |
| 74 | `CNNs/Gaussian_Blur.ipynb` | Position-independent classification |
| 75 | `CNNs/Discovering_Gaussian_Parameters.ipynb` | CNN regression: recovering generating parameters |
| 76 | `CNNs/CustomLossFunctions.ipynb` | L1, L2 and correlation losses compared |
| 77 | `CNNs/DoAutoEncoders_CleanGaussians_(CNN_AE).ipynb` | Does the bottleneck remove occlusions? |
| 78 | `CNNs/CodeChallenge_Cleaning_Occulsions_with_AEs.ipynb` | Supervised denoising on (degraded, clean) pairs |

### CNN milestone projects
| # | Notebook | What it covers |
|---|---|---|
| 79 | `CNN Milestone Projects/Fashion_MNIST.ipynb` | Where the confusion matrix matches human errors |
| 80 | `CNN Milestone Projects/CIFAR_10.ipynb` | Colour data, and the `permute` rule |
| 81 | `CNN Milestone Projects/CIFAR_10_AutoEncoder.ipynb` | Fully convolutional autoencoder; why MSE blurs |
| 82 | `CNN Milestone Projects/Psychometric_functions_in_CNNs.ipynb` | Probing a model with a controlled continuum |

### Transfer learning
| # | Notebook | What it covers |
|---|---|---|
| 83 | `Transfer Learning/TF_MNIST-_FMNIST.ipynb` | Copying weights by hand |
| 84 | `Transfer Learning/TF_ResNet-18.ipynb` | The four-step fine-tuning recipe on ImageNet weights |
| 85 | `Transfer Learning/TF_VGG-16.ipynb` | 138M parameters vs ResNet's 11M - and it is not better |
| 86 | `Transfer Learning/Pretraining_CIFAR-10.ipynb` | Self-supervised pretraining with an autoencoder |
| 87 | `Transfer Learning/Pretraining_with_AutoEncoders.ipynb` | Upsample+Conv decoders; matching output range |
| 88 | `Transfer Learning/CodeChallenge_Letters_to_Numbers.ipynb` | Partial transfer, done cleanly |

### Autoencoders
| # | Notebook | What it covers |
|---|---|---|
| 89 | `Autoencoders/DenoisingMNIST.ipynb` | Compression is denoising |
| 90 | `Autoencoders/AE_for_Occlusion.ipynb` | Inpainting is harder than filtering |
| 91 | `Autoencoders/Latent_Code_Of_MNIST.ipynb` | Inside the bottleneck; autoencoder vs PCA |
| 92 | `Autoencoders/AE_with_Tied_Weights.ipynb` | `nn.Parameter`, and weight tying as regularisation |
| 93 | `Autoencoders/CodeChallenge_LatentSpace.ipynb` | How big should the code be? |

### GANs
| # | Notebook | What it covers |
|---|---|---|
| 94 | `GANs/Linear_GANs_with_MNIST.ipynb` | Two networks, two optimizers, one adversarial loop |
| 95 | `GANs/Linear_GANs_FMNIST.ipynb` | Why a fully connected generator blurs |
| 96 | `GANs/CNN_GANs_Guassians.ipynb` | The DCGAN rules, tested on easy data |
| 97 | `GANs/CodeChallenge_Guassians_with_fewer_layers.ipynb` | How much architecture a GAN actually needs |
| 98 | `GANs/CNN_GANs_FMNIST.ipynb` | DCGAN on real images; the 28x28 awkwardness |
| 99 | `GANs/CNN_GANs_CIFAR-10.ipynb` | Ten visual modes from one unconditional generator |

### RNNs
| # | Notebook | What it covers |
|---|---|---|
| 100 | `RNNs/RNN_class_in_PyTorch.ipynb` | Hidden state, the `(seq, batch, features)` convention, why LSTM exists |
| 101 | `RNNs/Predicting_Alternating_Sequences.ipynb` | Choosing a metric that isolates what is learnable |
| 102 | `RNNs/Sine_Wave_Extrapolation.ipynb` | One-step prediction vs free-running generation |

### Style transfer
| # | Notebook | What it covers |
|---|---|---|
| 103 | `Style Transfer/Neural_Style_Transfer.ipynb` | The image is the parameter; Gram matrices as style |
| 104 | `Style Transfer/Style_transfer_with _AlexNet.ipynb` | The same algorithm on 5 layers instead of 16 |

### Reference
`CheatSheets/` - PDF summaries for metaparameters, model evaluation and
classification metrics.

---

## Rules worth memorising

**The training loop** - identical in every notebook:

```python
yhat = model(x)            # forward
loss = lossFunc(yhat, y)   # how wrong
optimizer.zero_grad()      # clear old gradients (PyTorch accumulates them)
loss.backward()            # autograd fills .grad
optimizer.step()           # apply the update
```

**Output and loss must match:**

| Task | Final layer | Loss | Targets |
|---|---|---|---|
| Regression | `Linear(n,1)`, no activation | `MSELoss` / `HuberLoss` | float `(N,1)` |
| Binary | `Linear(n,1)`, no activation | `BCEWithLogitsLoss` | float `(N,1)` |
| Binary (probabilities out) | `+ Sigmoid` | `BCELoss` | float `(N,1)` |
| Multiclass | `Linear(n,C)`, no activation | `CrossEntropyLoss` | **int64** `(N,)` |
| Multiclass (log-probs out) | `+ log_softmax` | `NLLLoss` | int64 `(N,)` |

A `Softmax` layer before `CrossEntropyLoss` is the most common PyTorch bug: it
trains, badly, and silently.

**Shapes.** Images are `(batch, channels, height, width)`; sequences are
`(seq_len, batch, features)` unless you pass `batch_first=True`. Convert numpy's
`(N,H,W,C)` with `.permute(0,3,1,2)` - never `view`, which reinterprets memory
and scrambles the image.

**Fit preprocessing on train only.** `scaler.fit_transform(train)`,
`scaler.transform(test)`. The scaler *learns* its statistics from whatever you
fit it on, so fitting on everything before splitting quietly leaks the test set
into training.

**Read two accuracy curves, not one.** Train alone cannot distinguish learning
from memorising. The gap between them is overfitting; the epoch where the test
curve stops improving is where you should have stopped.

**Plot before you tune.** A correlation matrix tells you whether the signal
exists at all (`CodeChallenge_PredictSugar`); a look at a few samples catches
orientation and scaling bugs no loss curve will ever reveal (`EMNIST`).

---

## Mistakes that don't announce themselves

Deep learning code fails quietly. A model with a swapped metric, a leaked test
set or a dropout layer that never runs trains happily and prints plausible
numbers. These are the failure modes this course puts you in a position to
notice - each one appears somewhere in the notebooks, with a comment where it
lands.

**The model looks fine, the numbers are wrong**
- `Softmax` before `CrossEntropyLoss`. It applies the softmax twice; training
  still runs, badly.
- sklearn metrics called as `(y_pred, y_true)`. Accuracy and F1 are symmetric,
  so nothing looks odd - but precision and recall have swapped places.
- Thresholding a logit at `0.5`. With `BCEWithLogitsLoss` the output is a logit,
  and `sigmoid(0) = 0.5`, so the boundary is `0`.
- Targets shaped `(N,)` against an `(N,1)` output. MSE broadcasts to an `(N,N)`
  matrix of differences and reports a meaningless loss.

**The comparison proves nothing**
- Building the model *outside* the training function during a sweep: run 2
  continues run 1's training, so you are comparing accumulated epochs, not
  settings.
- One run per configuration. Initialisation noise is often as large as the effect
  you are measuring - `CodeChallenge_KaimingVsXaivier` runs each condition 100
  times for exactly this reason.
- Reporting accuracy on imbalanced data. Predicting the majority class scores 80%
  on an 80/20 split. Use per-class recall, or a confusion matrix.

**The test set is not clean**
- `fit_transform` on test data, or scaling before splitting.
- Augmenting or duplicating before splitting, so near-copies straddle the split.
- Any threshold, mean or vocabulary computed on the full dataset.

**The code runs and does nothing**
- A layer created in `__init__` but never called in `forward`.
- `torch.no_grad()` without `with`.
- A transform pipeline defined but never attached to the Dataset.
- Layers held in a plain Python list instead of `nn.ModuleList` - they exist,
  they are never trained.
- `for x in xs: x = f(x)` - rebinding the loop variable changes no list.

**The bookkeeping lies**
- `losses[epoch] = loss` inside the batch loop records the last batch, not the
  epoch.
- Metrics appended outside the loop they belong to.
- Axis labels copy-pasted from the plot above.

**The augmentation changes the answer.** A mirrored shirt is a shirt; an
upside-down 6 is not a 6. Every augmentation has to leave the label true.

---

## Notes on running these

- Cell outputs were saved from earlier runs. Re-run a notebook to regenerate
  them rather than reading the stored numbers as current.
- Most notebooks assume a GPU. Where `'cuda'` is hardcoded, swap in
  `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
- `Weight Initialization/LearningRelatedWeightChanges.ipynb` is a stub; its
  header sketches the experiment if you want to write it.
- `Transfer Learning/TF_MNIST-_FMNIST.ipynb` keeps two training functions on
  purpose - the first fails on unscaled inputs, the second fixes it. The fix was
  the data, not the architecture, which is the point.
