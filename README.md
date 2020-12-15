# NezNet CNN
This is a python implementation of transfer learning on AlexNet.

## Data

The data used is the [Intel Image Classification dataset ](https://www.kaggle.com/puneet6060/intel-image-classification)
which consists of a `train`, `test`, and `predict` folders, all containing 
images of

1. Buildings
2. Forests
3. Glaciers
4. Mountains
5. Seas
6. Streets

The test and train have labeled folders for each of the image sets (which is 
necessary to train and test the model) and the predict simply has a folder full
of images that are not labeled.

## Setup

We need to start off by cloning the images into the `data` directory. The 
images are approximately 330MB which is not super large, but they are provided
in a zip file which needs to be unzipped. This will produce the following file
structure

```
.
├── data
│   ├── seg_pred
│   │   └── seg_pred
│   ├── seg_test
│   │   └── seg_test
│   │       ├── buildings
│   │       ├── forest
│   │       ├── glacier
│   │       ├── mountain
│   │       ├── sea
│   │       └── street
│   └── seg_train
│       └── seg_train
│           ├── buildings
│           ├── forest
│           ├── glacier
│           ├── mountain
│           ├── sea
│           └── street
```

So we don't destroy the global python environment please create a virtual 
environment. This can be achieved using 

```
python3 -m venv venv
```

You can then use the virtual environment (in linux) by running

```
source /venv/bin/activate
```

Then we need to install the requirements for deep learning, which are contained
within the `requirements.txt` file. This can be done by running:

```
pip install -r requirements.txt
```

The final result should be a structure like the following:

```
.
├── data
│   ├── seg_pred
│   │   └── seg_pred
│   ├── seg_test
│   │   └── seg_test
│   │       ├── buildings
│   │       ├── forest
│   │       ├── glacier
│   │       ├── mountain
│   │       ├── sea
│   │       └── street
│   └── seg_train
│       └── seg_train
│           ├── buildings
│           ├── forest
│           ├── glacier
│           ├── mountain
│           ├── sea
│           └── street
├── main.py
├── README.md
└── requirements.txt
```

## Running

To run the model simply run the `classification.py` file. I have created a basic run 
script which downloads the main AlexNet model, and fine-tunes the model on our
image classification task (identifying the landscape in the image). This will 
produce an output like the following

```
Epoch | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
-----------------------------------------------------------------
  1   |   0.6486   |     0.7587     |   0.5194  |     0.7970    |
  2   |   0.5188   |     0.8072     |   0.3315  |     0.8887    |
  3   |   0.4597   |     0.8292     |   0.3483  |     0.8690    |
  4   |   0.4379   |     0.8353     |   0.3597  |     0.8703    |
  5   |   0.4152   |     0.8453     |   0.2862  |     0.8967    |
  6   |   0.3990   |     0.8492     |   0.3078  |     0.8857    |
  7   |   0.3975   |     0.8529     |   0.2903  |     0.8990    |
  8   |   0.3781   |     0.8605     |   0.2619  |     0.9080    |
  9   |   0.3658   |     0.8644     |   0.2735  |     0.9027    |
  10  |   0.3647   |     0.8646     |   0.2900  |     0.8990    |
  11  |   0.3551   |     0.8658     |   0.2917  |     0.8900    |
  12  |   0.3474   |     0.8714     |   0.2569  |     0.9053    |
  13  |   0.3296   |     0.8774     |   0.2652  |     0.9043    |
  14  |   0.3315   |     0.8768     |   0.2396  |     0.9117    |
  15  |   0.3226   |     0.8762     |   0.2381  |     0.9103    |

Training complete in 16m 18s
Best Validation Accuracy: 0.9117
```

This will train and save the model in the models directory, you can continue to
use this model in further tasks by importing it. For now we simply use the 
created model that is stored in the object, and run some randomized tests on it, 
which can be visually checked.