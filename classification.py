# --------------------------------------------------------------------
# Main file for image classification, this is a training and testing file for
# transfer learning for AlexNet on a dataset of 6 classes.
#
#   1. Buildings
#   2. Forests
#   3. Glaciers
#   4. Mountains
#   5. Seas
#   6. Streets
#
# It achieves approximately 91.5% at the peak of the model after 15 .
# After training the model can be found under `./models/`. The training process
# takes approximately 1 minute per epoch with a midrange GPU.
#
# Written By: Nathan Nesbitt
# Date: 2020-12-11
#
# (C) 2020 Nathan Nesbitt, Canada
# --------------------------------------------------------------------

import torch
from PIL import Image
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import copy
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import random


class Classifier:
    """
    A class for training and testing a CNN based on AlexNet for the classification
    of 6 landscapes: Buildings, Forests, Glaciers, Mountains, Seas, and Streets.

    """

    def __init__(self, dataset_location="data/", import_model=False):
        """
        Initializes the object with the necessary information to train the
        model. The default value for dataset_location is defined according to the
        README, but can be changed if the data is in a different location.

        Keyword arguments:
            dataset_location -- Where the data can be found for training/testing
            import_model -- Allows for skipping of training by importing model
        """
        super().__init__()

        self.classes = {
            0: "Building",
            1: "Forest",
            2: "Glacier",
            3: "Mountain",
            4: "Sea",
            5: "Street",
        }

        self.import_model = import_model

        # Size of the batch (impact train speed)
        self.batch_size = 8

        # Classes are the image classes that we are trying to identify
        self.num_classes = 6

        # Sets the input size for alexnet, which is 224*244
        self.input_size = 224

        # Sets the model to run on the GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # This creates the model and input size, changing out the 7th layer
        self.model = self.initialize_model()

        if not self.import_model:

            # Stochastic Gradient Descent
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

            self.criterion = nn.CrossEntropyLoss()

        self.dataset_location = dataset_location

        self.dataloaders = self.initialize_dataset(dataset_location)

    def initialize_model(self):
        """
        Since we are doing transfer learning we need to pop the last layer
        off of the model, and provide a layer at the end that will detect for
        our more limited set.
        """

        # Imports the pre-trained model
        model = torch.hub.load("pytorch/vision:v0.6.0", "alexnet", pretrained=True)

        # Resets the 7th later to classify for 8 classes
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)

        # Sends the model to either the GPU/CPU
        model = model.to(self.device)

        return model

    def convertToONNX(self):
        """
        Converts the model to a ONNX model so it can be later converted to
        tensorflow.
        """
        # Try to open the model
        if self.import_model:
            try:
                self.model.load_state_dict(torch.load("models/model.pth"))
            except FileNotFoundError as e:
                raise e

        # Convert the model to onnx
        dummy_input = torch.randn(self.batch_size, 3, 224, 224).to(self.device)
        torch.onnx.export(self.model, dummy_input, "models/model.onnx")

    def initialize_dataset(self, dataset_location: str):
        """
        Imports the data, resizes and turns into a DataLoader, at the
        moment it is not normalizing the data. Both the training and
        testing data is stored in the same `dataloaders` object, and can
        be referenced by their names `seg_train/seg_train` or
        `seg_test/seg_test`.

        Keyword arguments:
            dataset_location -- base directory for dataset
        """

        # Data transforms for both sets of data
        data_transforms = {
            "seg_train/seg_train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
            "seg_test/seg_test": transforms.Compose(
                [
                    transforms.Resize(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                ]
            ),
        }

        # This finds all of the images using the {label}/{image} format
        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(dataset_location, x), data_transforms[x]
            )
            for x in ["seg_train/seg_train", "seg_test/seg_test"]
        }

        # Loads them into dataloaders
        dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
            )
            for x in ["seg_train/seg_train", "seg_test/seg_test"]
        }

        return dataloaders

    def train(self, num_epochs=15):
        """
        Fine-Tunes the specified model using the dataset specified in the
        initializer.

        Keyword arguments:
            num_epochs -- Number of epochs (runs) the model will train on
        """

        start_time = time.time()

        # History
        self.val_acc_history = []

        # Makes a copy of the model, which at the moment is the default to 0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        # Prints out the epochs
        print("\nEpoch | Train Loss | Train Accuracy | Test Loss | Test Accuracy |")
        print("-" * 65)

        for epoch in range(num_epochs):
            # Initialize some variables for the losses and accuracies
            train_loss = 0
            test_loss = 0
            train_acc = 0
            test_acc = 0

            # Each epoch has a training and validation phase
            for phase in ["seg_train/seg_train", "seg_test/seg_test"]:

                # Sets the model to training/eval based on dataset
                if phase == "seg_train/seg_train":
                    self.model.train()
                else:
                    self.model.eval()

                # We set the model loss and corrections
                running_loss = 0.0
                running_corrects = 0

                # Iterate over data in test and train data
                for images, labels in self.dataloaders[phase]:

                    # Sends the images/labels to the GPU/CPU
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    # If training, track the history
                    with torch.set_grad_enabled(phase == "seg_train/seg_train"):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)

                        _, predictions = torch.max(outputs, 1)

                        # We only back propagate and step when training.
                        if phase == "seg_train/seg_train":
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(predictions == labels.data)

                # Calculates the loss and accuracy for this epoch
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(
                    self.dataloaders[phase].dataset
                )

                # Storing the accuracy and loss for each so we can print all
                # at once instead of one line at a time.
                if phase == "seg_test/seg_test":
                    test_acc = epoch_acc
                    test_loss = epoch_loss
                else:
                    train_acc = epoch_acc
                    train_loss = epoch_loss

                # Stores the model weights if this is the best model so far
                if phase == "seg_test/seg_test" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == "seg_test/seg_test":
                    self.val_acc_history.append(epoch_acc)
            print(
                "{:^6}| {:^10.4f} | {:^15.4f}| {:^10.4f}| {:^14.4f}|".format(
                    epoch + 1, train_loss, train_acc, test_loss, test_acc
                )
            )

        # Prints out the total training time
        time_elapsed = time.time() - start_time
        print(
            "\nTraining complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        # This is the best accuracy obtained by the model
        print("Best Validation Accuracy: {:.4f}".format(best_acc))

        # Load and save the best model
        self.model.load_state_dict(best_model_wts)
        try:
            os.mkdir("models")
        except FileExistsError:
            pass
        torch.save(self.model.state_dict(), "models/model.pth")

    def test(self, input_dir="seg_pred/seg_pred"):
        """
        Tests the model created using `Classifier.train()`. Takes in an optional
        argument for the input directory, but defaults to the expected directory
        for the prediction images from the original dataset.

        Keyword arguments:
            input_dir -- Directory with unlabeled images
        """

        # Define the transforms
        data_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        # Select a batch size of images randomly from the prediction folder
        random_images = random.sample(
            os.listdir(self.dataset_location + input_dir),
            self.batch_size,
        )

        # Try to open the model
        if self.import_model:
            try:
                self.model.load_state_dict(torch.load("models/model.pth"))
            except FileNotFoundError as e:
                raise e

        # We can now predict and print out the classification of some images
        self.model.eval()

        # Iterate over data in test and train data
        for image in random_images:
            image = Image.open(self.dataset_location + input_dir + "/" + image)
            transformed_image = data_transform(image)

            # Sends the images/labels to the GPU/CPU
            transformed_image = transformed_image.to(self.device)

            # Gets the outputs from the model
            outputs = self.model(transformed_image.unsqueeze(0))
            _, predictions = torch.max(outputs, 1)

            plt.figure()
            plt.imshow(image)
            plt.title(self.classes[predictions[0].item()])
        plt.show(block=True)


if __name__ == "__main__":
    classifier = Classifier()
    classifier.train()
    classifier.test()