"""
    Main file for image classification. This takes in a user option for a model
    but defaults to AlexNet. It can run by default on the data 

    Written By: Nathan Nesbitt
    Date: 2020-12-11
"""

import torch


class Classifier:

    def __init__(self, dataset, model_name="alexnet"):
        """
            Initializes the object with the necessary information
        """
        super().__init__()
        self.dataset = dataset
        self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
        model.eval()

    def train(self):
        """
            Fine-Tunes the specified model using the dataset specified in the 
            previous section.
        """

    def test(self, input=None):
        """
            Tests the model created using `Classifier.train()`. Takes in an
            argument of 
        """

classifier = Classifier()
