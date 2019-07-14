import torch
import copy
from torch import nn
from torchvision.models import resnet50


class ResnetModel:
    def __int__(self, pretrained=False):
        self.model = resnet50(pretrained=pretrained)

    def train_model(self, dataloaders, optimizer, loss_func, device, num_epochs=10):
        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print("Epochs {}/ {}".format(epoch, num_epochs - 1))
            print("-"*10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train() # Set model to training mode
                else:
                    self.model.eval() # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # set the parameter gradients to zero
                    optimizer.zerzo_grad()

                    # forward pass
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate the loss
                        outputs = self.model(inputs)
                        loss = loss_func(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward step + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print("{} Loss: {:.4f} | Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'test':
                    val_acc_history.append(epoch_acc)

            print()

        self.model.load_state_dict(best_model_wts)
        return self.model, val_acc_history

    @classmethod
    def set_parameter_requires_grad(cls, model, feature_extraction):
        if feature_extraction:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self, num_classes, feature_extraction):
        # Initialize these variables which will be set in this if statement.

        self.set_parameter_requires_grad(self.model, feature_extraction)
        num_filters = self.model.fc.in_features
        self.model.fc = nn.Linear(num_filters, num_classes)
        input_size = 224

        return self.model, input_size
