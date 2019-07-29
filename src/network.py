import torch
import copy
from torch import nn
from torch import optim
from torchvision.models import resnet50


class ResnetModel:
    def __init__(self, pretrained=False):
        self.model = resnet50(pretrained=pretrained)

    def train(self, dataloaders, optimizer, loss_func, device, scheduler,
              num_epochs=10):

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        self.model.to(device)

        for epoch in range(num_epochs):
            print("Epochs {}/ {}".format(epoch, num_epochs - 1))
            print("-"*10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    # Set model to training mode
                    scheduler.step()
                    self.model.train()
                else:
                    # Set model to evaluate mode
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # set the parameter gradients to zero
                    optimizer.zero_grad()

                    # forward pass
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate the loss
                        outputs = self.model(inputs)
                        loss = loss_func(outputs, labels)

                        _, predictions = torch.max(outputs, 1)

                        # backward step + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(predictions == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} | Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            print()

        self.model.load_state_dict(best_model_wts)

        return self.model, val_acc_history

    def set_parameter_requires_grad(self, feature_extraction):

        if feature_extraction:
            for param in self.model.parameters():
                param.requires_grad = False

    def initialize_model(self, num_classes, feature_extraction=True):

        self.set_parameter_requires_grad(feature_extraction)
        num_filters = self. model.fc.in_features
        self.model.fc = nn.Linear(num_filters, num_classes)
        input_size = 224

        return self.model, input_size

    def optimizer(self):
        parameter_list = [
            {"params": self.model.fc.parameters(), "lr": 1e-3}
        ]
        optimizer = optim.Adam(params=parameter_list, lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5)

        loss_func = nn.CrossEntropyLoss()

        return optimizer, scheduler, loss_func

    def test(self, dataloader, device):

        phase = 'test'
        predictions = []
        for inputs in dataloader[phase]:
            inputs = inputs.to(device)
            outputs = self.model(inputs)

            _, preds = torch.max(outputs, 1)
            predictions.extend(preds)

        return predictions
