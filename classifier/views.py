from django.shortcuts import render
import torch
from rest_framework.views import APIView
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.exceptions import ParseError
from .models import ImageClassificationModel
from .serializers import ImageClassificationModelSerializer
from torchvision import models, datasets, transforms
import os
from rest_framework.decorators import action
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import random
import base64
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights
from django.template import loader
from django.http import HttpResponse


def home(request):
  template = loader.get_template('home.html')
  return HttpResponse(template.render())


root_dir = r"\datasets\datasets"  # data directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class-to-label mapping
class_to_label = {
    0: "Prudence",
    1: "Stranger",
}

class ImageClassificationModelViewSet(viewsets.ViewSet):
    queryset = ImageClassificationModel.objects.all()
    serializer_class = ImageClassificationModelSerializer
    
    @action(detail=False, methods=['post'])
    def train_model(self, request):
        NUM_WORKERS = os.cpu_count()
        weights = models.EfficientNet_B0_Weights
        batch_size = 32

        simple_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_data = datasets.ImageFolder(root_dir+'/train/', transform=simple_transform)
        test_data = datasets.ImageFolder(root_dir+'/val/', transform=simple_transform)

        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        model = models.efficientnet_b0(weights=weights).to(device)
        for param in model.features.parameters():
            param.requires_grad = False

        model.classifier[1] = nn.Linear(1280, 4).to(device)
        summary(model, (32, 3, 224, 224))

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        def train_step(model, dataloader, loss_fn, optimizer, device):
            model.train()
            train_loss, train_acc = 0, 0
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)

                loss = loss_fn(y_pred, y)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                train_acc += (y_pred_class == y).sum().item() / len(y_pred)
            train_loss = train_loss / len(dataloader)
            train_acc = train_acc / len(dataloader)
            return train_loss, train_acc

        def test_step(model, dataloader, loss_fn, device):
            model.eval()
            test_loss, test_acc = 0, 0
            with torch.inference_mode():
                for X, y in dataloader:
                    X, y = X.to(device), y.to(device)

                    test_pred_logits = model(X)
                    loss = loss_fn(test_pred_logits, y)
                    test_loss += loss.item()
                    test_pred_labels = test_pred_logits.argmax(dim=1)
                    test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
            test_loss = test_loss / len(dataloader)
            test_acc = test_acc / len(dataloader)
            return test_loss, test_acc

        EPOCHS = 15
        results = {"train_loss": [],
                   "train_acc": [],
                   "test_loss": [],
                   "test_acc": []
                   }

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
            test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        # Save model
        torch.save(model.state_dict(), r"C:\Users\phumziler\image_class_backend\imageclassificationproject\cnnapp")

        return HttpResponse("Model training complete. Check console for results.")

    @action(detail=False, methods=['post'])
    def make_prediction(self, request):
        # Define the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load trained model
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(1280, 4).to(device)
        model.load_state_dict(torch.load(r"C:\Users\phumziler\myProjects\ImageClassify\classifier\efficientnet_b0_prudence_stranger.pth", map_location=device))
        model.to(device)
        model.eval()

        image_data = request.data.get('image')
        image_file = request.FILES.get('image')

        if image_data and isinstance(image_data, str):
            try:
                format, imgstr = image_data.split(';base64,')
                # ext = format.split('/')[-1]
                img = base64.b64decode(imgstr)
                image = Image.open(BytesIO(img))
            except ValueError:
                return Response({'error': 'Invalid image data format or base64 encoding.'}, status=400)
            except UnidentifiedImageError:
                return Response({'error': 'Unable to identify or open the image file.'}, status=400)
            except Exception as e:
                return Response({'error': f'Unexpected error: {str(e)}'}, status=500)

            try:
                transformed_img = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])(image).unsqueeze(0).to(device)
            except Exception as e:
                return Response({'error': f'Error during image transformation: {str(e)}'}, status=500)
            
            with torch.no_grad():
                output = model(transformed_img)
                pred = torch.softmax(output, dim=1).squeeze(0)
                confidence, predicted_index = torch.max(pred, 0)

            predicted_class = class_to_label[int(predicted_index)]

            result = {
                'prediction': predicted_class,
                'confidence': confidence.item()
            }
            return Response(result)

        elif image_file:
            try:
                image = Image.open(image_file)
            except UnidentifiedImageError:
                return Response({'error': 'Unable to identify or open the image file.'}, status=400)
            except Exception as e:
                return Response({'error': f'Unexpected error: {str(e)}'}, status=500)

            try:
                transformed_img = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])(image).unsqueeze(0).to(device)
            except Exception as e:
                return Response({'error': f'Error during image transformation: {str(e)}'}, status=500)

            with torch.no_grad():
                output = model(transformed_img)
                pred = torch.softmax(output, dim=1).squeeze(0)
                confidence, predicted_index = torch.max(pred, 0)

            predicted_class = class_to_label[int(predicted_index)]

            result = {
                'prediction': predicted_class,
                'confidence': confidence.item()
            }
            return Response(result)
        
        else:
            return Response({'error': 'No image file found in request.'}, status=400)