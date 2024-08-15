# Image Classification & User Authentication Application

This repository contains an image classification application built using Django and Django REST Framework (DRF) for the backend API, with a frontend designed using Bootstrap, HTML, Axios, and JavaScript. The primary purpose of this application is to identify the correct user based on an image and log them into the system.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Setup Instructions](#setup-instructions)
- [API Endpoints](#api-endpoints)
- [Model Details](#model-details)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

This application uses a trained EfficientNet model for image classification. The main goal is to authenticate users based on their images and log them into the system securely. The application is designed to work seamlessly on a tablet or similar device, providing a user-friendly interface.

## Features

- **User Authentication**: Identifies and authenticates users based on images.
- **Responsive Frontend**: The frontend is designed using Bootstrap, ensuring a responsive and user-friendly experience across devices.
- **Real-time Processing**: Captures images using the device's camera and processes them in real-time for user authentication.
- **API Integration**: The backend API handles the image processing and classification, returning results to the frontend for further action.

## Architecture

- **Backend**: Built using Django and Django REST Framework.
- **Frontend**: Developed with Bootstrap, HTML, Axios, and JavaScript.
- **Model**: Utilizes the EfficientNet architecture for image classification.

## Setup Instructions

### Prerequisites

- Python 3.8+
- Django 3.2+
- Django REST Framework 3.12+
- Node.js (for frontend dependencies)
- A virtual environment tool like `venv` or `virtualenv`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/MissPhumy/Image-Classification.git
    cd Image-Classification
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Django development server:
    ```bash
    python manage.py runserver
    ```

5. Navigate to `http://localhost:8000` to view the application.

## API Endpoints

- **POST `/api/image-classification/make_prediction/`**: Receives an image, classifies it, and logs in the correct user if identified.
- **GET `/api/users/`**: Returns a list of users (for admin purposes).

## Model Details

The application uses the EfficientNet architecture for image classification. This model is known for its efficiency and accuracy in processing images, making it well-suited for real-time user identification.

## Usage

1. Open the application in a browser.
2. Use the camera to capture an image.
3. The image is sent to the backend API, which processes it using the EfficientNet model.
4. If the user is identified, they are logged into the system.

## Future Improvements

- Implement multi-user support with role-based access control.
- Enhance the model with transfer learning to improve accuracy.
- Add support for more complex authentication scenarios, such as multi-factor authentication.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
