# Content Moderator

## Introduction
The Content Moderator is a machine learning-based application designed to analyze and classify content as safe, explicit, or sensitive. It utilizes deep learning models for both image and text analysis, providing a robust solution for content moderation.

## Files and Folders Structure
```
.
├── app
│   ├── api
│   │   ├── models.py          # Contains data models and enums for analysis responses.
│   │   └── routes.py          # (Currently empty) Intended for defining API routes.
│   ├── config.py              # Configuration settings for the application.
│   └── main.py                # Entry point for the FastAPI application.
├── models
│   └── ml                     # Directory for machine learning models and related files.
│       ├── sensitive_words.json # JSON file containing sensitive words for text analysis.
│       └── nsfw_mobilenet.pth  # Pre-trained model weights for NSFW image classification.
├── requirements.txt           # List of required Python packages.
├── services                   # Contains service classes for analyzing images and text.
│   ├── content_classifier.py   # (Currently empty) Intended for content classification logic.
│   ├── image_analyzer.py       # Service for analyzing images using a pre-trained model.
│   └── text_analyzer.py        # Service for analyzing text using a pre-trained model.
├── training                   # Contains scripts for training models.
│   ├── train_nsfw_detector.py  # Script for training an NSFW image classification model.
│   └── train_nsfw_model.py     # Script for training a fine-tuned MobileNet model.
├── utils                      # Utility functions and scripts.
│   ├── download_yolo.py        # Script to download YOLO model files.
│   └── helpers.py              # (Currently empty) Intended for helper functions.
└── .gitignore                 # Specifies files and directories to ignore in version control.
```

### Explanation of Structure
- **app**: Contains the main application logic, including API routes and configuration.
- **models**: Stores machine learning models and related files, such as sensitive words for text analysis.
- **requirements.txt**: Lists all the dependencies required to run the application.
- **services**: Contains classes that handle the logic for analyzing images and text.
- **training**: Scripts for training the models used in the application.
- **utils**: Utility scripts for tasks like downloading necessary files.

## How to Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Packages**:
   Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

## How to Train the Model
### For Image Analysis
1. Prepare your dataset in the following structure:
   ```
   training/dataset/
   ├── train/
   │   ├── safe/    # Contains safe images
   │   └── unsafe/  # Contains NSFW images
   └── valid/
       ├── safe/
       └── unsafe/
   ```

2. Run the training script:
   ```bash
   python training/train_nsfw_model.py
   ```

### For Text Analysis
1. Create a JSON file named `sensitive_words.json` in the `models/ml` directory with the sensitive words you want to include.

2. The text analysis model is pre-trained and does not require additional training. You can directly use the `TextAnalyzer` class.

## How to Use
1. Start the FastAPI application:
   ```bash
   uvicorn app.main:app --reload
   ```

2. Use the following endpoints to analyze content:
   - **Text Analysis**:
     - **POST** `/analyze`
       - Request Body: 
         ```json
         {
           "content": "Your text here"
         }
         ```
     - Response: Returns analysis results including safety status and confidence score.

   - **Image Analysis**:
     - **POST** `/analyze/image`
       - Form Data: Upload an image file.
     - Response: Returns analysis results including safety status and confidence score.

## Conclusion
This Content Moderator application provides a comprehensive solution for analyzing and moderating content using advanced machine learning techniques. Follow the setup instructions to get started and utilize the provided endpoints for content analysis.
