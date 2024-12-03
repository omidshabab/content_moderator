import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict
import json
from app.api.models import ContentFlag  # Import the ContentFlag enum

class TextAnalyzer:
    def __init__(self):
        # Using DistilBERT fine-tuned on toxic comment classification
        self.model_name = "unitary/toxic-bert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Load custom sensitive words dictionary
        with open('models/ml/sensitive_words.json', 'r') as f:
            self.sensitive_words = json.load(f)

    def analyze(self, text: str) -> Dict:
        # Tokenize and predict
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        predictions = torch.sigmoid(outputs.logits)
        
        # Get prediction scores
        scores = predictions[0].detach().numpy()
        
        # Map scores to categories
        categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        model_scores = {cat: float(score) for cat, score in zip(categories, scores)}
        
        # Check for sensitive words
        found_sensitive = []
        for category, words in self.sensitive_words.items():
            if any(word in text.lower() for word in words):
                found_sensitive.append(category)
        
        # Calculate overall safety
        max_toxicity = max(scores)
        is_safe = max_toxicity < 0.5 and not found_sensitive
        confidence_score = float(1 - min(1, max_toxicity))
        
        # Determine appropriate flags
        flags = []
        if is_safe:
            flags.append(ContentFlag.SAFE)
        else:
            if max_toxicity >= 0.5:
                flags.append(ContentFlag.EXPLICIT)
            if found_sensitive:
                flags.append(ContentFlag.SENSITIVE)
            if model_scores['threat'] >= 0.5:
                flags.append(ContentFlag.SENSITIVE)
        
        # Construct response matching AnalysisResponse model
        return {
            "is_safe": is_safe,
            "flags": flags,  # List[ContentFlag]
            "confidence_score": confidence_score,
            "details": {  # Optional additional details
                "model_scores": model_scores,
                "found_sensitive_categories": found_sensitive
            }
        }