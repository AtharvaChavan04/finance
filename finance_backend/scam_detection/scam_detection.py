import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re
import numpy as np
from torch.nn.functional import softmax
import pytesseract
from PIL import Image
import librosa
import speech_recognition as sr
import language_tool_python
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline as tf_pipeline

class EnhancedScamDetector:
    def __init__(self):
        try:
            # Text analysis models
            self.model_name = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
            
            # Financial sentiment analysis
            self.financial_sentiment = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            
            # General sentiment analysis
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Language checking
            self.language_tool = language_tool_python.LanguageTool('en-US')
            
            # NLP for language analysis
            self.nlp = spacy.load('en_core_web_sm')
            
            # Initialize OCR
            self.ocr = pytesseract
            
            # Initialize speech recognizer
            self.speech_recognizer = sr.Recognizer()
            
            # Initialize common scam patterns
            self.scam_patterns = {
                'urgency': [
                    r'urgent',
                    r'immediate action',
                    r'act now',
                    r'limited time',
                    r'expires soon',
                    r'last chance'
                ],
                'financial': [
                    r'bank account.*compromised',
                    r'verify.*account',
                    r'win.*prize',
                    r'lottery.*winner',
                    r'investment opportunity',
                    r'guaranteed returns'
                ],
                'sensitive_info': [
                    r'confirm.*details',
                    r'verify.*identity',
                    r'send.*password',
                    r'provide.*card',
                    r'social security',
                    r'mother\'s maiden name'
                ],
                'suspicious_contact': [
                    r'contact.*immediately',
                    r'call.*number',
                    r'click.*link',
                    r'reply.*urgent',
                    r'foreign prince',
                    r'overseas beneficiary'
                ],
                'emotional_manipulation': [
                    r'help.*urgent',
                    r'family emergency',
                    r'life.*death',
                    r'desperate',
                    r'please help'
                ]
            }
            
            # Compile regex patterns
            self.compiled_patterns = {
                category: [re.compile(pattern, re.IGNORECASE) 
                          for pattern in patterns]
                for category, patterns in self.scam_patterns.items()
            }
            
        except Exception as e:
            print(f"Error initializing EnhancedScamDetector: {str(e)}")
            print("Some features may be unavailable...")

    def analyze_text(self, text):
        """
        Comprehensive text analysis including sentiment, patterns, and language
        """
        analysis = {
            'risk_score': 0.0,
            'risk_factors': [],
            'detected_patterns': {},
            'language_analysis': {},
            'sentiment_analysis': {},
            'grammar_check': {},
            'recommendation': ''
        }
        
        # Pattern-based detection
        for category, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                if pattern.search(text):
                    matches.append(pattern.pattern)
            
            if matches:
                analysis['detected_patterns'][category] = matches
                analysis['risk_score'] += 0.25

        # Sentiment analysis
        try:
            # Financial sentiment
            financial_sentiment = self.financial_sentiment(text)[0]
            analysis['sentiment_analysis']['financial'] = financial_sentiment
            
            # General sentiment
            vader_sentiment = self.sentiment_analyzer.polarity_scores(text)
            analysis['sentiment_analysis']['general'] = vader_sentiment
            
            # Adjust risk score based on negative sentiment
            if vader_sentiment['compound'] < -0.5:
                analysis['risk_score'] += 0.15
        except Exception as e:
            print(f"Sentiment analysis error: {str(e)}")

        # Language analysis
        doc = self.nlp(text)
        analysis['language_analysis'] = {
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'urgency_indicators': self._analyze_urgency(doc),
            'emotional_manipulation': self._analyze_emotional_content(doc)
        }

        # Grammar check
        matches = self.language_tool.check(text)
        analysis['grammar_check'] = {
            'error_count': len(matches),
            'errors': [str(match) for match in matches[:5]]  # Show first 5 errors
        }
        
        # Adjust risk score based on grammar errors
        if len(matches) > 5:
            analysis['risk_score'] += 0.1

        analysis['risk_score'] = min(analysis['risk_score'], 1.0)
        analysis['recommendation'] = self._generate_recommendation(analysis['risk_score'])
        
        return analysis

    def analyze_image(self, image_path):
        """
        Analyze image for scam indicators
        """
        try:
            image = Image.open(image_path)
            
            analysis = {
                'text_content': {},
                'image_characteristics': {},
                'combined_risk': 0.0
            }
            
            # Extract text using OCR
            extracted_text = self.ocr.image_to_string(image)
            if extracted_text.strip():
                analysis['text_content'] = self.analyze_text(extracted_text)
            
            # Basic image characteristics
            analysis['image_characteristics'] = {
                'size': image.size,
                'format': image.format,
                'mode': image.mode
            }
            
            # Calculate combined risk score
            text_risk = analysis['text_content'].get('risk_score', 0) if extracted_text.strip() else 0
            analysis['combined_risk'] = text_risk
            
            return analysis
            
        except Exception as e:
            return {
                'error': f"Error analyzing image: {str(e)}",
                'risk_score': 0.5
            }

    def _analyze_urgency(self, doc):
        """
        Analyze text for urgency indicators
        """
        urgency_indicators = []
        urgency_words = ['immediately', 'urgent', 'hurry', 'quick', 'fast', 'now']
        
        for token in doc:
            if token.text.lower() in urgency_words:
                urgency_indicators.append(token.text)
                
        return urgency_indicators

    def _analyze_emotional_content(self, doc):
        """
        Analyze text for emotional manipulation
        """
        emotional_indicators = []
        emotional_words = ['please', 'help', 'desperate', 'need', 'emergency', 'critical']
        
        for token in doc:
            if token.text.lower() in emotional_words:
                emotional_indicators.append(token.text)
                
        return emotional_indicators

    def _generate_recommendation(self, risk_score):
        """
        Generate detailed recommendation based on risk score
        """
        if risk_score >= 0.8:
            return ("HIGH RISK: This content shows strong indicators of being a scam. "
                   "Multiple risk factors detected. Do not respond or provide any information. "
                   "Report to relevant authorities if necessary.")
        elif risk_score >= 0.5:
            return ("MODERATE RISK: This content shows some suspicious characteristics. "
                   "Verify through official channels before taking any action. "
                   "Do not share sensitive information.")
        else:
            return ("LOW RISK: While this content appears legitimate, always be cautious "
                   "with sensitive information. Verify sender identity when in doubt.")

def main():
    detector = EnhancedScamDetector()
    
    print("Enhanced Scam Detection System")
    print("-" * 50)
    print("\nChoose analysis type:")
    print("1. Text Analysis")
    print("2. Image Analysis")
    
    choice = input("\nEnter your choice (1 or 2): ")
    
    if choice == '1':
        text = input("\nEnter the text to analyze: ")
        result = detector.analyze_text(text)
        
    elif choice == '2':
        image_path = input("\nEnter the path to the image file: ")
        result = detector.analyze_image(image_path)
        
    else:
        print("Invalid choice!")
        return
    
    print("\nAnalysis Results:")
    print("-" * 50)
    for key, value in result.items():
        print(f"\n{key.replace('_', ' ').title()}:")
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"  - {k}: {v}")
        else:
            print(f"  {value}")

if __name__ == "__main__":
    main()