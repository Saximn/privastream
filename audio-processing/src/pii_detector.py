"""
PII Detection processor using trained DeBERTa models for livestream pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    AutoModelForTokenClassification
)

from pipeline_types import (
    TranscriptionResult,
    PIIDetection,
    PIIType,
    RedactionResult
)
from model_multi_dropouts import CustomModel


class PIIDetector:
    """
    PII Detection processor using trained DeBERTa models for real-time text analysis.
    """
    
    def __init__(
        self,
        model_path: str = "./models/",
        tokenizer_name: str = "microsoft/deberta-v3-large",
        device: str = "cuda",
        confidence_threshold: float = 0.7,
        max_length: int = 512,
        stride: int = 128
    ):
        """
        Initialize the PII detector.
        
        Args:
            model_path: Path to the trained model directory
            tokenizer_name: Name/path of the tokenizer
            device: Device to run inference on (cuda, cpu)
            confidence_threshold: Minimum confidence for PII detection
            max_length: Maximum sequence length for tokenization
            stride: Stride for sliding window on long texts
        """
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self.stride = stride
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load tokenizer
        self.logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            add_prefix_space=True
        )
        
        # PII label mapping
        self.id2label = {
            0: "O",
            1: "B-EMAIL",
            2: "B-ID_NUM", 
            3: "B-NAME_STUDENT",
            4: "B-PHONE_NUM",
            5: "B-STREET_ADDRESS",
            6: "B-URL_PERSONAL",
            7: "B-USERNAME",
            8: "I-ID_NUM",
            9: "I-NAME_STUDENT", 
            10: "I-PHONE_NUM",
            11: "I-STREET_ADDRESS",
            12: "I-URL_PERSONAL"
        }
        
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # Load model (your trained DeBERTaV3 ensemble)
        self.model = self._load_model()
        
        # Apply optimizations for low latency
        self._optimize_for_latency()
        
        # Statistics
        self.stats = {
            'processed_texts': 0,
            'detected_pii': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'errors': 0
        }
        
        # Initialize post-processing rules (from your sophisticated approach)
        self._init_post_processing_rules()
        
        self.logger.info(f"PII detector initialized successfully with {self.device} optimization")
    
    def _load_model(self):
        """Load the trained DeBERTa model."""
        try:
            # Check if model directory exists
            model_path_obj = Path(self.model_path)
            if not model_path_obj.exists():
                self.logger.warning(f"Model directory {self.model_path} does not exist. Creating it.")
                model_path_obj.mkdir(parents=True, exist_ok=True)
            
            # Try to load config
            config_path = model_path_obj / "config.json"
            try:
                if config_path.exists():
                    config = AutoConfig.from_pretrained(
                        self.model_path,
                        num_labels=len(self.id2label),
                        id2label=self.id2label,
                        label2id=self.label2id
                    )
                    self.logger.info("Loaded model config from local path")
                else:
                    raise FileNotFoundError("Config file not found")
            except Exception as e:
                # Create config if not found
                self.logger.warning(f"Model config not found ({e}), creating default config from tokenizer")
                config = AutoConfig.from_pretrained(
                    self.tokenizer_name,
                    num_labels=len(self.id2label),
                    id2label=self.id2label,
                    label2id=self.label2id
                )
            
            # Load custom model
            custom_model_files = list(model_path_obj.glob("*.bin")) + list(model_path_obj.glob("*.safetensors"))
            try:
                if custom_model_files:
                    model = CustomModel.from_pretrained(
                        self.model_path,
                        config=config,
                        ignore_mismatched_sizes=True
                    )
                    self.logger.info(f"Loaded custom model from {self.model_path}")
                else:
                    raise FileNotFoundError("No custom model files found")
            except Exception as e:
                # Fallback to standard model
                self.logger.warning(f"Custom model not found ({e}), using standard AutoModel")
                try:
                    model = AutoModelForTokenClassification.from_pretrained(
                        self.tokenizer_name,
                        config=config,
                        ignore_mismatched_sizes=True
                    )
                    self.logger.info(f"Loaded standard model: {self.tokenizer_name}")
                except Exception as fallback_e:
                    self.logger.error(f"Failed to load any model: {fallback_e}")
                    raise
            
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better PII detection.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Basic text cleaning while preserving PII patterns
        text = text.strip()
        
        # Normalize whitespace but keep structure
        text = re.sub(r'\s+', ' ', text)
        
        # Handle common transcription artifacts
        text = text.replace("...", " ")
        text = text.replace(" - ", " ")
        
        return text
    
    def tokenize_text(
        self, 
        text: str
    ) -> Dict[str, Union[torch.Tensor, List[int]]]:
        """
        Tokenize text with sliding window for long texts.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dictionary containing tokenized inputs and metadata
        """
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Tokenize with sliding window if text is too long
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        
        # Move to device
        tokens = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in tokens.items()}
        
        return tokens
    
    def detect_pii_in_text(
        self, 
        text: str,
        transcription_start_time: float = 0.0
    ) -> List[PIIDetection]:
        """
        Detect PII in a given text.
        
        Args:
            text: Input text to analyze
            transcription_start_time: Start time of the transcription for timestamp calculation
            
        Returns:
            List of PIIDetection objects
        """
        start_time = time.time()
        
        try:
            if not text.strip():
                return []
            
            # Tokenize text
            tokens = self.tokenize_text(text)
            
            # Perform inference with mixed precision
            with torch.no_grad():
                # Use automatic mixed precision for faster inference
                with torch.cuda.amp.autocast(enabled=getattr(self, 'use_amp', False)):
                    outputs = self.model(**{k: v for k, v in tokens.items() 
                                          if k in ['input_ids', 'attention_mask', 'token_type_ids']})
                
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_labels = torch.argmax(predictions, dim=-1)
                confidence_scores = torch.max(predictions, dim=-1)[0]
            
            # Extract PII detections
            detections = self._extract_detections(
                text=text,
                tokens=tokens,
                predictions=predicted_labels,
                confidence_scores=confidence_scores,
                transcription_start_time=transcription_start_time
            )
            
            # Apply sophisticated post-processing (your methodology)
            filtered_detections = self.apply_sophisticated_post_processing(detections, text)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['processed_texts'] += 1
            self.stats['detected_pii'] += len(filtered_detections)
            self.stats['total_processing_time'] += processing_time
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['processed_texts']
            )
            
            self.logger.debug(
                f"Detected {len(filtered_detections)} PII instances in {processing_time:.3f}s"
            )
            
            return filtered_detections
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Error detecting PII: {e}")
            return []
    
    def _extract_detections(
        self,
        text: str,
        tokens: Dict,
        predictions: torch.Tensor,
        confidence_scores: torch.Tensor,
        transcription_start_time: float
    ) -> List[PIIDetection]:
        """Extract PII detections from model predictions."""
        detections = []
        
        # Get offset mappings for character-level positions
        offset_mapping = tokens.get('offset_mapping', [None])[0]
        if offset_mapping is None:
            return detections
        
        # Convert predictions to CPU numpy
        predictions = predictions.cpu().numpy()
        confidence_scores = confidence_scores.cpu().numpy()
        
        # Process each sequence (in case of sliding window)
        for seq_idx in range(predictions.shape[0]):
            seq_predictions = predictions[seq_idx]
            seq_confidences = confidence_scores[seq_idx]
            seq_offset_mapping = offset_mapping if offset_mapping.dim() == 2 else offset_mapping[seq_idx]
            
            # Group consecutive B- and I- tags
            current_entity = None
            current_tokens = []
            
            for token_idx, (pred_label, confidence) in enumerate(zip(seq_predictions, seq_confidences)):
                label = self.id2label.get(pred_label, "O")
                
                if label.startswith("B-"):
                    # Start of new entity
                    if current_entity:
                        # Save previous entity
                        detections.append(self._create_detection(
                            current_entity, current_tokens, text, 
                            seq_offset_mapping, transcription_start_time
                        ))
                    
                    current_entity = label[2:]  # Remove "B-" prefix
                    current_tokens = [(token_idx, confidence)]
                    
                elif label.startswith("I-") and current_entity == label[2:]:
                    # Continuation of current entity
                    current_tokens.append((token_idx, confidence))
                    
                else:
                    # End of current entity or "O" tag
                    if current_entity:
                        detections.append(self._create_detection(
                            current_entity, current_tokens, text,
                            seq_offset_mapping, transcription_start_time
                        ))
                        current_entity = None
                        current_tokens = []
            
            # Handle entity at end of sequence
            if current_entity:
                detections.append(self._create_detection(
                    current_entity, current_tokens, text,
                    seq_offset_mapping, transcription_start_time
                ))
        
        return detections
    
    def _create_detection(
        self,
        entity_type: str,
        token_positions: List[Tuple[int, float]],
        text: str,
        offset_mapping: torch.Tensor,
        transcription_start_time: float
    ) -> PIIDetection:
        """Create a PIIDetection object from entity information."""
        
        # Calculate character positions
        start_token_idx = token_positions[0][0]
        end_token_idx = token_positions[-1][0]
        
        try:
            start_char = offset_mapping[start_token_idx][0].item()
            end_char = offset_mapping[end_token_idx][1].item()
        except:
            start_char = 0
            end_char = len(text)
        
        # Extract text
        pii_text = text[start_char:end_char].strip()
        
        # Calculate average confidence
        avg_confidence = np.mean([conf for _, conf in token_positions])
        
        # Estimate time positions (rough approximation)
        # This is a simple linear estimation - could be improved with word-level timestamps
        text_position_ratio = start_char / len(text) if len(text) > 0 else 0
        estimated_start_time = transcription_start_time + (text_position_ratio * 5.0)  # Assume 5s segments
        estimated_end_time = estimated_start_time + (len(pii_text.split()) * 0.4)  # ~0.4s per word
        
        # Map entity type to PIIType enum (handle BIO tags by removing prefixes)
        try:
            # Remove B- and I- prefixes for BIO tagging
            clean_entity_type = entity_type.replace("B-", "").replace("I-", "")
            pii_type = PIIType(clean_entity_type)
        except ValueError:
            # Fallback for unknown types
            pii_type = PIIType.OTHER
        
        return PIIDetection(
            pii_type=pii_type,
            text=pii_text,
            start_char=start_char,
            end_char=end_char,
            confidence=avg_confidence,
            start_time=estimated_start_time,
            end_time=estimated_end_time,
            word_indices=[idx for idx, _ in token_positions]
        )
    
    def redact_text(
        self, 
        text: str, 
        detections: List[PIIDetection],
        redaction_symbol: str = "[REDACTED]"
    ) -> str:
        """
        Redact PII from text based on detections.
        
        Args:
            text: Original text
            detections: List of PIIDetection objects
            redaction_symbol: Symbol to replace PII with
            
        Returns:
            Redacted text
        """
        if not detections:
            return text
        
        # Sort detections by start position (reverse order for replacement)
        sorted_detections = sorted(detections, key=lambda x: x.start_char, reverse=True)
        
        redacted_text = text
        for detection in sorted_detections:
            # Create type-specific redaction symbol
            type_specific_redaction = f"[{detection.pii_type.value}]"
            
            redacted_text = (
                redacted_text[:detection.start_char] + 
                type_specific_redaction + 
                redacted_text[detection.end_char:]
            )
        
        return redacted_text
    
    def process_transcription(
        self, 
        transcription: TranscriptionResult
    ) -> RedactionResult:
        """
        Process a transcription result to detect and redact PII.
        
        Args:
            transcription: TranscriptionResult to process
            
        Returns:
            RedactionResult with detections and redacted text
        """
        start_time = time.time()
        
        # Detect PII in transcription
        detections = self.detect_pii_in_text(
            transcription.text,
            transcription.start_time
        )
        
        # Redact text
        redacted_text = self.redact_text(transcription.text, detections)
        
        processing_time = time.time() - start_time
        
        return RedactionResult(
            original_text=transcription.text,
            redacted_text=redacted_text,
            detections=detections,
            segment_id=transcription.segment_id,
            processing_time=processing_time
        )
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'processed_texts': 0,
            'detected_pii': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'errors': 0
        }
    
    def cleanup(self):
        """Clean up resources."""
        # Clear CUDA cache if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("PII detector cleaned up")
    
    def _optimize_for_latency(self):
        """Apply optimizations for low-latency real-time processing."""
        try:
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # Disable gradient computation
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Enable mixed precision if using CUDA
            self.use_amp = torch.cuda.is_available() and self.device == "cuda"
            
            # Optimize for inference
            if hasattr(torch, 'compile') and self.device == "cuda":
                try:
                    # Use torch.compile for faster inference (PyTorch 2.0+)
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    self.logger.info("Applied torch.compile optimization")
                except Exception as e:
                    self.logger.warning(f"torch.compile failed: {e}")
            
            # Set optimized parameters
            self.max_length = min(self.max_length, 512)  # Reduce for speed
            self.stride = min(self.stride, 64)
            
            self.logger.info("Applied latency optimizations")
            
        except Exception as e:
            self.logger.warning(f"Some optimizations failed: {e}")
    
    def _init_post_processing_rules(self):
        """Initialize sophisticated post-processing rules from your methodology."""
        import re
        
        # Label-specific confidence thresholds (from your approach)
        self.label_thresholds = {
            'NAME_STUDENT': 0.75,
            'EMAIL': 0.80,
            'PHONE_NUM': 0.75,
            'STREET_ADDRESS': 0.70,
            'ID_NUM': 0.75,
            'USERNAME': 0.70,
            'URL_PERSONAL': 0.70,
            'O': 0.30
        }
        
        # Validation patterns (core rules from your approach)
        self.validation_patterns = {
            # Student names must be title case without digits/underscores
            'name_pattern': re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$'),
            # Phone numbers with 9+ digits get converted to ID_NUM
            'phone_digits': re.compile(r'\d'),
            # Email must contain @
            'email_at': re.compile(r'@'),
            # Username merging pattern
            'username_sequence': re.compile(r'(.*?)(\s*[-–—]\s*)+(.*)'),
        }
        
        self.logger.debug("Initialized post-processing rules")
    
    def apply_sophisticated_post_processing(self, detections: List[PIIDetection], text: str) -> List[PIIDetection]:
        """
        Apply your sophisticated post-processing rules for optimal accuracy.
        """
        filtered_detections = []
        
        for detection in detections:
            # Apply label-specific thresholds
            threshold = self.label_thresholds.get(
                detection.pii_type.value, 
                self.confidence_threshold
            )
            
            if detection.confidence < threshold:
                continue
            
            # Core rules from your methodology
            detection_text = detection.text.strip()
            
            # Student name cleanup
            if detection.pii_type == PIIType.NAME_STUDENT:
                if not self.validation_patterns['name_pattern'].match(detection_text):
                    self.logger.debug(f"Filtered invalid name: {detection_text}")
                    continue
            
            # Phone → ID rule (9+ digits)
            elif detection.pii_type == PIIType.PHONE_NUM:
                digit_count = len(self.validation_patterns['phone_digits'].findall(detection_text))
                if digit_count >= 9:
                    detection.pii_type = PIIType.ID_NUM
                    self.logger.debug(f"Converted phone to ID: {detection_text}")
            
            # Email validation
            elif detection.pii_type == PIIType.EMAIL:
                if not self.validation_patterns['email_at'].search(detection_text):
                    self.logger.debug(f"Filtered invalid email: {detection_text}")
                    continue
            
            # ID_NUM length validation (4-25 chars)
            elif detection.pii_type == PIIType.ID_NUM:
                if len(detection_text) < 4 or len(detection_text) > 25:
                    self.logger.debug(f"Filtered invalid ID length: {detection_text}")
                    continue
            
            # URL_PERSONAL minimum length (10 chars)
            elif detection.pii_type == PIIType.URL_PERSONAL:
                if len(detection_text) < 10:
                    self.logger.debug(f"Filtered short URL: {detection_text}")
                    continue
            
            filtered_detections.append(detection)
        
        # Apply document-wide consistency rule
        filtered_detections = self._apply_document_consistency(filtered_detections, text)
        
        # Apply username merging rule
        filtered_detections = self._merge_username_sequences(filtered_detections)
        
        return filtered_detections
    
    def _apply_document_consistency(self, detections: List[PIIDetection], text: str) -> List[PIIDetection]:
        """Apply document-wide consistency: if X is tagged as NAME_STUDENT once, tag all instances."""
        enhanced_detections = list(detections)
        
        # Find all confirmed names
        confirmed_names = set()
        for detection in detections:
            if detection.pii_type == PIIType.NAME_STUDENT and detection.confidence > 0.8:
                confirmed_names.add(detection.text.strip().lower())
        
        # Look for other instances in text
        for name in confirmed_names:
            import re
            pattern = re.compile(re.escape(name), re.IGNORECASE)
            for match in pattern.finditer(text):
                start_char = match.start()
                end_char = match.end()
                
                # Check if already covered
                already_covered = any(
                    d.start_char <= start_char < d.end_char or 
                    start_char <= d.start_char < end_char
                    for d in enhanced_detections
                )
                
                if not already_covered:
                    consistent_detection = PIIDetection(
                        pii_type=PIIType.NAME_STUDENT,
                        text=match.group(),
                        start_char=start_char,
                        end_char=end_char,
                        confidence=0.90,  # High confidence for consistency
                        start_time=0.0,
                        end_time=0.0,
                        word_indices=[]
                    )
                    enhanced_detections.append(consistent_detection)
        
        return enhanced_detections
    
    def _merge_username_sequences(self, detections: List[PIIDetection]) -> List[PIIDetection]:
        """Merge B-USERNAME – - – B-USERNAME sequences."""
        merged_detections = []
        i = 0
        
        while i < len(detections):
            current = detections[i]
            
            if current.pii_type == PIIType.USERNAME:
                # Look for merge opportunities
                merged_text = current.text
                merged_end = current.end_char
                j = i + 1
                
                # Look ahead for username sequence
                while j < len(detections) and j < i + 5:  # Limit lookahead
                    next_detection = detections[j]
                    if (next_detection.pii_type == PIIType.USERNAME and 
                        next_detection.start_char - merged_end <= 5):  # Small gap
                        merged_text += " " + next_detection.text
                        merged_end = next_detection.end_char
                        j += 1
                    else:
                        break
                
                if j > i + 1:  # We merged something
                    merged_detection = PIIDetection(
                        pii_type=PIIType.USERNAME,
                        text=merged_text.strip(),
                        start_char=current.start_char,
                        end_char=merged_end,
                        confidence=max(detections[i:j], key=lambda x: x.confidence).confidence,
                        start_time=current.start_time,
                        end_time=detections[j-1].end_time,
                        word_indices=[]
                    )
                    merged_detections.append(merged_detection)
                    i = j
                else:
                    merged_detections.append(current)
                    i += 1
            else:
                merged_detections.append(current)
                i += 1
        
        return merged_detections
