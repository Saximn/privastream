"""
PII Detection processor using trained DeBERTa models for livestream pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
import re
from typing import Dict, List, Tuple, Optional, Union
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    AutoModelForTokenClassification
)

from .pipeline_types import (
    TranscriptionResult,
    PIIDetection,
    PIIType,
    RedactionResult
)
from .model_multi_dropouts import CustomModel


DEFAULT_PII_KEYWORDS = (
    "password",
    "passcode",
    "pin",
    "secret",
    "api key",
    "access token",
    "auth token",
    "private key",
    "security question",
    "mother's maiden name"
)


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
        stride: int = 128,
        enable_deterministic_detection: bool = True,
        deterministic_confidence: float = 0.98,
        pii_keywords: Optional[List[str]] = None
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
            enable_deterministic_detection: Whether to merge regex/checksum-based PII detections
            deterministic_confidence: Confidence assigned to deterministic recognizer matches
            pii_keywords: Keyword phrases to detect as sensitive PII context
        """
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self.stride = stride
        self.enable_deterministic_detection = enable_deterministic_detection
        self.deterministic_confidence = deterministic_confidence
        self.pii_keywords = pii_keywords or list(DEFAULT_PII_KEYWORDS)
        
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
        
        # Load model
        self.model = self._load_model()
        
        # Statistics
        self.stats = {
            'processed_texts': 0,
            'detected_pii': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'errors': 0
        }
        
        self.logger.info("PII detector initialized successfully")
    
    def _load_model(self):
        """Load the trained DeBERTa model."""
        try:
            # Try to load config
            config_path = f"{self.model_path}/config.json"
            try:
                config = AutoConfig.from_pretrained(
                    self.model_path,
                    num_labels=len(self.id2label),
                    id2label=self.id2label,
                    label2id=self.label2id
                )
            except:
                # Create config if not found
                self.logger.warning("Model config not found, creating default config")
                config = AutoConfig.from_pretrained(
                    self.tokenizer_name,
                    num_labels=len(self.id2label),
                    id2label=self.id2label,
                    label2id=self.label2id
                )
            
            # Load custom model
            try:
                model = CustomModel.from_pretrained(
                    self.model_path,
                    config=config,
                    ignore_mismatched_sizes=True
                )
                self.logger.info(f"Loaded custom model from {self.model_path}")
            except:
                # Fallback to standard model
                self.logger.warning("Custom model not found, using standard AutoModel")
                model = AutoModelForTokenClassification.from_pretrained(
                    self.tokenizer_name,
                    config=config,
                    ignore_mismatched_sizes=True
                )
            
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
        deterministic_detections = []
        
        try:
            if not text.strip():
                return []
            
            deterministic_detections = (
                self._detect_deterministic_pii(text, transcription_start_time)
                if self.enable_deterministic_detection
                else []
            )

            # Tokenize text
            tokens = self.tokenize_text(text)
            
            # Perform inference
            with torch.no_grad():
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
            
            # Merge deterministic recognizers with model detections, then filter by confidence threshold
            merged_detections = self._merge_detections(
                detections,
                deterministic_detections
            )
            filtered_detections = [
                detection for detection in merged_detections
                if detection.confidence >= self.confidence_threshold
            ]
            
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
            return [
                detection for detection in deterministic_detections
                if detection.confidence >= self.confidence_threshold
            ]

    def _detect_deterministic_pii(
        self,
        text: str,
        transcription_start_time: float
    ) -> List[PIIDetection]:
        """Detect high-precision deterministic PII patterns."""
        detections = []

        pattern_specs = [
            (PIIType.EMAIL, re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')),
            (PIIType.PHONE_NUM, re.compile(r'(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)')),
            (PIIType.OTHER, re.compile(r'\b(?:https?://|www\.)[^\s<>"\']+\b', re.IGNORECASE)),
            (PIIType.OTHER, re.compile(r'\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b')),
            (PIIType.ID_NUM, re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
            (PIIType.OTHER, re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{1,2}-\d{1,2})\b')),
        ]

        for pii_type, pattern in pattern_specs:
            for match in pattern.finditer(text):
                detections.append(self._create_deterministic_detection(
                    pii_type,
                    match.group(0),
                    match.start(),
                    match.end(),
                    text,
                    transcription_start_time
                ))

        for match in re.finditer(r'(?<!\d)(?:\d[ -]?){13,19}(?!\d)', text):
            candidate = match.group(0).strip()
            digits = re.sub(r'\D', '', candidate)
            if self._passes_luhn(digits):
                detections.append(self._create_deterministic_detection(
                    PIIType.ID_NUM,
                    candidate,
                    match.start(),
                    match.end(),
                    text,
                    transcription_start_time
                ))

        keyword_pattern = self._compile_keyword_pattern()
        if keyword_pattern:
            for match in keyword_pattern.finditer(text):
                detections.append(self._create_deterministic_detection(
                    PIIType.OTHER,
                    match.group(0),
                    match.start(),
                    match.end(),
                    text,
                    transcription_start_time
                ))

        ordered_detections = sorted(
            detections,
            key=lambda detection: (detection.start_char, detection.end_char, detection.text)
        )
        return self._merge_detections([], ordered_detections)

    def _compile_keyword_pattern(self) -> Optional[re.Pattern]:
        """Compile configured sensitive keyword phrases."""
        keywords = [keyword.strip() for keyword in self.pii_keywords if keyword.strip()]
        if not keywords:
            return None

        keyword_alternation = "|".join(re.escape(keyword) for keyword in keywords)
        return re.compile(
            rf'\b(?:{keyword_alternation})\b(?:\s*(?:is|:|=)\s*\S+)?',
            re.IGNORECASE
        )

    def _passes_luhn(self, digits: str) -> bool:
        """Validate a digit string with the Luhn checksum."""
        if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
            return False

        total = 0
        reverse_digits = digits[::-1]
        for idx, digit in enumerate(reverse_digits):
            value = int(digit)
            if idx % 2 == 1:
                value *= 2
                if value > 9:
                    value -= 9
            total += value

        return total % 10 == 0

    def _create_deterministic_detection(
        self,
        pii_type: PIIType,
        pii_text: str,
        start_char: int,
        end_char: int,
        full_text: str,
        transcription_start_time: float
    ) -> PIIDetection:
        """Create a deterministic PIIDetection with estimated timing."""
        estimated_start_time, estimated_end_time = self._estimate_detection_times(
            full_text,
            pii_text,
            start_char,
            transcription_start_time
        )

        return PIIDetection(
            pii_type=pii_type,
            text=pii_text,
            start_char=start_char,
            end_char=end_char,
            confidence=self.deterministic_confidence,
            start_time=estimated_start_time,
            end_time=estimated_end_time,
            word_indices=[]
        )

    def _estimate_detection_times(
        self,
        text: str,
        pii_text: str,
        start_char: int,
        transcription_start_time: float
    ) -> Tuple[float, float]:
        """Estimate detection start/end timestamps from character position."""
        text_position_ratio = start_char / len(text) if len(text) > 0 else 0
        estimated_start_time = transcription_start_time + (text_position_ratio * 5.0)
        estimated_end_time = estimated_start_time + (len(pii_text.split()) * 0.4)
        return estimated_start_time, estimated_end_time

    def _merge_detections(
        self,
        model_detections: List[PIIDetection],
        deterministic_detections: List[PIIDetection]
    ) -> List[PIIDetection]:
        """Merge model and deterministic detections while avoiding duplicate overlaps."""
        merged = list(model_detections)

        for deterministic_detection in deterministic_detections:
            overlapping_index = next(
                (
                    idx for idx, detection in enumerate(merged)
                    if self._detections_overlap(detection, deterministic_detection)
                ),
                None
            )

            if overlapping_index is None:
                merged.append(deterministic_detection)
                continue

            existing_detection = merged[overlapping_index]
            if (
                deterministic_detection.confidence > existing_detection.confidence
                or (
                    deterministic_detection.confidence == existing_detection.confidence
                    and len(deterministic_detection.text) > len(existing_detection.text)
                )
            ):
                merged[overlapping_index] = deterministic_detection

        return sorted(merged, key=lambda detection: detection.start_char)

    def _detections_overlap(
        self,
        first_detection: PIIDetection,
        second_detection: PIIDetection
    ) -> bool:
        """Return whether two detections overlap in character span."""
        return (
            first_detection.start_char < second_detection.end_char
            and second_detection.start_char < first_detection.end_char
        )
    
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
        
        # Map entity type to PIIType enum
        try:
            pii_type = PIIType(entity_type)
        except ValueError:
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
