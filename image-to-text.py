import streamlit as st
import cv2
import numpy as np
import easyocr
import pytesseract
from typing import List, Dict, Tuple, Optional
import json
import torch
from PIL import Image
import io
import os
from dataclasses import dataclass
from enum import Enum
import logging
import time
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_gpu():
    """Setup and verify GPU availability"""
    if torch.cuda.is_available():
        try:
            # Test CUDA initialization
            torch.cuda.init()
            device_name = torch.cuda.get_device_name(0)
            return True, f"GPU Available: {device_name}"
        except Exception as e:
            logger.error(f"GPU initialization error: {str(e)}")
            return False, "GPU Found but CUDA initialization failed"
    return False, "No GPU detected"

class OCREngine(Enum):
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"

@dataclass
class OCRResult:
    text: str
    confidence: float
    bounding_box: Optional[List[Tuple[int, int]]] = None
    engine: Optional[OCREngine] = None

class ImagePreprocessor:
    """Handles image preprocessing for optimal OCR results."""
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> List[np.ndarray]:
        """Enhanced preprocessing pipeline."""
        enhanced_images = []
        
        # Original image
        enhanced_images.append(image.copy())
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced_images.append(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        enhanced_images.append(denoised)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced_images.append(enhanced)
        
        # Thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced_images.append(binary)
        
        # Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        enhanced_images.append(adaptive_thresh)
        
        return enhanced_images

class OCRProcessor:
    """Enhanced OCR processor with improved GPU handling."""
    
    def __init__(self):
        # Initialize GPU
        self.gpu_available, self.gpu_status = setup_gpu()
        
        # Initialize EasyOCR with proper GPU settings
        try:
            self.reader = easyocr.Reader(
                ['en'],
                gpu=self.gpu_available,
                model_storage_directory='./models',
                download_enabled=True
            )
        except Exception as e:
            logger.error(f"EasyOCR initialization error: {str(e)}")
            raise RuntimeError("Failed to initialize EasyOCR")
    
    def process_image(self, image: np.ndarray) -> List[OCRResult]:
        """Process image with enhanced error handling."""
        results = []
        
        try:
            # Process with EasyOCR
            easy_results = self.reader.readtext(image)
            
            for bbox, text, conf in easy_results:
                if text.strip():  # Only include non-empty text
                    results.append(OCRResult(
                        text=text,
                        confidence=conf,
                        bounding_box=bbox,
                        engine=OCREngine.EASYOCR
                    ))
            
            # Backup with Tesseract if EasyOCR results are poor
            if not results:
                tesseract_results = self._process_with_tesseract(image)
                results.extend(tesseract_results)
                
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            # Fallback to Tesseract
            results.extend(self._process_with_tesseract(image))
        
        return results
    
    def _process_with_tesseract(self, image: np.ndarray) -> List[OCRResult]:
        """Backup OCR using Tesseract."""
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            results = []
            
            for i, text in enumerate(data['text']):
                if text.strip():
                    conf = float(data['conf'][i])
                    if conf > 0:  # Only include results with positive confidence
                        results.append(OCRResult(
                            text=text,
                            confidence=conf/100,
                            engine=OCREngine.TESSERACT
                        ))
            return results
        except Exception as e:
            logger.error(f"Tesseract error: {str(e)}")
            return []

def main():
    st.set_page_config(page_title="Enhanced OCR System", layout="wide")
    
    st.title("Enhanced OCR System")
    
    try:
        # Initialize OCR processor
        ocr_processor = OCRProcessor()
        st.sidebar.success(ocr_processor.gpu_status)
    except Exception as e:
        st.error(f"Error initializing OCR system: {str(e)}")
        return
    
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
    )
    
    if uploaded_file:
        try:
            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                
                # Convert image for processing
                image_np = np.array(image)
                if len(image_np.shape) == 2:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
                elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            
            # Process image with enhanced versions
            with st.spinner("Processing image..."):
                preprocessor = ImagePreprocessor()
                enhanced_images = preprocessor.enhance_image(image_np)
                
                all_results = []
                for enhanced_img in enhanced_images:
                    results = ocr_processor.process_image(enhanced_img)
                    all_results.extend(results)
                
                # Remove duplicates and sort by confidence
                seen_texts = set()
                unique_results = []
                for result in sorted(all_results, key=lambda x: x.confidence, reverse=True):
                    normalized_text = result.text.lower().strip()
                    if normalized_text and normalized_text not in seen_texts:
                        unique_results.append(result)
                        seen_texts.add(normalized_text)
                
                # Display results
                with col2:
                    st.subheader("Extracted Text")
                    
                    if not unique_results:
                        st.warning("No text was detected in the image. Try uploading a clearer image or one with more visible text.")
                    else:
                        # Group by confidence
                        high_conf = [r for r in unique_results if r.confidence > 0.8]
                        med_conf = [r for r in unique_results if 0.5 <= r.confidence <= 0.8]
                        low_conf = [r for r in unique_results if r.confidence < 0.5]
                        
                        # Display results in expandable sections
                        with st.expander("High Confidence Results", expanded=True):
                            for r in high_conf:
                                st.markdown(f"**{r.text}** (Confidence: {r.confidence:.2f})")
                        
                        with st.expander("Medium Confidence Results"):
                            for r in med_conf:
                                st.markdown(f"**{r.text}** (Confidence: {r.confidence:.2f})")
                        
                        with st.expander("Low Confidence Results"):
                            for r in low_conf:
                                st.markdown(f"**{r.text}** (Confidence: {r.confidence:.2f})")
                        
                        # Download buttons
                        all_text = "\n\n".join([
                            "=== High Confidence ===\n" + "\n".join([r.text for r in high_conf]),
                            "=== Medium Confidence ===\n" + "\n".join([r.text for r in med_conf]),
                            "=== Low Confidence ===\n" + "\n".join([r.text for r in low_conf])
                        ])
                        
                        st.download_button(
                            label="Download Results",
                            data=all_text,
                            file_name="ocr_results.txt",
                            mime="text/plain"
                        )
                        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            logger.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()