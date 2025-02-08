# Advanced OCR System with Deep Learning 🔍

A production-grade OCR (Optical Character Recognition) system that combines deep learning models with advanced image preprocessing techniques to accurately extract text from images. The system utilizes multiple OCR engines and provides a user-friendly interface built with Streamlit.

## 🌟 Features

### Core Functionality
- **Multiple OCR Engines**
  - EasyOCR (Primary engine, based on CRAFT text detector and CRNN recognition)
  - Tesseract OCR (Backup engine)

- **Advanced Image Preprocessing**
  - Adaptive thresholding
  - CLAHE contrast enhancement
  - Noise reduction
  - Multiple image enhancement techniques
  - Automatic image quality optimization

- **Intelligent Text Processing**
  - Confidence-based result classification
  - Duplicate text removal
  - Multi-stage verification
  - Parallel processing of enhanced images

### Technical Features
- GPU acceleration support (CUDA)
- Comprehensive error handling
- Detailed logging system
- Type hints for better code maintainability
- Parallel processing capabilities
- Memory-efficient processing

### User Interface
- Clean, intuitive Streamlit interface
- Real-time processing status
- Confidence-based result organization
- Multiple export options (TXT, JSON)
- Processing time tracking
- GPU status monitoring

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA Toolkit (for GPU acceleration)
- Updated GPU drivers (for NVIDIA GPUs)

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/image-text-extractor.git
cd Image-Text-Exractor

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install required packages
pip install -r requirements.txt
```

### GPU Support Installation
For GPU acceleration, install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📋 Requirements
```text
easyocr>=1.7.0
pytesseract>=0.3.10
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.24.0
opencv-python>=4.8.0
pillow>=9.5.0
numpy>=1.24.0
```

## 🎯 Usage

### Running the Application
```bash
streamlit run app.py
```

### Using the Interface
1. Upload an image using the file uploader
2. Wait for the processing to complete
3. View results organized by confidence levels
4. Download results in your preferred format

### Code Example
```python
from ocr_processor import OCRProcessor, ImagePreprocessor

# Initialize the OCR processor
processor = OCRProcessor()

# Process an image
image = cv2.imread('your_image.jpg')
results = processor.process_image(image)

# Access results
for result in results:
    print(f"Text: {result.text}")
    print(f"Confidence: {result.confidence}")
```

## ⚙️ Configuration

### Environment Variables
```bash
TESSERACT_PATH=/path/to/tesseract  # Optional: Custom Tesseract path
CUDA_VISIBLE_DEVICES=0             # Optional: Specify GPU device
```

### GPU Configuration
The system automatically detects and utilizes available GPU resources. You can monitor GPU status in the application sidebar.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Future Improvements
- [ ] Multi-language support
- [ ] Handwritten text recognition
- [ ] Layout analysis
- [ ] Table structure recognition
- [ ] PDF document support
- [ ] API endpoint implementation
- [ ] Docker containerization
- [ ] Cloud deployment support

## 📞 Support
For support, email humayunsaeed75@gmail.com or open an issue in the GitHub repository.

## 🙏 Acknowledgments
- EasyOCR team for their amazing OCR engine
- Tesseract OCR community
- Streamlit team for the great UI framework

## 📚 Citation
If you use this project in your research or work, please cite it as:
```
@software{advanced_ocr_system,
  author = {Humayun Saeed},
  title = {Advanced OCR System with Deep Learning},
  year = {2025},
  url = {https://github.com/yourusername/advanced-ocr-system}
}
```

---
Made with ❤️ by [Humayun Saeed]
