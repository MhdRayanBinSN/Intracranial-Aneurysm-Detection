# MedGemma - Medical AI Project

> Google's Open Medical AI Model for Aneurysm Detection

## 📁 Project Structure
```
medgemma/
├── README.md           # This file
├── requirements.txt    # Dependencies
├── config.py           # Configuration
├── inference.py        # Main inference script
├── demo.py             # Demo with sample CT
└── utils/
    ├── __init__.py
    ├── image_loader.py # Load DICOM/CT scans
    └── report_gen.py   # Generate reports
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Hugging Face Token
```bash
# Get token from: https://huggingface.co/settings/tokens
set HF_TOKEN=your_token_here
```

### 3. Run Demo
```bash
python demo.py --input path/to/ct_scan.dcm
```

## 🧠 What is MedGemma?

MedGemma is Google's **open-source medical AI model** built on Gemma 3.
- **4B Multimodal**: For image + text (recommended for CT analysis)
- **27B Text-only**: For clinical reasoning
- **27B Multimodal**: Large model for complex cases

## 🔗 Comparison with ResNet

| Feature | 3D ResNet (Current) | MedGemma |
|---------|---------------------|----------|
| Type | CNN | LLM + Vision |
| Output | Probability scores | Natural language report |
| Training | Requires training | Pre-trained by Google |
| Segmentation | No | Text-based localization |

## 📚 References
- [MedGemma on Hugging Face](https://huggingface.co/google/medgemma-4b-it)
- [Google AI Blog](https://ai.google.dev/gemma/docs/medgemma)
