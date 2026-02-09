"""
MedGemma Inference Pipeline
Analyzes CT scans using Google's MedGemma model.

Based on official HuggingFace docs: https://huggingface.co/google/medgemma-4b-it
"""

import torch
from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Union

from config import (
    MODEL_ID, HF_TOKEN, CACHE_DIR,
    MAX_NEW_TOKENS,
    SYSTEM_PROMPT, ANALYSIS_PROMPT,
    CT_WINDOW_CENTER, CT_WINDOW_WIDTH
)


class MedGemmaInference:
    """
    MedGemma inference class for medical image analysis.
    
    Usage:
        model = MedGemmaInference()
        result = model.analyze_image(image_path)
        print(result)
    """
    
    def __init__(self, model_id: str = MODEL_ID, use_pipeline: bool = True, offline_mode: bool = True):
        """Initialize MedGemma model.
        
        Args:
            model_id: HuggingFace model ID
            use_pipeline: If True, use simplified pipeline API
            offline_mode: If True, use cached model without network (default: True)
        """
        print("=" * 60)
        print("🏥 MEDGEMMA - Medical AI Assistant")
        print("=" * 60)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Device: {self.device}")
        
        if HF_TOKEN is None:
            print("⚠️  Warning: HF_TOKEN not set. You may need it for gated models.")
            print("   Set it with: set HF_TOKEN=your_token_here")
        
        print(f"📥 Loading model: {model_id}")
        if offline_mode:
            print("   (Offline mode - using cached model)")
        else:
            print("   (This may take a few minutes on first run...)")
        
        self.use_pipeline = use_pipeline
        
        # Set environment for offline mode
        import os
        if offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        try:
            if use_pipeline:
                # Use the simpler pipeline API (recommended)
                self.pipe = pipeline(
                    "image-text-to-text",
                    model=model_id,
                    torch_dtype=torch.bfloat16,
                    device=self.device if self.device == "cuda" else -1,
                    token=HF_TOKEN,
                    model_kwargs={"local_files_only": offline_mode}
                )
                self.processor = None
                self.model = None
            else:
                # Load model directly for more control
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if self.device == "cuda" else None,
                    token=HF_TOKEN,
                    local_files_only=offline_mode,
                )
                self.processor = AutoProcessor.from_pretrained(
                    model_id, 
                    token=HF_TOKEN,
                    local_files_only=offline_mode
                )
                self.pipe = None
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            # If offline mode fails, try online
            if offline_mode:
                print(f"⚠️  Offline loading failed, trying online mode...")
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                self.__init__(model_id, use_pipeline, offline_mode=False)
            else:
                print(f"❌ Error loading model: {e}")
                raise
    
    def preprocess_ct_slice(
        self, 
        pixel_array: np.ndarray,
        window_center: int = CT_WINDOW_CENTER,
        window_width: int = CT_WINDOW_WIDTH
    ) -> Image.Image:
        """
        Preprocess CT slice for MedGemma input.
        Applies windowing and converts to PIL Image.
        """
        # Apply CT windowing
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        
        windowed = np.clip(pixel_array, lower, upper)
        windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
        
        # Convert to RGB (MedGemma expects RGB)
        if len(windowed.shape) == 2:
            rgb = np.stack([windowed, windowed, windowed], axis=-1)
        else:
            rgb = windowed
        
        return Image.fromarray(rgb)
    
    def analyze_image(
        self, 
        image: Union[Image.Image, np.ndarray, str, Path],
        prompt: str = ANALYSIS_PROMPT,
        system_prompt: str = SYSTEM_PROMPT
    ) -> str:
        """
        Analyze a medical image using MedGemma.
        
        Args:
            image: PIL Image, numpy array, or path to image
            prompt: Question/instruction for the model
            system_prompt: System context
            
        Returns:
            Model's analysis as text
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = self.preprocess_ct_slice(image)
        
        # Format conversation
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]
        
        if self.use_pipeline:
            # Use pipeline API
            output = self.pipe(text=messages, max_new_tokens=MAX_NEW_TOKENS)
            response = output[0]["generated_text"][-1]["content"]
        else:
            # Use model directly
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False
                )
                generation = generation[0][input_len:]
            
            response = self.processor.decode(generation, skip_special_tokens=True)
        
        return response
    
    def analyze_dicom(self, dicom_path: Union[str, Path]) -> str:
        """Analyze a DICOM file."""
        import pydicom
        
        dcm = pydicom.dcmread(str(dicom_path))
        pixel_array = dcm.pixel_array.astype(np.float32)
        
        # Apply rescale if available
        slope = float(getattr(dcm, 'RescaleSlope', 1))
        intercept = float(getattr(dcm, 'RescaleIntercept', 0))
        pixel_array = pixel_array * slope + intercept
        
        # Convert to image and analyze
        image = self.preprocess_ct_slice(pixel_array)
        return self.analyze_image(image)


# Simple test
if __name__ == "__main__":
    print("\n🧪 Testing MedGemma Inference Pipeline...")
    
    # Check if HF token is set
    if HF_TOKEN is None:
        print("\n❌ Error: HF_TOKEN not set!")
        print("   Please set your Hugging Face token:")
        print("   set HF_TOKEN=your_token_here")
        print("\n   Get your token from: https://huggingface.co/settings/tokens")
    else:
        print("✅ HF_TOKEN is set")
        
        # Try to initialize (will download model)
        try:
            model = MedGemmaInference()
            print("\n✅ MedGemma is ready!")
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
