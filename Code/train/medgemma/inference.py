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
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        
        try:
            if use_pipeline:
                # Use the simpler pipeline API (recommended)
                # Note: offline mode is controlled via env vars, not model_kwargs
                pipe_kwargs = {
                    "model": model_id,
                    "torch_dtype": torch.bfloat16,
                    "device": self.device if self.device == "cuda" else -1,
                    "token": HF_TOKEN,
                }
                self.pipe = pipeline("image-text-to-text", **pipe_kwargs)
                self.processor = None
                self.model = None
            else:
                # Load model directly for more control
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if self.device == "cuda" else None,
                    token=HF_TOKEN,
                )
                self.processor = AutoProcessor.from_pretrained(
                    model_id, 
                    token=HF_TOKEN,
                )
                self.pipe = None
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            # If offline mode fails, try online
            if offline_mode:
                print(f"⚠️  Offline loading failed, trying online mode...")
                # Clear offline env and retry
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                try:
                    if use_pipeline:
                        pipe_kwargs = {
                            "model": model_id,
                            "torch_dtype": torch.bfloat16,
                            "device": self.device if self.device == "cuda" else -1,
                            "token": HF_TOKEN,
                        }
                        self.pipe = pipeline("image-text-to-text", **pipe_kwargs)
                        self.processor = None
                        self.model = None
                    else:
                        self.model = AutoModelForImageTextToText.from_pretrained(
                            model_id,
                            torch_dtype=torch.bfloat16,
                            device_map="auto" if self.device == "cuda" else None,
                            token=HF_TOKEN,
                        )
                        self.processor = AutoProcessor.from_pretrained(
                            model_id, 
                            token=HF_TOKEN,
                        )
                        self.pipe = None
                    print("✅ Model loaded successfully (online mode)!")
                except Exception as e2:
                    print(f"❌ Error loading model: {e2}")
                    raise e2
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
        Preprocess CT slice for MedGemma input using Kaggle 1st Place Logic.
        Converts to 3-channel RGB with different windows:
        1. Brain Window (Red): Soft tissue, gray/white matter differentiation
        2. Blood/Subdural Window (Green): Hemorrhage detection
        3. Bone/Vessel Window (Blue): Aneurysm and calcification
        """
        try:
            # Try to use the advanced preprocessor from ml/data
            import sys
            sys.path.append(str(Path(__file__).parent.parent / 'ml'))
            from data.advanced_preprocessing import KaggleWinningPreprocessor
            
            preprocessor = KaggleWinningPreprocessor()
            
            # Detect modality to handle MRI specifically if needed
            # But here we assume CT for the windowing logic
            
            # Create 3-channel Multi-Window image
            # This aligns with Ian Pan's 1st place solution (using specific windows for channels)
            rgb = preprocessor.create_multi_window_image(pixel_array)
            
            # Apply CLAHE for local contrast (vessel enhancement)
            if preprocessor.use_clahe:
                for c in range(3):
                    rgb[:, :, c] = preprocessor.apply_clahe(rgb[:, :, c])
            
            # Convert to uint8 0-255
            rgb = (rgb * 255).astype(np.uint8)
            return Image.fromarray(rgb)
            
        except ImportError:
            print("⚠️ Advanced preprocessor not found. Using basic windowing.")
            # Fallback to basic windowing
            lower = window_center - window_width / 2
            upper = window_center + window_width / 2
            
            windowed = np.clip(pixel_array, lower, upper)
            windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
            
            if len(windowed.shape) == 2:
                rgb = np.stack([windowed, windowed, windowed], axis=-1)
            else:
                rgb = windowed
            return Image.fromarray(rgb)
    
    def analyze_image(
        self, 
        image: Union[Image.Image, np.ndarray, str, Path],
        prompt: str = None,
        system_prompt: str = SYSTEM_PROMPT
    ) -> str:
        """
        Analyze a medical image using MedGemma.
        
        Args:
            image: PIL Image, numpy array, or path to image
            prompt: Question/instruction for the model (defaults to general analysis)
            system_prompt: System context
            
        Returns:
            Model's analysis as text
        """
        if prompt is None:
            prompt = ANALYSIS_PROMPT

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
            output = self.pipe(messages, max_new_tokens=MAX_NEW_TOKENS)
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
    
    def analyze_for_kaggle(self, image: Union[Image.Image, np.ndarray], modality: str = "CTA", slice_info: str = "Unknown Slice") -> str:
        """
        Perform detailed analysis specifically for Kaggle competition requirements.
        Based on winning solution insights (Ian Pan, et al.) and clinical overview.
        
        Args:
            image: The image crop to analyze
            modality: Imaging modality (CTA, MRA, T1, T2)
            slice_info: Slice number or ID (e.g., "Slice 15")
        
        TARGETS: 13 Specific Anatomical Locations (Saccular Aneurysms ONLY)
        """
        base_context = f"Analyzing {modality} image, {slice_info}."
        
        prompts = [
            # 1. Anterior Circulation (ICA, MCA, ACA, AComm)
            f"{base_context} Inspect the Anterior Circulation: Look for saccular aneurysms in the Left/Right ICA (Infraclinoid/Supraclinoid), MCA, ACA, or AComm.",
            
            # 2. Posterior Circulation (Vertebral, Basilar, PCA)
            f"Inspect the Posterior Circulation: Check the Vertebral Arteries, Basilar Tip, and PComm for any focal outpouchings.",
            
            # 3. Pathological Signs (Star Sign, Edema)
            f"Check for 'Star Sign' in basal cisterns (SAH) or mass effect (edema/shift) on this {modality} slice.",
            
            # 4. Classification & exclusions
            "Classify findings: 1. Negative, 2. Positive (Unruptured), 3. Positive (Ruptured). EXCLUDE if fusiform/infundibulum.",
            
            # 5. Detailed 13-Location Checklist (RSNA Requirement)
            """Check each of the 13 specific locations for aneurysm presence:
            1. Left Infraclinoid ICA
            2. Right Infraclinoid ICA
            3. Left Supraclinoid ICA
            4. Right Supraclinoid ICA
            5. Left MCA
            6. Right MCA
            7. Anterior Communicating Artery (AComm)
            8. Left ACA
            9. Right ACA
            10. Left PComm
            11. Right PComm
            12. Basilar Tip
            13. Other Posterior Circulation
            
            Report 'Negative' or 'Positive' for each."""
        ]
        
        full_report = "### MedGemma RSNA 13-Location Analysis\n\n"
        
        for p in prompts:
            response = self.analyze_image(image, prompt=p)
            q_text = "Analysis Query"
            if "Anterior" in p: q_text = "Anterior Circulation"
            elif "Posterior" in p: q_text = "Posterior Circulation"
            elif "13" in p: q_text = "13-Point Location Check"
            elif "Star" in p: q_text = "Pathology Check"
            
            full_report += f"**{q_text}:**\n{response}\n\n"
            
        return full_report

    def analyze_dicom(self, dicom_path: Union[str, Path]) -> str:
        """Analyze a DICOM file."""
        import pydicom
        
        try:
            dcm = pydicom.dcmread(str(dicom_path))
            pixel_array = dcm.pixel_array.astype(np.float32)
            
            # Apply rescale if available
            slope = float(getattr(dcm, 'RescaleSlope', 1))
            intercept = float(getattr(dcm, 'RescaleIntercept', 0))
            pixel_array = pixel_array * slope + intercept
            
            # Convert to image and analyze
            image = self.preprocess_ct_slice(pixel_array)
            return self.analyze_for_kaggle(image)
            
        except Exception as e:
            return f"Error analyzing DICOM: {str(e)}"


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
