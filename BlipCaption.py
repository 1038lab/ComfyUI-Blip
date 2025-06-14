import torch
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import hf_hub_download
import numpy as np
import folder_paths

MODEL_CACHE = {}

BLIP_MODEL_DIR_BASE = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(BLIP_MODEL_DIR_BASE, exist_ok=True)

class BlipCaption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to generate caption for"}),
                "model_name": (["Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large"], {"tooltip": "BLIP model version: base (faster) or large (more detailed)"}),
                "max_length": ("INT", {"default": 50, "min": 1, "max": 100, "tooltip": "Maximum length of the generated caption"}),
                "use_nucleus_sampling": ("BOOLEAN", {"default": False, "tooltip": "Use nucleus sampling for more creative captions"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate_caption"
    CATEGORY = "🧪AILab/📝BlipCaption"

    def _download_model_files(self, repo_id, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        
        files = [
            "config.json", 
            "preprocessor_config.json", 
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json", 
            "vocab.txt"
        ]
        
        weight_files = ["model.safetensors", "pytorch_model.bin"]
        weight_downloaded = False
        
        for weight_file in weight_files:
            try:
                hf_hub_download(repo_id=repo_id, filename=weight_file, local_dir=local_dir)
                weight_downloaded = True
                break
            except Exception:
                continue
                
        if not weight_downloaded:
            raise RuntimeError(f"Failed to download model weights for {repo_id}")
            
        for file in files:
            try:
                hf_hub_download(repo_id=repo_id, filename=file, local_dir=local_dir)
            except Exception:
                if file in ["config.json", "preprocessor_config.json"]:
                    raise RuntimeError(f"Failed to download required file {file}")

    def generate_caption(self, image, model_name, max_length, use_nucleus_sampling):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_id_folder = model_name.split('/')[-1]
        model_local_path = os.path.join(BLIP_MODEL_DIR_BASE, model_id_folder)
        
        if model_name not in MODEL_CACHE:
            if not os.path.exists(os.path.join(model_local_path, "config.json")):
                print(f"Downloading model {model_name} to {model_local_path}")
                self._download_model_files(model_name, model_local_path)
                
            print(f"Loading model {model_name}")
            processor = BlipProcessor.from_pretrained(model_local_path)
            model = BlipForConditionalGeneration.from_pretrained(model_local_path).to(device)
            MODEL_CACHE[model_name] = {"processor": processor, "model": model}
        
        processor = MODEL_CACHE[model_name]["processor"]
        model = MODEL_CACHE[model_name]["model"]

        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        inputs = processor(img, return_tensors="pt").to(device)
        
        if use_nucleus_sampling:
            gen_kwargs = {
                "max_length": max_length,
                "min_length": 5,
                "do_sample": True,
                "top_p": 0.9,
                "num_return_sequences": 1
            }
        else:
            gen_kwargs = {
                "max_length": max_length,
                "min_length": 5,
                "num_beams": 5,
                "do_sample": False
            }
            
        outputs = model.generate(**inputs, **gen_kwargs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        return (caption,)

    @classmethod
    def IS_CHANGED(cls):
        return ""


class BlipCaptionAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to generate caption for"}),
                "model_name": (["Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large"], {"tooltip": "BLIP model version: base (faster) or large (more detailed)"}),
                "use_nucleus_sampling": ("BOOLEAN", {"default": False, "tooltip": "Use nucleus sampling instead of beam search"}),
                "min_length": ("INT", {"default": 5, "min": 1, "max": 50, "tooltip": "Minimum length of the generated caption"}),
                "max_length": ("INT", {"default": 50, "min": 1, "max": 100, "tooltip": "Maximum length of the generated caption"}),
                "num_beams": ("INT", {"default": 5, "min": 1, "max": 10, "tooltip": "Number of beams for beam search (used when nucleus sampling is off)"}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Top-p for nucleus sampling (used when nucleus sampling is on)"}),
                "force_refresh": ("BOOLEAN", {"default": False, "tooltip": "Force reload model from disk and clear cache"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional prompt to condition the caption (e.g. 'a photo of'). Leave empty for unconditional captioning."})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate_caption"
    CATEGORY = "🧪AILab/📝BlipCaption"

    def _download_model_files(self, repo_id, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        
        files = [
            "config.json", 
            "preprocessor_config.json", 
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json", 
            "vocab.txt"
        ]
        
        weight_files = ["model.safetensors", "pytorch_model.bin"]
        weight_downloaded = False
        
        for weight_file in weight_files:
            try:
                hf_hub_download(repo_id=repo_id, filename=weight_file, local_dir=local_dir)
                weight_downloaded = True
                break
            except Exception:
                continue
                
        if not weight_downloaded:
            raise RuntimeError(f"Failed to download model weights for {repo_id}")
            
        for file in files:
            try:
                hf_hub_download(repo_id=repo_id, filename=file, local_dir=local_dir)
            except Exception:
                if file in ["config.json", "preprocessor_config.json"]:
                    raise RuntimeError(f"Failed to download required file {file}")

    def generate_caption(self, image, model_name, use_nucleus_sampling, min_length, max_length, num_beams, top_p, prompt, force_refresh):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_id_folder = model_name.split('/')[-1]
        model_local_path = os.path.join(BLIP_MODEL_DIR_BASE, model_id_folder)
        
        if force_refresh and model_name in MODEL_CACHE:
            print(f"Forcing refresh of model {model_name}")
            del MODEL_CACHE[model_name]
        
        if model_name not in MODEL_CACHE:
            if not os.path.exists(os.path.join(model_local_path, "config.json")):
                print(f"Downloading model {model_name} to {model_local_path}")
                self._download_model_files(model_name, model_local_path)
                
            print(f"Loading model {model_name}")
            processor = BlipProcessor.from_pretrained(model_local_path)
            model = BlipForConditionalGeneration.from_pretrained(model_local_path).to(device)
            MODEL_CACHE[model_name] = {"processor": processor, "model": model}
        
        processor = MODEL_CACHE[model_name]["processor"]
        model = MODEL_CACHE[model_name]["model"]

        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        if prompt and prompt.strip():
            inputs = processor(img, prompt, return_tensors="pt").to(device)
        else:
            inputs = processor(img, return_tensors="pt").to(device)
        
        if use_nucleus_sampling:
            gen_kwargs = {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": True,
                "top_p": top_p,
                "num_return_sequences": 1
            }
        else:
            gen_kwargs = {
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": num_beams,
                "do_sample": False
            }
            
        outputs = model.generate(**inputs, **gen_kwargs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        return (caption,)

    @classmethod
    def IS_CHANGED(cls):
        return ""


NODE_CLASS_MAPPINGS = {
    "BlipCaption": BlipCaption,
    "BlipCaptionAdvanced": BlipCaptionAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlipCaption": "Blip Caption",
    "BlipCaptionAdvanced": "Blip Caption (Advanced)"
}
