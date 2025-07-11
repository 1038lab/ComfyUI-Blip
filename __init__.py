# ComfyUI-Blip
# Image captioning node based on Salesforce BLIP model

from .BlipCaption import NODE_CLASS_MAPPINGS as BLIP_NODE_CLASS_MAPPINGS
from .BlipCaption import NODE_DISPLAY_NAME_MAPPINGS as BLIP_NODE_DISPLAY_NAME_MAPPINGS

from .TextTranslator import NODE_CLASS_MAPPINGS as TRANSLATOR_NODE_CLASS_MAPPINGS
from .TextTranslator import NODE_DISPLAY_NAME_MAPPINGS as TRANSLATOR_NODE_DISPLAY_NAME_MAPPINGS

# Merge the dictionaries
NODE_CLASS_MAPPINGS = {**BLIP_NODE_CLASS_MAPPINGS, **TRANSLATOR_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**BLIP_NODE_DISPLAY_NAME_MAPPINGS, **TRANSLATOR_NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']