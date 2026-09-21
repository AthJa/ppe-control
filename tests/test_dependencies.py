"""Test file to verify all required dependencies load correctly in the current environment."""

def test_imports():
    # Core data/vision
    import cv2
    import numpy as np
    import pandas as pd
    
    # ML / Models
    import torch
    import torchvision
    import ultralytics
    
    # Specific face recognition components
    import facenet_pytorch
    from facenet_pytorch import InceptionResnetV1
    
    # UI
    import streamlit

    assert True, "All dependencies imported successfully"
