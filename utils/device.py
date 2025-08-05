import torch
import logging


def get_device(preferred_device: str = 'cuda') -> torch.device:
    """
    Get the best available device.
    
    Args:
        preferred_device: Preferred device ('cuda', 'cpu', 'auto')
    
    Returns:
        torch.device object
    """
    if preferred_device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif preferred_device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logging.warning("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device(preferred_device)
    
    return device


def setup_device(preferred_device: str = 'cuda') -> torch.device:
    """
    Setup device and print system information.
    
    Args:
        preferred_device: Preferred device type
    
    Returns:
        Configured torch.device
    """
    device = get_device(preferred_device)
    
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"Max memory reserved: {torch.cuda.max_memory_reserved(0) / 1024**3:.2f} GB")
        
        # Set memory fraction for RTX 4080 Super (16GB)
        if torch.cuda.get_device_properties(0).total_memory > 15e9:  # >15GB
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available memory
            print("Set memory fraction to 90% for RTX 4080 Super")
    
    return device