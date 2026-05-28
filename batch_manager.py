import torch

class DeviceBatchManager:
    """管理设备上的batch数据，自动清理"""
    
    def __init__(self, batch, device):
        self.batch = {k: v.to(device) for k, v in batch.items()}
        self.device = device
    
    def __enter__(self):
        return self.batch
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 清理batch数据
        for key in list(self.batch.keys()):
            del self.batch[key]
        torch.cuda.empty_cache()