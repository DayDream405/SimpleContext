import torch
from training import WSDPredictor
from train_monitor import OptimizedModuleSelector
from typing import List

class HookBasedMasker:
    """基于钩子的层masker"""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.masking_strategies = {}
    
    def add_layer_mask(self, layer_pattern: List, mask_strategy="zero", mask_strength=1.0):
        """
        添加层mask
        
        Args:
            mask_strategy: "zero", "reduce", "shuffle", "freeze", "noise", "identity"
            mask_strength: mask强度 (0-1)
        """

        module_selector = OptimizedModuleSelector(self.model, layer_pattern)
        for name, module in module_selector.select_target_modules():
            
            hook = self._create_masking_hook(name, mask_strategy, mask_strength)
            handle = module.register_forward_hook(hook)
            self.hooks.append(handle)
            self.masking_strategies[name] = (mask_strategy, mask_strength)
            print(f"为层 {name} 添加{mask_strategy} mask")
    
    def _create_masking_hook(self, layer_name, strategy, strength):
        """创建masking钩子"""
        def masking_hook(module, input, output):
            # 处理不同类型的输出
            if isinstance(input, torch.Tensor):
                # 单个Tensor输出
                original_input = input
                masked_output = self._apply_mask_to_tensor(original_input, strategy, strength)
                # print(f"层 {layer_name} 应用了{strategy} mask (强度: {strength})")
                return masked_output
                
            elif isinstance(input, tuple):
                # 元组输出（常见于Transformer）
                original_inputs = input
                masked_outputs = []
                
                for i, out in enumerate(original_inputs):
                    if isinstance(out, torch.Tensor):
                        masked_out = self._apply_mask_to_tensor(out, strategy, strength)
                        masked_outputs.append(masked_out)
                    else:
                        # 非Tensor元素保持不变
                        masked_outputs.append(out)
                
                # print(f"层 {layer_name} 应用了{strategy} mask到元组输出 (强度: {strength})")
                return tuple(masked_outputs)
                
            else:
                # 其他类型的输出
                # print(f"层 {layer_name} 输出类型 {type(output)} 不支持masking")
                return output
        
        return masking_hook
    
    def _apply_mask_to_tensor(self, tensor, strategy, strength):
        """对单个Tensor应用mask"""
        if strategy == "zero":
            # 完全归零
            masked_tensor = torch.zeros_like(tensor)
            return tensor * (1 - strength) + masked_tensor * strength
            
        elif strategy == "reduce":
            # 减小输出幅度
            return tensor * (1 - strength)
            
        elif strategy == "shuffle":
            # 打乱输出
            if tensor.dim() >= 2:
                batch_size, hidden_size = tensor.shape[0], tensor.shape[-1]
                # 保持批次维度，打乱特征维度
                original_shape = tensor.shape
                flattened = tensor.reshape(batch_size, -1, hidden_size)
                indices = torch.randperm(flattened.size(1))
                shuffled = flattened[:, indices, :].reshape(original_shape)
                return tensor * (1 - strength) + shuffled * strength
            else:
                return tensor
                
        elif strategy == "freeze":
            # 使用固定值
            # if not hasattr(self, 'frozen_tensors'):
            #     self.frozen_tensors = {}
            # key = id(tensor)
            # if key not in self.frozen_tensors:
            #     self.frozen_tensors[key] = tensor.detach().clone()
            # frozen = self.frozen_tensors[key]
            return tensor + tensor * strength
            
        elif strategy == "noise":
            # 添加噪声
            noise = torch.randn_like(tensor) * strength
            return tensor + noise
            
        elif strategy == "identity":
            return tensor
        else:
            return tensor

    def remove_all_masks(self):
        """移除所有mask"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.masking_strategies.clear()
        print("所有mask已移除")

LAYER_CONFIGS = [
    {'layer_pattern': 'encoder.layer.7', 'strategy': 'identity', 'strength': 1.0},
    {'layer_pattern': 'encoder.layer.10', 'strategy': 'identity', 'strength': 1.0},
    {'layer_pattern': 'encoder.layer.11', 'strategy': 'identity', 'strength': 1.0},
    # {'layer_pattern': 'encoder.layer.8', 'strategy': 'identity', 'strength': 1.0},
]
class MaskableWSDPredictor(WSDPredictor):
    """支持层mask的WSD预测器"""
    
    def __init__(self, model_path="./saved_models/best_model"):
        super().__init__(model_path)
        self.masker = HookBasedMasker(self.model)
        self.active_masks = {}
    
    def set_layer_mask(self, layer_configs=LAYER_CONFIGS):
        """
        设置层mask配置
        
        Args:
            layer_configs: [{
                'layer_pattern': 'encoder.layer.8',
                'strategy': 'zero', 
                'strength': 1.0
            }]
        """
        self.masker.remove_all_masks()
        
        for config in layer_configs:
            self.masker.add_layer_mask(
                [config['layer_pattern']],
                config.get('strategy', 'zero'),
                config.get('strength', 1.0)
            )
            self.active_masks[config['layer_pattern']] = config
    
    def get_ablation_analysis(self, context, target_word, layer_groups):
        """消融分析：测试mask不同层组的影响"""
        baseline = self.predict(context, target_word)
        baseline_confidence = baseline['confidence']
        
        results = {}
        
        for group_name, layers in layer_groups.items():
            # 设置当前组的mask
            mask_configs = [{'layer_pattern': layer, 'strategy': 'zero'} for layer in layers]
            self.set_layer_mask(mask_configs)
            
            # 预测
            masked_result = self.predict(context, target_word)
            confidence_drop = baseline_confidence - masked_result['confidence']
            
            results[group_name] = {
                'baseline_confidence': baseline_confidence,
                'masked_confidence': masked_result['confidence'],
                'confidence_drop': confidence_drop,
                'prediction_changed': masked_result['predicted_definition'] != baseline['predicted_definition']
            }
        
        # 恢复无mask状态
        self.set_layer_mask([])
        
        return results

# 使用示例
def comprehensive_layer_analysis():
    """全面的层重要性分析"""
    predictor = MaskableWSDPredictor()
    
    # 定义层组
    layer_groups = {
        'vocabulary_layers': ['embeddings'],
        'lower_layers': ['encoder.layer.0', 'encoder.layer.1', 'encoder.layer.2'],
        'middle_layers': ['encoder.layer.4', 'encoder.layer.5', 'encoder.layer.6', 'encoder.layer.7'],
        'upper_layers': ['encoder.layer.8', 'encoder.layer.9', 'encoder.layer.10', 'encoder.layer.11'],
        'task_head': ['pooler', 'classifier']
    }
    
    test_cases = [
        ("He went to the bank to withdraw money.", "bank"),
        ("The river bank was steep and muddy.", "bank"),
        ("She works at the bank as a teller.", "bank")
    ]
    
    for context, target in test_cases:
        print(f"\n测试: '{target}' in '{context}'")
        analysis = predictor.get_ablation_analysis(context, target, layer_groups)
        
        for group, result in analysis.items():
            if result['confidence_drop'] > 0.1:
                print(f"  ⚠️ {group}: 置信度下降 {result['confidence_drop']:.3f}")
            elif result['prediction_changed']:
                print(f"  🔄 {group}: 预测改变")