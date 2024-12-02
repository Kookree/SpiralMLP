from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class SpiralConfig:
    layers: List[int] 
    embed_dims: List[int] 
    swin_info: str
    layer: int = None
    embed_dim: int = None
    patch_size: int = 7
    transitions: List[bool] = field(default_factory=lambda: [True, True, True, True])
    mlp_ratio: int = None
    mlp_ratios: List[int] = field(default_factory=lambda: [4, 4, 4, 4])
    num_classes: int = 1000
    attn_drop: float = 0.1
    proj_drop: float = 0.1
    drop_path: float = 0.1
    skip_lam: float = 1.0
    fork_feat: bool = False
    pretrained: str = None
#    extra_config: Any = field(default_factory=dict)

class Classification:
    def __init__(self):
        pass
    @staticmethod
    def SpiralMLP_B1_Config():
        return SpiralConfig(layers=[2, 2, 4, 2],embed_dims=[64, 128, 320, 512], pretrained=None, swin_info="Spiral_S")

    @staticmethod
    def SpiralMLP_B2_Config():
        return SpiralConfig(layers=[2, 3, 10, 3],embed_dims=[64, 128, 320, 512], pretrained=None, swin_info="Spiral_S")

    @staticmethod
    def SpiralMLP_B3_Config():
        return SpiralConfig(layers=[3, 4, 18, 3],embed_dims=[64, 128, 320, 512], pretrained=None, swin_info="Spiral_M")

    @staticmethod
    def SpiralMLP_B4_Config():
        return SpiralConfig(layers=[3, 8, 27, 3],embed_dims=[64, 128, 320, 512], pretrained=None, swin_info="Spiral_L")

    @staticmethod
    def SpiralMLP_B5_Config():
        return SpiralConfig(layers=[3, 4, 24, 3],embed_dims=[96, 192, 384, 768], pretrained=None, swin_info="Spiral_L")

class DensePredict:
    def __init__(self):
        pass


if __name__ == '__main__':
    config = Classification.SpiralMLP_B1_Config()
    print(config)
    print(config.layers)
