from openpoints.utils import registry

# 创建名为 MODELS 的模型注册表
MODELS = registry.Registry('models')


def build_model_from_cfg(cfg, **kwargs):
    """
    根据配置文件中的 'NAME' 参数构建模型。

    参数:
        cfg (eDICT): 配置文件，包含模型名称 'NAME'。
        **kwargs: 关键字参数

    返回:
        Model: 根据 'NAME' 指定的模型构建而成的模型。
    """
    return MODELS.build(cfg, **kwargs)
