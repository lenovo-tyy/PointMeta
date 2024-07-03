# Acknowledgement: built upon mmcv
import inspect
import warnings
from functools import partial
import copy 


class Registry:
    """一个将字符串映射到类的注册表。
    可以从注册表中构建已注册的对象。
    示例：
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(NAME='ResNet'))
    请参考 https://mmcv.readthedocs.io/en/latest/registry.html 了解高级用法。
    Args:
        name (str): 注册表名称。
        build_func(func, optional): 用于从注册表构建实例的构建函数。如果未指定``parent``或``build_func``，将使用``build_from_cfg``函数。如果指定了``parent``但未提供``build_func``，则将从``parent``继承``build_func``。默认值：None。
        parent (Registry, optional): 父注册表。在子注册表中注册的类可以从父注册表中构建。默认值：None。
        scope (str, optional): 注册表的作用域。它是搜索子注册表的关键。如果未指定，则作用域将是定义类的包的名称，例如 mmdet、mmcls、mmseg。默认值：None。
    """

    def __init__(self, name, build_func=None, parent=None, scope=None):
        # 初始化函数，设置注册表的基本属性和构建函数
        self._name = name  # 注册表名称
        self._module_dict = dict()  # 模块字典，将字符串映射到类
        self._children = dict()  # 子注册表字典
        self._scope = self.infer_scope() if scope is None else scope  # 注册表的作用域

        # self.build_func 将按以下优先级设置：
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        if build_func is None:  # 如果未提供构建函数
            if parent is not None:  # 如果存在父注册表
                self.build_func = parent.build_func  # 从父注册表继承构建函数
            else:  # 否则使用默认的构建函数 build_from_cfg
                self.build_func = build_from_cfg
        else:  # 如果提供了构建函数
            self.build_func = build_func  # 使用提供的构建函数
        if parent is not None:  # 如果存在父注册表
            assert isinstance(parent, Registry)
            parent._add_children(self)  # 将当前注册表添加为父注册表的子注册表
            self.parent = parent  # 设置父注册表
        else:
            self.parent = None  # 如果没有父注册表，则设置父注册表为None


    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._module_dict})'
        return format_str

    @staticmethod
    def infer_scope():
        """Infer the scope of registry.
        The name of the package where registry is defined will be returned.
        Example:
            # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            The scope of ``ResNet`` will be ``mmdet``.
        Returns:
            scope (str): The inferred scope name.
        """
        # inspect.stack() trace where this function is called, the index-2
        # indicates the frame where `infer_scope()` is called
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        split_filename = filename.split('.')
        return split_filename[0]

    @staticmethod
    def split_scope_key(key):
        """Split scope and key.
        The first scope will be split from key.
        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'
        Return:
            scope (str, None): The first scope.
            key (str): The remaining key.
        """
        split_index = key.find('.')
        if split_index != -1:
            return key[:split_index], key[split_index + 1:]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    def get(self, key):
        """Get the registry record.
        Args:
            key (str): The class name in string format.
        Returns:
            class: The corresponding class.
        """
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        else:
            # get from self._children
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # goto root
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(key)

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    def _add_children(self, registry):
        """Add children for a registry.
        The ``registry`` will be added as children based on its scope.
        The parent registry could build objects from children registry.
        Example:
            >>> models = Registry('models')
            >>> mmdet_models = Registry('models', parent=models)
            >>> @mmdet_models.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = models.build(dict(NAME='mmdet.ResNet'))
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert registry.scope not in self.children, \
            f'scope {registry.scope} exists in {self.name} registry'
        self.children[registry.scope] = registry

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, '
                            f'but got {type(module_class)}')

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f'{name} is already registered '
                               f'in {self.name}')
            self._module_dict[name] = module_class

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn(
            'The old API of register_module(module, force=False) '
            'is deprecated and will be removed, please use the new API '
            'register_module(name=None, force=False, module=None) instead.')
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
        """Register a module.
        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.
        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)
        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')
        # NOTE: This is a walkaround to be compatible with the old api,
        # while it may introduce unexpected bugs.
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or misc.is_seq_of(name, str)):
            raise TypeError(
                'name must be either of None, an instance of str or a sequence'
                f'  of str, but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(
                module_class=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(
                module_class=cls, module_name=name, force=force)
            return cls

        return _register


def build_from_cfg(cfg, registry, default_args=None):
    """
    从配置字典中构建一个模块。

    参数:
        cfg (edict): 配置字典。至少应该包含键 "NAME"。
        registry (:obj:`Registry`): 从中搜索类型的注册表。
        default_args (dict, optional): 默认参数。默认值为 None。

    返回:
        object: 构建的对象。
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg必须是字典类型，但得到的是 {type(cfg)}')
    if 'NAME' not in cfg:
        if default_args is None or 'NAME' not in default_args:
            raise KeyError(
                '`cfg` 或 `default_args` 必须包含键 "NAME"，但得到的是 {cfg}\n{default_args}')
    if not isinstance(registry, Registry):
        raise TypeError('registry必须是 mmcv.Registry 对象，但得到的是 {type(registry)}')

    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args必须是字典类型或者None，但得到的是 {type(default_args)}')

    obj_type = cfg.get('NAME')

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} 不在 {registry.name} 注册表中')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type必须是一个字符串或有效的类型，但得到的是 {type(obj_type)}')
    try:
        # 深拷贝传入的配置信息
        obj_cfg = copy.deepcopy(cfg)
        # 如果存在默认参数，将默认参数与传入配置信息合并
        if default_args is not None:
            obj_cfg.update(default_args)
        # 移除 'NAME' 键
        obj_cfg.pop('NAME')
        # 使用修改后的配置信息创建对象实例
        return obj_cls(**obj_cfg)

    except Exception as e:
        # 普通的TypeError不会打印类名。
        raise type(e)(f'{obj_cls.__name__}: {e}')

