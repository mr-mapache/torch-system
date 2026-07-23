# Copyright 2024 Eric M. Cardozo (Eric Hermosis)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You can obtain a copy of the License at:
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This software is distributed "AS IS," without warranties or conditions.
# See the License for specific terms.
#
# For inquiries, visit: eric-m-cardozo.github.io/TorchSystem/

from typing import Any
from json import dumps
from hashlib import md5
from copy import deepcopy
from typing import overload
from typing import Optional
from typing import Any
from collections.abc import Callable 
from torch.nn import Module
from torch.serialization import add_safe_globals
from torchsystem.registry import core

def getarguments(obj: object) -> dict[str, Any]:
    """
    A function to get the arguments captured by the __init__ method of a class when an instance of the
    given type is initialized.

    Args:
        obj (object): The object to get the arguments from.

    Raises:
        AttributeError: If the object was not registered.

    Returns:
        dict[str, Any]: The arguments captured by the __init__ method of the object.
    """
    if not hasattr(obj, '__model__arguments__'):
        raise AttributeError(f"The object {obj} was not registered")
    return getattr(obj, '__model__arguments__')

def getname(obj: object) -> str:
    """
    A function to get the name of the object. If the object has a __model__name__ attribute, it will be
    returned. Otherwise, the class name will be returned.

    Args:
        obj (object): The object to get the name from.

    Returns:
        str: The name of the object.
    """
    if hasattr(obj, '__model__name__'):
        return getattr(obj, '__model__name__')
    else:
        return obj.__class__.__name__

def gethash(obj: object) -> str:
    """
    A function to get an unique deterministic hash of the object calculated from the name and the arguments
    captured by the __init__ method of the object. If the object was not registered, an AttributeError will be
    raised. The hash will be calculated using the md5 algorithm by default but can be setted manually using the
    sethash function.

    Args:
        obj (object): The object to get the hash from.

    Returns:
        str: The hash of the object.

    Raises:
        AttributeError: If the object was not registered and does not have a hash setted. 
    """
    if not hasattr(obj, '__model__arguments__') and not hasattr(obj, '__model__hash__'):
        raise AttributeError(f"The object {obj} was not registered and does not have a hash")
    
    if hasattr(obj, '__model__hash__'):
        return getattr(obj, '__model__hash__')
    else:
        arguments = getarguments(obj)
        return md5((getname(obj) + dumps(arguments)).encode()).hexdigest()

def sethash(obj: object, hash: str | None = None) -> None:
    """
    A function to set the hash of the object. If the hash is not provided, it will be calculated using the
    md5 algorithm from the name and the arguments captured by the __init__ method of the object by default.
    If a hash is provided, it will be setted as the hash of the object.

    Args:
        obj (object): _description_
        hash (str, optional): _description_. Defaults to None.
    """
    if not hash:
        setattr(obj, '__model__hash__', gethash(obj))
    else:
        setattr(obj, '__model__hash__', hash)

def setname(obj: object, name: str | None = None) -> None:
    """
    A function to set the name of the object. If the name is not provided, it will be retrieved from the
    class name. If a name is provided, it will be setted as the name of the object.

    Args:
        obj (object): The object to set the name.
        name (str, optional): The name to set. Defaults to None.
    """
    if not name:
        setattr(obj, '__model__name__', getname(obj))
    else:
        setattr(obj, '__model__name__', name)


def getmetadata(obj: object) -> dict[str, Any]:
    """
    A function to get the metadata of the object. The metadata is a dictionary containing the name, the
    arguments and the hash of the object.

    Args:
        obj (object): The object to get the metadata.

    Returns:
        dict[str, Any]: The metadata of the object.
    """
    hash_field = {'hash': getattr(obj, '__model__hash__')} if hasattr(obj, '__model__hash__') else {}
    name_field = {'name': getattr(obj, '__model__name__')} if hasattr(obj, '__model__name__') else {}
    arguments_field = {'arguments': getattr(obj, '__model__arguments__')} if hasattr(obj, '__model__arguments__') else {}
    return deepcopy(hash_field | name_field | arguments_field)




@overload
def register(cls: type, excluded_args: list[int] | None = None, excluded_kwargs: set[str] | None = None):
    """
    A function to override the __init__ method of a class in order to capture the arguments passed
    to it when an instance of the given type is initialized. Can be used as a raw decorator. 
 
    Args:
        cls (type): The class to override the __init__ method.
        excluded_args (list[int], optional): The indexes of the arguments to exclude from the capture. Defaults to None.
        excluded_kwargs (set[str], optional): The names of the keyword arguments to exclude from the capture. Defaults to None.
    
    Returns:
        type: The class with the __init__ method overriden.

    Example: 
        ```python	
        @register
        class Bar:
            def __init__(self, x: int, y: float, z: str):
                pass
                
        class Foo:
            def __init__(self, x: int, y: float, z: str):
                pass
                
        register(Foo, excluded_args=[0], excluded_kwargs=['z'])
        ```	
    """
    ...

@overload
def register(cls: str, excluded_args: list[int] | None = None, excluded_kwargs: set[str] | None = None):
    """
    A decorator to override the __init__ method of a class in order to capture the arguments passed 
    to it when an instance of the given type is initialized. Can be used as a decorator with a name
    argument to set the name of the class in the registry.

    Args:
        cls (str): The name of the class in the registry. Will be retrieved when calling the getname function.
        excluded_args (list[int], optional): The indexes of the arguments to exclude from the capture. Defaults to None.
        excluded_kwargs (set[str], optional): The names of the keyword arguments to exclude from the capture. Defaults to None.
    
    Returns:
        type: A decorator to override the __init__ method of a class.

    Example: 
        ```python	
        @register('bar')
        class Bar:
            def __init__(self, x: int, y: float, z: str):
                pass
        ```
    """
    ...

def register(cls: type | str | None, excluded_args: list[int] | None = None, excluded_kwargs: set[str] | None = None) -> type | Callable[[type], type]:
    """
    A function to override the __init__ method of a class in order to capture the arguments passed 
    to it when an instance of the given type is initialized. Can be used as a raw decorator or as a
    decorator with a name argument to set the name of the class in the registry.

    Args:
        cls (type | str): The class to override the __init__ method or the name of the class in the registry.
        excluded_args (list[int], optional): The indexes of the arguments to exclude from the capture. Defaults to None.
        excluded_kwargs (set[str], optional): The names of the keyword arguments to exclude from the capture. Defaults to None.

    Returns:
        type | Callable: The class with the __init__ method overriden or a decorator to override the __init__ method of a class.
    """
    if isinstance(cls, type):
        return core.cls_override_init(cls, excluded_args, excluded_kwargs)
    elif isinstance(cls, str) or cls is None:
        def wrapper(type: type):
            return core.cls_override_init(type, excluded_args, excluded_kwargs, cls)
        return wrapper

class Registry[T]:
    """
    A class to register and retrieve types and their signatures. It acts as collection of types and is usefull in cases
    where a python object needs to be created dynamically based on a string name.

    Attributes:
        types (dict): a dictionary of registered types.
        signatures (dict): a dictionary of registered types signatures.

    Methods:
        register: 
            a decorator to register a type.
        get: 
            get a registered type by name.
        keys: 
            get the list of registered type names.
        signature: 
            get the signature of a registered type by.
    
    
    Example:
        ```python	
        from mlregistry.registry import Registry

        registry = Registry()

        @registry.register
        class Foo:
            def __init__(self, x: int, y: float, z: str):
                self.x = x
                self.y = y
                self.z = z

        instance = registry.get('Foo')(1, 2.0, '3') # instance of Foo
        signature = registry.signature('Foo') # {'x': 'int', 'y': 'float', 'z': 'str'}
        keys = registry.keys() # ['Foo']
        ```
    """
    def __init__(self):
        self.types = dict()
        self.signatures = dict()

    @overload
    def register(self, cls: str, excluded_args: list[int] | None = None, excluded_kwargs: set[str] | None = None) -> Callable[[type[T]], type[T]]:
        ...

    @overload
    def register(self, cls: type, excluded_args: list[int] | None = None, excluded_kwargs: set[str] | None = None) -> type[T]:
        ...

    def register(self, cls: type | str, excluded_args: list[int] | None = None, excluded_kwargs: set[str] | None = None) -> type[T] | Callable[[type[T]], type[T]]:
        """
        Register a class type with the registry and override its __init__ method in order to capture the arguments
        passed to the constructor during the object instantiation. The captured arguments can be retrieved using the
        `getarguments` function. The `excluded_args` and `excluded_kwargs` parameters can be used to exclude the arguments
        from being captured.

        Types can be registered after their definition or using the register method as a decorato and optionally setting the
        name of the class in the registry.

        Args:
            cls (type | str): the class type to be registered
            excluded_args (list[int], optional): The list of argument indexes to be excluded. Defaults to None.
            excluded_kwargs (set[str], optional): The dictionary of keyword arguments to be excluded. Defaults to None.

        Returns:
            type[T] | Callable: the registered class type.
        """
        
        
        if isinstance(cls, type):
            self.types[cls.__name__] = cls
            self.signatures[cls.__name__] = core.cls_signature(cls, excluded_args, excluded_kwargs)
            core.cls_override_init(cls, excluded_args, excluded_kwargs)
            if issubclass(cls, Module):
                add_safe_globals([cls])
            return cls
            
        elif isinstance(cls, str):
            def wrapper(type_: type):
                self.types[cls] = type_
                self.signatures[cls] = core.cls_signature(type_, excluded_args, excluded_kwargs)
                core.cls_override_init(type_, excluded_args, excluded_kwargs, cls)
 
                if issubclass(type_, Module):
                    add_safe_globals([type_])

                return type_
            return wrapper
        else:
            raise TypeError("The argument should be a class type or a string")
        

    def get(self, name: str) -> Optional[type[T]]:
        """
        Get a registered type by name from the registry.

        Args:
            name (str): the name of the type to be retrieved

        Returns:
            Optional[type[T]]: the registered type if found, otherwise None
        """
        return self.types.get(name, None)

    def keys(self) -> list[str]:
        '''
        Get the list of registered type names.

        Returns:
            list[str]: the list of registered type names
        '''
        return list(self.types.keys())

    def signature(self, name: str) -> Optional[dict[str, str]]:
        '''
        Get the signature of a registered type by name.

        Parameters:
            name (str): the name of the type to be retrieved.

        Returns:
            dict[str, str]: the signature of the registered type.
        '''
        return self.signatures.get(name, None)