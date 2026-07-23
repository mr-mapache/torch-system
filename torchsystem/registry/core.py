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
from copy import deepcopy
from inspect import signature

def cls_signature(cls: type, excluded_args: list[int] | None = None, excluded_kwargs: set[str] | None = None):
    excluded_args = excluded_args or []
    excluded_kwargs = excluded_kwargs or set()
    cls_signature = {}
    for index, (key, value) in enumerate(signature(cls).parameters.items()):
        if index not in excluded_args and key not in excluded_kwargs:
            cls_signature[key] = value.annotation.__name__ if value.annotation != value.empty else "Any"
    return deepcopy(cls_signature)


def handle_arg(arg):
    if hasattr(arg, '__model__arguments__'):
        arguments = getattr(arg, '__model__arguments__')
        if arguments:
            return deepcopy({
                'name': getattr(arg, '__model__name__') if hasattr(arg, '__model__name__') else arg.__class__.__name__,
                'arguments': getattr(arg, '__model__arguments__')
            })
        else:
            return deepcopy(getattr(arg, '__model__name__') if hasattr(arg, '__model__name__') else arg.__class__.__name__)
    else:
        return arg    

def cls_parse_args(args: tuple[Any], excluded_args: list[int], signature: dict[str, str]) -> dict[str, Any]: 
    kargs = {}
    for index, (arg, key) in enumerate(zip(args, signature.keys())):
        if index not in excluded_args:
            kargs[key] = handle_arg(arg)
    return deepcopy(kargs)

def cls_parse_kwargs(kwargs: dict[str, Any], excluded_kwargs: set[str]) -> dict[str, Any]:
    kargs = {}
    for key, arg in kwargs.items():
        if key not in excluded_kwargs:
            kargs[key] = handle_arg(arg)
    return deepcopy(kargs)
    
def cls_override_init(
    cls: type,
    excluded_args: list[int] | None = None,
    excluded_kwargs: set[str] | None = None,
    name: str | None = None
):
    init = getattr(cls, '__init__')
    signature = cls_signature(cls)
    excluded_args = excluded_args or []
    excluded_kwargs = excluded_kwargs or set()
    def init_wrapper(obj, *args, **kwargs):
        init(obj, *args, **kwargs)
        arguments = cls_parse_args(args, excluded_args, signature) | cls_parse_kwargs(kwargs, excluded_kwargs)
        setattr(obj, '__model__arguments__', arguments)
        if name:
            setattr(obj, '__model__name__', name)
    setattr(cls, '__init__', init_wrapper) 
    return cls 