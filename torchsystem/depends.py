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

from typing import Generator
from inspect import signature
from contextlib import ExitStack, contextmanager
from collections.abc import Callable
from functools import wraps

class Provider:
    def __init__(self):
        self.dependency_overrides = dict()
    
    def override(self, dependency: Callable, override: Callable):
        self.dependency_overrides[dependency] = override

class Dependency:
    def __init__(self, callable: Callable):
        self.callable = callable

def resolve(function: Callable, provider: Provider, *args, **kwargs):
    parameters = signature(function).parameters
    bounded = signature(function).bind_partial(*args, **kwargs)
    exit_stack = ExitStack()
    
    for name, parameter in parameters.items():
        if name not in bounded.arguments and isinstance(parameter.default, Dependency):
            dependency = parameter.default.callable
            if dependency in provider.dependency_overrides:
                dependency = provider.dependency_overrides[dependency]
            
            dep_args, dep_stack = resolve(dependency, provider)
            with dep_stack:
                dep_instance = dependency(*dep_args.args, **dep_args.kwargs)
            
            if isinstance(dep_instance, Generator):
                bounded.arguments[name] = exit_stack.enter_context(_managed_dependency(dep_instance))
            else:
                bounded.arguments[name] = dep_instance
    
    return bounded, exit_stack

@contextmanager
def _managed_dependency(generator: Generator):
    try:
        value = next(generator)
        yield value
    finally:
        next(generator, None)   

def Depends(callable: Callable):
    return Dependency(callable)

def inject(provider: Provider):
    def decorator(function: Callable):
        @wraps(function)
        def wrapper(*args, **kwargs):
            bounded, exit_stack = resolve(function, provider, *args, **kwargs)
            with exit_stack:
                return function(*bounded.args, **bounded.kwargs)
        return wrapper
    return decorator