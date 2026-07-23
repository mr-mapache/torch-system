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
from collections.abc import Callable

from torch import compile as compile
from torchsystem.depends import Depends as Depends
from torchsystem.depends import inject
from torchsystem.depends import Provider

class Compiler[T]:
    """
    Sometimes AGGREGATES may require complex initialization and be built from multiple sources, like database
    queries, API calls, or other external resources are required to build it. Each one with it's own lifecycle
    that may require some initialization method or cleanup after they are no longer needed.
     
    In the context of neural networks, AGGREGATES not only should be built but also compiled. Compilation
    is the process of converting a high-level neural network model into a low-level representation that can
    be executed on a specific hardware platform, and can be seen as an integral part of the process of building 
    the AGGREGATE. This process get even more complex when performing distributed training. 

    A `Compiler` is a class that compiles a pipeline of functions to be executed in sequence in order
    to build an a low-level representation of the AGGREGATE. Since some compilation steps sometimes
    requires runtime information, the `Compiler` provides a mechanism to inject dependencies into
    the pipeline.

    Attributes:
        steps (list[Callable[..., Any]]): A list of functions to be executed in sequence.

    Methods:
        compile:
            Execute the pipeline of functions in sequence. The output of each function is passed as
            input to the next function. The compiled AGGREGATE should be returned as a result of the
            execution of the pipeline.

        step:
            A decorator that adds a function to the pipeline. The function should take as input the
            output of the previous function in the pipeline and return the input of the next function
            in the pipeline.

    Example:
        ```python	
        from logging import getLogger
        from torch import cuda
        from torch.nn import Module
        from torchsystem.compiler import Compiler
        from torchsystem.compiler import Depends
        from torchsystem.compiler import compile

        compiler = Compiler()
        logger = getLogger(__name__)

        def device():
            return 'cuda' if cuda.is_available() else 'cpu'

        def epoch():
            raise NotImplementedError('Override this function with a concrete implementation')

        @compiler.step
        def build_classifier(model: Module, criterion: Module, optimizer: Module):
            logger.info(f'Building classifier')
            logger.info(f'- model: {model.__class__.__name__}')
            logger.info(f'- criterion: {criterion.__class__.__name__}')
            logger.info(f'- optimizer: {optimizer.__class__.__name__}')
            return Classifier(model, criterion, optimizer)

        @compiler.step
        def move_to_device(classifier: Classifier, device = Depends(device)):
            logger.info(f'Moving classifier to device: {device}')
            return classifier.to(device)

        @compiler.step
        def compile_classifier(classifier: Classifier):
            logger.info(f'Compiling classifier')
            return compile(classifier)

        @compiler.step
        def set_epoch(classifier: Classifier, epoch: int = Depends(epoch)):
            logger.info(f'Setting classifier epoch: {epoch}')
            classifier.epoch = epoch
            return classifier
        ...

        compiler.dependency_overrides[epoch] = lambda: 10
        classifier = compiler.compie(model, criterion, optimizer)
        assert classifier.epoch == 10
        ```
    """
    def __init__(
        self,
        *,
        provider: Provider | None = None,
    ):
        """
        Initialize the Compiler.

        Args:
            provider (Provider): The dependency provider. Defaults to None.
        """
        self.steps = list[Callable]()
        self.provider = provider or Provider()
    
    @property
    def dependency_overrides(self) -> dict:
        """
        Get the dependency overrides. Dependency overrides are used to inject dependencies into the
        pipeline. This is useful for late binding, testing and changing the behavior of the compiler
        in runtime.

        Returns:
            dict: The dependency map.

        Example:        
            ```python	
            def device():...
            ...

            compiler.dependency_overrides[device] = lambda: 'cuda' if cuda.is_available() else 'cpu'
            ```
        """
        return self.provider.dependency_overrides
    
    
    def override(self, dependency: Callable, implementation: Callable):
        """
        Overrides a dependency with an implementation. 

        Args:
            dependency (Callable): The dependency function to override.
            implementation (Callable): The implementation of the function.
        """
        self.dependency_overrides[dependency] = implementation
    

    def step(self, callable: Callable) -> Any:
        """
        Add a function to the pipeline. The function should take as input the output of the previous
        function in the pipeline and return the input of the next function in the pipeline.

        Args:
            callable (Callable): The function to be added to the pipeline.

        Returns:
            Any: The requirements for the next step in the pipeline.
        """
        injected = inject(self.provider)(callable)
        self.steps.append(injected)
        return injected
    
    def compile(self, *args, **kwargs) -> T | Any | None:
        """
        Execute the pipeline of functions in sequence. The output of each function is passed as input
        to the next function. The compiled AGGREGATE should be returned by the last function in the pipeline.
        
        Returns:
            T: The compiled AGGREGATE.
        """
        result = None
        for step in self.steps:
            if result is None:
                result = step(*args, **kwargs)
            else:
                result = step(*result) if isinstance(result, tuple) else step(result)
        return result