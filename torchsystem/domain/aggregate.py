# Copyright 2025 Eric M. Cardozo (Eric Hermosis).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 
#
# 
# For inquiries, visit the documentation at eric-m-cardozo.github.io/TorchSystem/


from abc import ABC
from typing import Any
from typing import Literal
from torch.nn import Module 

type Phase = Literal['train', 'evaluation'] | str

class Aggregate(Module, ABC):
    """
    An AGGREGATE is a cluster of associated objects that we treat as a unit for the purpose
    of data changes. Each AGGREGATE has a root and a boundary. The boundary defines what is
    inside the AGGREGATE. The root is a single, specific ENTITY contained in the AGGREGATE and
    provides the IDENTITY of the AGGREGATE. The root is the only member of the AGGREGATE that
    outside objects are allowed to hold references to.

    In deep learning, an AGGREGATE consist not only of a neural network, but also several other
    components such as optimizers, schedulers, tokenizers, etc.  For example, a transformer model
    is just a neural network, and in order to perform tasks such as text completion or translation,
    it needs to be part of an AGGREGATE that includes other components like a tokenizer. The AGGREGATE
    is responsible for coordinating the interactions between these components.

    Attributes:
        id (Any): The id of the AGGREGATE ROOT. It should be unique within the AGGREGATE boundary.
        phase (str): The phase of the AGGREGATE.
        events (Events): The domain events of the AGGREGATE.

    Methods:
        onphase:
            A hook that is called when the phase changes. Implement this method to add custom behavior.

        onepoch:
            A hook that is called when the epoch changes. Implement this method to add custom behavior.

    Example:
        ```python	
        from torch import Tensor
        from torch.nn import Module
        from torch.optim import Optimizer
        from torchsystem import Aggregate
        from torchsystem.registry import gethash

        class Classifier(Aggregate):
            def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
                super().__init__()
                self.epoch = 0
                self.model = model
                self.criterion = criterion
                self.optimizer = optimizer

            @property
            def id(self) -> str:
                return gethash(self.model) # See the registry module for more information.

            def onepoch(self):
                print(f'Epoch: {self.epoch}')

            def onphase(self):
                print(f'Phase: {self.phase}')

            def forward(self, input: Tensor) -> Tensor:
                return self.model(input)
            
            def loss(self, output: Tensor, target: Tensor) -> Tensor:
                return self.criterion(output, target)

            def fit(self, input: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
                self.optimizer.zero_grad()
                output = self(input)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                return output, loss

            def evaluate(self, input: Tensor, target: Tensor) -> tuple[Tensor, Tensor]: 
                output = self(input)
                loss = self.loss(output, target)
                return output, loss
        ```
    """
    
    def __init__(self):
        super().__init__()
        self.__id = None


    @property
    def id(self) -> Any:
        """
        The identity of the AGGREGATE ROOT.

        In Domain-Driven Design, the AGGREGATE ROOT is the single entry point 
        to a cluster of domain objects, and its identity uniquely distinguishes 
        the AGGREGATE within its boundary. This property returns that identity.

        Key points:
            - Immutable once assigned to preserve AGGREGATE consistency and invariants.
            - Accessing the ID before initialization raises an error to prevent 
            unsafe operations on an uninitialized AGGREGATE ROOT.
            - Ensures that all domain operations relying on identity are deterministic 
            and safe.

        Raises:
            ValueError: If the AGGREGATE ROOT's ID has not been initialized.
        
        Returns:
            Any: The unique identifier of the AGGREGATE ROOT within its boundary.
        """
        if self.__id is None:
            raise ValueError("Aggregate ID is not initialized")
        return self.__id


    def initialize(self, id: Any):
        """
        Initializes the identity of the AGGREGATE ROOT.

        This method allows deferred assignment of identity when the creation of 
        the AGGREGATE depends on external factors (e.g., database-generated IDs, 
        hashes of components). Once set, the ID is immutable to uphold the AGGREGATE's 
        invariants and consistency rules.

        Args:
            id (Any): A unique identifier for the AGGREGATE ROOT.

        Raises:
            ValueError: If the AGGREGATE ROOT already has an initialized identity.
        """
        if self.__id is not None:
            raise ValueError("Aggregate ID already initialized")
        self.__id = id


    def is_initialized(self) -> bool:
        """
        Checks whether the AGGREGATE ROOT has an assigned identity.

        Some domain operations require that the AGGREGATE ROOT has a valid identity 
        before executing (e.g., persisting the aggregate, emitting domain events). 
        This method allows the domain logic to verify readiness safely.

        Returns:
            bool: True if the AGGREGATE ROOT has an initialized identity, False otherwise.

        Warning:
            If the `id` property is overridden in a subclass (e.g., computed dynamically 
            instead of stored in `self.__id`), this method may return False even if the 
            aggregate has a valid identity.
        """
        return self.__id is not None


    @property
    def phase(self) -> Literal['train', 'evaluation']:
        """
        The phase of the AGGREGATE. The phase is a property of neural networks that not only describes
        the current state of the network, but also determines how the network should behave. 
        
        During the training phase, the network stores the gradients of the weights and biases, and uses them
        to update the weights and biases. During the evaluation phase, the network does not store the gradients
        of the weights and biases, and does not update the weights and biases.

        Returns:
            Literal['train', 'evaluation']: The current phase of the AGGREGATE.
        """
        return 'train' if self.training else 'evaluation'
    
    @phase.setter
    def phase(self, value: Phase):
        """
        Set the phase of the AGGREGATE. When the phase changes, the onphase hook method is called.
        The phase will be set to 'train' if the value is 'train', otherwise it will be set to 'evaluation'.
        
        Changing the phase of the AGGREGATE will set all the modules in the AGGREGATE to the training or
        evaluation mode respectively.

        Args:
            value (str): The phase of the AGGREGATE. It can be either 'train' or 'evaluation'.
        """
        self.train() if value == 'train' else self.eval()
        self.onphase()

    def onphase(self):
        """
        A hook that is called when the phase changes. Implement this method to add custom behavior.
        """

    def onepoch(self):
        """
        A hook that is called when the epoch changes. Implement this method to add custom behavior.
        """

    def __setattr__(self, name, value):
        if name == 'epoch' and hasattr(self, 'epoch'):
            super().__setattr__(name, value)
            self.onepoch()
        else:        
            super().__setattr__(name, value)