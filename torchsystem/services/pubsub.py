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

from torchsystem.depends import inject, Provider
from torchsystem.depends import Depends as Depends
        
class Subscriber:
    """
    A SUBSCRIBER is a component that listens for messages published processes them accordingly.

    Unlike a CONSUMER, a SUBSCRIBER receives messages only from the topics it has subscribed to
    and it's the PUBLISHER's responsibility to route the messages accordingly.

    Methods:
        register:
            Registers a message type and its corresponding handler function.

        subscribe:
            Decorator for registering a handler function to one or more topics.

        receive:
            Receives a message from a given topic and triggers the corresponding handler functions
            to process it.

    Example:
        ```python	
        from torchsystem import Depends
        from torchsystem.services import Subscriber
        from torchsystem.services import Publisher

        subscriber = Subscriber()
        metricsdb = []

        def metrics():
            return metricsdb

        @subscriber.subscribe('loss', 'accuracy')
        def store_metric(metric, metrics: list = Depends(metrics)):
            metrics.append(metric)

        @subscriber.subscribe('accuracy')
        def on_accuracy_to_high(metric):
            if metric > 0.99:
                raise StopIteration
            
        publisher = Publisher()
        publisher.register(subscriber)

        publisher.publish(0.1, 'loss')
        publisher.publish(0.9, 'accuracy')
        assert metricsdb == [0.1, 0.9]

        try:
            publisher.publish(1.0, 'accuracy')
        except StopIteration:
            print("Early stopping") 
        ```
    """
    def __init__(
        self, 
        name: str | None = None,
        *,
        provider: Provider | None = None,
    ):
        self.name = name
        self.provider = provider or Provider()
        self.handlers = dict[str, list[Callable[..., None]]]()
    
    @property
    def dependency_overrides(self) -> dict:
        """
        Returns the dependency overrides for the subscriber. This is useful for late binding,
        testing and changing the behavior of the subscriber in runtime.

        Returns:
            dict: A dictionary of the dependency map.

        Example:
            ```python	
            subscriber = Subscriber()
            ...

            subscriber.dependency_overrides[db] = lambda: []
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
    
    def register(self, topic: str, wrapped: Callable[..., None]) -> None:       
        """
        Registers a handler function with a given topic.

        Args:
            topic (str): The topic to register the handler function to.
            wrapped (Callable[..., None]): The handler function to register.
        """ 
        injected = inject(self.provider)(wrapped)
        self.handlers.setdefault(topic, []).append(injected)
    
    def subscribe(self, *topics: str) -> Callable[..., None]:
        """
        Decorator for registering a handler function to one or more topics. 

        Args:

            *topics (str): The topics to register the handler function to.

        Returns:
            Callable[..., None]: The decorated handler function.

        Example:
            ```python	
            subscriber = Subscriber()

            @subscriber.subscribe('loss', 'accuracy')
            def store_metric(metric, metrics = Depends(metrics)):
                ...
            ```
        """
        def handler(wrapped: Callable[..., None]):
            for topic in topics:
                self.register(topic, wrapped)
            return wrapped
        return handler

    def receive(self, message: Any, topic: str):
        """
        Receives a message from a given topic and triggers the corresponding handler functions
        to process it. This is called by the PUBLISHER but is also useful for deliver messages
        between handlers directly.

        Args:
            message (Any): The message to process.
            topic (str): The topic to process the message from.

        Example:
            ```python	
            subscriber = Subscriber()

            @subscriber.subscribe('metrics')
            def store_metric(metrics: list):
                for metric in metrics:
                    subscriber.receive(metric, metric['name'])

            @subscriber.subscribe('loss')
            def on_loss(loss):
                print(f"Loss: {loss['value']}")

            @subscriber.subscribe('accuracy')
            def on_accuracy(accuracy):
                print(f"Accuracy: {accuracy['value']}")

            subscriber.receive([
                {'name': 'loss', 'value': 0.1}, 
                {'name': 'accuracy', 'value': 0.9}], 
            topic='metrics')

            # Output:
            # Loss: 0.1
            # Accuracy: 0.9
            ```        
        """
        for handler in self.handlers.get(topic, []):
            handler(message)

class Publisher:
    """
    A PUBLISHER is a component that sends messages to one or more SUBSCRIBERS. Unlike a PRODUCER
    It's the PUBLISHER's responsibility to route the messages to the corresponding SUBSCRIBERS.
    
    Methods:
        register: Registers one or more SUBSCRIBERS to the PUBLISHER.
        publish: Publishes a message to one or more SUBSCRIBERS based on the topic.

    Example:    
        ```python	
        from torchsystem.services import Subscriber

        subscriber = Subscriber()

        @subscriber.subscribe('loss')
        def on_loss(loss):
            print(f"Loss: {loss}")

        publisher = Publisher()
        publisher.register(subscriber)
        publisher.publish(0.1, 'loss')
        ```
    """
    def __init__(self) -> None:
        self.subscribers = list[Subscriber]()

    def publish(self, message: Any, topic: str) -> None:
        """
        Publishes a message to one or more SUBSCRIBERS based on the topic.

        Args:
            message (Any): The message to publish.
            topic (str): The topic to publish the message to.
        """
        for subscriber in self.subscribers:
            subscriber.receive(message, topic)

    def register(self, *subscribers: Subscriber) -> None:
        """
        Registers one or more SUBSCRIBERS to the PUBLISHER.
        """
        for subscriber in subscribers:
            self.subscribers.append(subscriber)