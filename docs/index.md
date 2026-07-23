# Welcome to the TorchSystem documentation.

This framework will help you to create powerful and scalable systems using the PyTorch library. It is designed under the principles of domain driven design (DDD) and includes built-in message patterns and a robust dependency injection system. It enables the creation of stateless, modular service layers and robust domain models. This design facilitates better separation of concerns, testability, and scalability, making it ideal for complex IA training systems.

## Disclaimer:

TorchSystem is an independent, open-source project and is not affiliated with, endorsed by, or sponsored by Meta or PyTorch. The name is for descriptive purposes only and does not imply any official connection.


## Table of contents:

- [Introduction](#introduction)
- [Installation](#installation)
- [Getting Started](#getting-started) 
- [Features](#features)
- [License](#license)

## Introduction

In domain driven design, an aggregate is a cluster of associated objects that we treat as a unit for the purpose of data changes. It acts as a boundary around its constituent objects, encapsulating their behavior and ensuring that all changes to its state occur through well-defined entry points.

In the context of deep learning, a model not only consists of a neural network but also a set of associated objects that are necessary for the tasks it performs, such as loss functions, tokenizers, classification heads etc. This cluster of objects defines an aggregate.

While aggregates are in charge of data, in order to perform actions, we need to define services. Services are stateless operations that fulfill domain-specific tasks. For example, when training a neural network, the model doesn't own the data on which it is trained or how the training is performed. The training process is a stateless operation that resides outside the model and should be defined as a service.

Services may produce data, such as events, metrics, or logs, that are not their responsibility to handle. This introduces the need for a messaging system that allows services to communicate with each other.

With all this in mind, the need for a well-defined framework that defines aggregates and handles service interactions becomes evident. While it is up to the developer to define his domain, this framework provides a set of tools to facilitate their implementation.

## Installation

To install the framework, you can use pip:

```bash
pip install torchsystem
```

The framework is written in pure python and doesn't require any infrastructure. 

## Getting Started

### Aggregates

Usually, a machine learning model consists of more than just a neural network. In practice, it involves a cluster of interrelated objects that need to work together to perform a specific task. For example, in a classification problem, you may need:

- The neural network itself

- A loss function to guide training

- An optimizer to update parameters

- Metrics to evaluate performance

- Additional logic for training, evaluation, or preprocessing

All these components form a cohesive unit that must be managed consistently. To reflect this in our code and ensure proper encapsulation, we can define an `Aggregate`.

```python 
from torch import Tensor 
from torch.nn import Module
from torch.optim import Optimizer
from torchsystem import Aggregate
from torchsystem.registry import gethash, getname

class Classifier(Aggregate):
    def __init__(self, nn: Module, criterion: Module, optimizer: Optimizer):
        super().__init__()
        self.epoch = 0
        self.nn = nn
        self.criterion = criterion
        self.optimizer = optimizer
        self.name = getname(nn)
        self.hash = gethash(nn)

    def forward(self, input: Tensor) -> Tensor:
        return self.nn(input)
     
    def loss(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return self.criterion(outputs, targets)

    def fit(self, inputs: Tensor, targets: Tensor) -> Tensor:
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def evaluate(self, inputs: Tensor, targets: Tensor) -> Tensor: 
        outputs = self(inputs)
        return self.loss(outputs, targets)
```

This aggregate wraps all the essential components of a classification model and exposes clear methods (fit and evaluate) for performing domain-specific tasks. 

By using an aggregate:

- State changes are managed consistently.

- Training and evaluation logic is encapsulated within the model itself.

- Additional components, such as metrics or logging, can be easily integrated.

This approach ensures that your model is more than just a network—it’s a self-contained, manageable unit ready to interact with services and pipelines.

Note: We imported two utility functions from the registry module: `gethash` and `getname`.

- `gethash(nn)` provides a locally unique identifier for the neural network.

- `getname(nn)` provides a human-readable, non-unique name for the network.

These attributes are part of the model’s overall identity, helping to distinguish and reference the model in logs, checkpoints, and tracking systems. 

To use functions from the registry module, you first need to register your neural network with the framework. For example:

```python 
from model import MLP
from torchsystem import registry

registry.register(MLP)
```
For more details and advanced usage, see the full documentation.

### Services

Let's now train our aggregate. The training process involves orchestrating data, batching, and optimization steps, which are external to the model’s internal state . To manage this, we use **services**. Services are stateless operations that act on aggregates and orchestrate domain-specific tasks like training or evaluation without directly modifying the model’s internal logic.

The service layer often produces data that needs to be communicated to external consumers, such as metrics, logs, or checkpoints. However, it is best not to couple services directly to any infrastructure. Events provide a clean, decoupled mechanism to notify other components of these outcomes, allowing consumers to handle persistence, monitoring, or other side effects independently of the service logic.

Let's implement the service layer with `torchsystem`:

```python
from typing import Iterable 
from torch import Tensor
from torch import inference_mode
from torchsystem.depends import Depends, Provider
from torchsystem.services import Service, Producer, event 
from src.classifier import Classifier 

provider = Provider()
producer = Producer() 
service = Service(provider=provider)

def device() -> str:...

@event
class Trained:
    model: Classifier 
    results: dict[str, Tensor] 

@event
class Evaluated:
    model: Classifier
    results: dict[str, Tensor]

@service.handler
def train(model: Classifier, loader: Iterable[tuple[Tensor, Tensor]], device: str = Depends(device)):
    model.train()
    for inputs, targets in loader: 
        inputs, targets = inputs.to(device), targets.to(device)  
        loss = model.fit(inputs, targets)
    producer.dispatch(Trained(model, {"loss": loss}))

@service.handler
def evaluate(model: Classifier, loader: Iterable[tuple[Tensor, Tensor]], device: str = Depends(device)):
    model.eval()
    with inference_mode():
        for inputs, targets in loader: 
            inputs, targets = inputs.to(device), targets.to(device)  
            loss = model.evaluate(inputs, targets)
        producer.dispatch(Evaluated(model, {"loss": loss})) 
```

Notice that the service it is completely decoupled from the implementation of the domain. It's only task is to orchestrate the training process and produce events from it. It doesn't provide any storage logic or data manipulation, only stateless training logic. Now you can now build a whole data storage system, logging or any other service you need around this simple service, without modifying it further.

Note: `torchsystem` uses a built-in dependency injection system to provide services with the resources they need without hardcoding them.

- `Provider` manages available dependencies and allows overriding them in different contexts.

- `Depends` declares that a function parameter should be automatically injected with a dependency (e.g., the device or a collection of models).

Those dependencies can be overriden later.

### Consumers

Consumers are components that react to events produced by services. They handle side effects such as logging, metrics collection, or persisting model checkpoints, without modifying the service or aggregate logic. This keeps the system decoupled and maintainable.

Let's create a consumer for our training service:

```python
from os import makedirs
from torch import save
from torchsystem import Depends
from torchsystem.services import Consumer
from src.training import provider
from src.training import Trained, Evaluated 

consumer = Consumer(provider=provider)

@consumer.handler
def bump_epoch(event: Trained):
    event.model.epoch += 1 

@consumer.handler
def log_metrics(event: Trained | Evaluated):
    print("-----------------------------------------------------------------")
    print(
        f"Epoch: {event.model.epoch}, "
        f"Average loss: {event.results['loss'].item()}, "
    )
    print("-----------------------------------------------------------------")

@consumer.handler
def persist_model(event: Evaluated):
    makedirs(f"data/weights", exist_ok=True)
    path = f"data/weights/{event.model.name}-{event.model.hash}.pth"
    checkpoint = {
        'epoch': event.model.epoch,
        'nn': event.model.nn.state_dict(),
        'optimizer': event.model.optimizer.state_dict()
    }
    save(checkpoint, path) 
    print(f"Saved model weights at: {path}")
```

Note: We attach the provider created in the training service because consumers also support dependency injection with the same fashion. This of course is completly optional, like everything else in the library. 

Consumers rely on type annotations to determine which events their handlers should react to. Unions and generics are supported: if a handler is declared with a union as an argument, it will listen to all events included in that union. In this example, for instance, metrics are logged during both the training and evaluation phases.

In this example:

- The consumer reacts to a Trained event and increments the epoch counter.

- When it receives either a Trained or Evaluated event, it prints the results.

- Each time an Evaluated event is received, it saves a checkpoint.

You can customize consumers to your needs, injecting any infrastructure you like, such as TensorBoard, a database, or other logging systems. If you prefer not to plug in infrastructure yet, you can use the pub/sub utilities provided by the library.

### Pubsub

Let's say you haven't decided which infrastructure you want to use for logging metrics when writing the consumer. Let's see how we can solve this issue using the Pub/Sub pattern. We'll add a publisher and an additional handler to the consumer.

```python
from dataclasses import dataclass

publisher = Publisher()

@dataclass
class Metric:
    name: str
    value: float

@consumer.handler
def deliver_metrics(event: Trained | Evaluated):
    for name, value in event.results.items():
        publisher.publish(Metric(
            name=name,
            value=value.item(),
        ), 'metrics')
```

Next, we can create a subscriber to handle the metrics:

```python
from logging import getLogger 
from torchsystem.services import Subscriber
from src.checkpoints import Metric, provider

subscriber = Subscriber(provider=provider)
logger = getLogger(__name__)

@subscriber.subscribe('metrics')
def log_metrics(metric: Metric):  
    logger.info(f"Average {metric.name}: {metric.value}")  
```

Here, we are using Python's logger for simplicity, but you can plug in any logging or monitoring solution you want using dependency injection. 


### Compiler

The Classifier aggregate we just created can be built and compiled in a simple way. However, you will find yourself in situations where you need to pick a torch backend, create and clean multiprocessing resources, move modules to devices, pickle modules, etc., and you will need a tool to build the aggregate and compile it. 

This library provides a simple implementation of the builder pattern under the name of compiler, the idea is to encapsulate all the construction and compilation process of your aggregate as a simple pipeline. Let's create one:

```python
import os
from torch import load 
from torch.nn import Module
from torch.optim import Optimizer
from torchsystem import Depends
from torchsystem.compiler import Compiler, compile
from src.classifier import Classifier
from src.training import device, provider 

compiler = Compiler[Classifier](provider=provider)

@compiler.step
def build_model(nn: Module, criterion: Module, optimizer: Optimizer, device: str = Depends(device)):
    if device != 'cpu':
        print(f"Moving classifier to device {device}...") 
        return Classifier(nn, criterion, optimizer).to(device)
    else:
        return Classifier(nn, criterion, optimizer)

@compiler.step
def restore_weights(classifier: Classifier, device: str = Depends(device)): 
    path = f"data/weights/{classifier.name}-{classifier.hash}.pth"
    if os.path.exists(path):
        (f"Restoring model weights from: {path}")   
        checkpoint = load(path, map_location=device) 
        classifier.epoch = checkpoint['epoch']
        classifier.nn.load_state_dict(checkpoint['nn'])
        classifier.optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print(f"No weights found at {path}, skipping restore")
    return classifier

@compiler.step
def compile_model(classifier: Classifier, device: str = Depends(device)):
    if device != 'cpu':
        print("Compiling model...") 
        return compile(classifier) 
    else:
        return classifier

@compiler.step
def debug_model(classifier: Classifier):
    print(
        f"Compiled model with:\n"
        f"Name:  {classifier.name}\n"
        f"Hash:  {classifier.hash}\n"
        f"Epochs: {classifier.epoch}"
    )
    print(classifier)
    return classifier
```

Each time the model is "compiled," this pipeline will:

- Move it to the appropriate device.

- Restore its weights if a checkpoint exists and bring it to the current epoch.

- Optionally compile it using PyTorch’s torch.compile.

- Debug-print the model summary.

You can run this pipeline easily:

```python
classifier = compiler.compile(nn, criterion, optimizer)
```

### Training script

Finally we put all togheter in a main file to train a simple model:

```python
from torch import cuda

def device() -> str:
    return 'cuda' if cuda.is_available() else 'cpu'

if __name__ == '__main__':
    from src import training, checkpoints, logs
    from src.compilation import compiler

    from model import MLP
    from dataset import Digits
 
    from torch.nn import CrossEntropyLoss
    from torch.optim import SGD
    from torch.utils.data import DataLoader
    from torchsystem import registry

    registry.register(MLP)
    training.provider.override(training.device, device) 
    training.producer.register(checkpoints.consumer)
    checkpoints.publisher.register(logs.subscriber)

    nn = MLP(input_size=784, hidden_size=256, output_size=10, dropout=0.5)
    criterion = CrossEntropyLoss()
    optimizer = SGD(nn.parameters(), lr=0.001)
    classifier = compiler.compile(nn, criterion, optimizer)

    datasets = {
        'train': Digits(train=True, normalize=True),
        'evaluation': Digits(train=False,  normalize=True),
    }

    loaders = {
        'train': DataLoader(datasets['train'], batch_size=256, shuffle=True, pin_memory=True, pin_memory_device='cuda', num_workers=4),
        'evaluation': DataLoader(datasets['evaluation'], batch_size=256, shuffle=False, pin_memory=True, pin_memory_device='cuda', num_workers=4) 
    } if cuda.is_available() else {
        'train': DataLoader(datasets['train'], batch_size=256, shuffle=True),
        'evaluation': DataLoader(datasets['evaluation'], batch_size=256, shuffle=False) 
    }

    for epoch in range(5):
        training.train(classifier, loaders['train'])
        training.evaluate(classifier, loaders['evaluation'])
```

Each time you run this script:

- The compiler will attempt to load existing weights and continue training from the last saved epoch.

- After evaluation, the consumer will automatically store checkpoints.

The hash of the model ensures that different configurations (e.g., changing hidden layer size) generate separate checkpoints, preventing overwriting previous models. You can find the full example [here](https://github.com/eric-m-cardozo/TorchSystem/tree/main/examples/mnist-mlp). Make sure you have torch, torchvision, and torchsystem installed to run it. Simply execute python main.py, and it will download the dataset and start training.

## Features

Here is a more detailed list of features with links to their documentation.

- [**Aggregates**](https://eric-m-cardozo.github.io/TorchSystem/domain/): Define the structure of your domain by grouping related entities and enforcing consistency within their boundaries. They encapsulate both data and behavior, ensuring that all modifications occur through controlled operations.

- [**Domain Events**](https://eric-m-cardozo.github.io/TorchSystem/domain/): Aggregates can produce and consume domain events, which signal meaningful changes in the system or trigger actions elsewhere. Exceptions are supported to be treated as domain events, allowing them to be enqueued and handled or raised as needed. This makes it trivial to implement features like early stopping (Just enqueue an exception and raise it when needed).

- [**Registry**](https://eric-m-cardozo.github.io/TorchSystem/registry/): The registry module allows you to treat your models as entities by providing a way to calculate locally unique hashes for them that can act as their identifier. This module also provides several other utilities to help you handle the data from your domain.

- [**Dependency Injection**](https://eric-m-cardozo.github.io/TorchSystem/depends/): The framework provides a robust dependency injection system that allows you to define and inject dependencies. This enables you to define your logic in terms of interfaces and inject implementations later. 

- [**Compilers**](https://eric-m-cardozo.github.io/TorchSystem/compiler/): Building aggregates can be a complex process. In the context of deep learning, aggregates not only need to be built but also compiled, making compilation an integral part of the construction process. This framework provides a Compiler class to help define and manage the compilation process for your aggregates

- [**Services**](https://eric-m-cardozo.github.io/TorchSystem/services/): Define stateless operations that fulfill domain-specific tasks using ubiquitous language. 

- [**Producers/Consumers**](https://eric-m-cardozo.github.io/TorchSystem/prodcon/): Events produced by services can be delivered by producers to several consumers. This allows you to decouple services and define complex interactions between them. 

- [**Publisher/Subscriber**](https://eric-m-cardozo.github.io/TorchSystem/pubsub/): Data also can be delivered with the publisher/subscriber pattern. Publishers can send data to subscribers using a topic-based system.

## License

This project is licensed under the Apache License 2.0. You can view a copy of the license at the following link:

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

### Summary of the Apache License 2.0

The Apache License 2.0 allows for the use, modification, and distribution of the software under the conditions specified in the full license. Some key conditions include:

- You must include a copy of the copyright notice and the license in any distribution of the software.
- You may not use the project's names or trademarks without explicit permission.
- The software is provided "as is", without warranties of any kind.

For full details, please review the complete license text [here](http://www.apache.org/licenses/LICENSE-2.0).