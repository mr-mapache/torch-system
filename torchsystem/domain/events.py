# Copyright 2025 Eric M. Cardozo (Eric Hermosis).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#s
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

from typing import Optional
from typing import overload
from typing import Sequence
from typing import Iterable
from inspect import signature
from collections import deque
from collections.abc import Callable

class Event:
    """
    A DOMAIN EVENT is a representation of something that has happened in the DOMAIN.

    This class is a base class for creating custom DOMAIN EVENTS. It is a simple class that can be
    optionally subclassed to write self-documented code when creating custom events.
    """

type EVENT = Event | type[Event] | Exception | type[Exception]
type HANDLERS = Callable | Sequence[Callable]

class Events:
    """
    A collection of DOMAIN EVENTS that have occurred within a Bounded context. The EVENTS
    class is responsible for managing the events that have occurred within the Bounded context
    and dispatching them to the appropriate handlers.

    When an event is enqueued, it is added to the queue of events to be processed. When the `commit`
    method is called, the events are dequeued and dispatched to the appropriate handlers. If no handler
    is found for an event, the event is ignored, except if the event is an exception.

    Exceptions are treated as domain events but they are raised when the `commit` method is called by
    default if no handler is found for it's type.

    Attributes:
        queue (deque[Event]): A queue of DOMAIN EVENTS that have occurred within the Bounded context.
        handlers (dict[type[Event], Sequence[Callable]]): A dictionary of handlers that are responsible for handling
            DOMAIN EVENTS. The key is the type of the event and the value is the handler function.

    Example:
        ```python
        from torchsystem.domain import Events, Event

        class ClsEvent(Event):...

        class ObjEvent(Event):
            def __init__(self, value):
                self.value = value

        class OtherObjEvent(Event):
            def __init__(self, willbeignored):
                self.value = willbeignored

        events = Events()
        events.enqueue(ClsEvent)
        events.enqueue(KeyError) # Enqueues a KeyError exception event
        events.enqueue(ObjEvent('somevalue'))
        events.enqueue(OtherObjEvent('willbeignored'))
        events.enqueue(StopIteration) # Enqueues a StopIteration exception event

        events.handlers[ClsEvent] = lambda: print('ClsEvent was handled.')
        events.handlers[KeyError] = lambda: print('KeyError was handled.')
        events.handlers[ObjEvent] = lambda event: print(f'ObjEvent was handled with value: {event.value}')
        events.handlers[OtherObjEvent] = lambda: print('OtherObjEvent was handled.')

        try:
            events.commit()
        except StopIteration:
            print('StopIteration exception was raised.')

        # Output:
        #ClsEvent was handled.
        #KeyError was handled.
        #ObjEvent was handled with value: somevalue
        #OtherObjEvent was handled.
        #StopIteration exception was raised. Usefull for early stopping in training loops.
        ```
    """
    def __init__(self):
        self.queue = deque[Event | Exception | type[Event] | type[Exception]]()
        self.handlers = dict[type, Callable | Sequence[Callable]]()

    @overload
    def enqueue(self, event: Event) -> None: ...

    @overload
    def enqueue(self, event: type[Event]) -> None: ...

    @overload
    def enqueue(self, event: Exception) -> None: ...

    @overload
    def enqueue(self, event: type[Exception]) -> None: ...

    def enqueue(self, event: EVENT) -> None:
        """
        Enqueue a DOMAIN EVENT into the EVENTS queue to be processed when the `commit`
        method is called. Exceptions can also be enqueued as domain events.

        Args:
            event (Event): The DOMAIN EVENT or exception to be enqueued.
        """
        self.queue.append(event)

    def dequeue(self) -> Optional[EVENT]:
        """
        Dequeue a DOMAIN EVENT from the EVENTS queue to be processed by the `commit` method.

        Returns:
            Optional[Event]: The DOMAIN EVENT or exception to be processed.
        """
        return self.queue.popleft() if self.queue else None

    @overload
    def handle(self, event: Event) -> None: ...

    @overload
    def handle(self, event: type[Event]) -> None: ...

    @overload
    def handle(self, event: type[Exception]) -> None: ...

    @overload
    def handle(self, event: Exception) -> None: ...

    def handle(self, event: Event | Exception | type[Event] | type[Exception]) -> None:
        """
        Handles a DOMAIN EVENT by dispatching it to the appropriate handler or group of handlers. If no handler 
        is found for the event, the event is ignored, except if the event is an exception. If the event is an
        exception, it is raised by default if no handler is found for it's type.

        Both classes and instances of DOMAIN EVENTS are supported. The method also will look at the
        signature of the handler to determine if the event should be passed as an argument to the handler
        or if the handler should be called without arguments.
        
        Args:
            event (Event): The DOMAIN EVENT or exception to be handled.

        Raises:
            event: If no handler is found for the event and the event is an exception.
        """
        handlers = self.handlers.get(event) if isinstance(event, type) else self.handlers.get(type(event)) 
        if handlers:
            for handler in handlers if isinstance(handlers, Iterable) else [handlers]:
                handler() if len(signature(handler).parameters) == 0 else handler(event)
        
        elif isinstance(event, Exception) or isinstance(event, type) and issubclass(event, Exception):
            raise event

    def commit(self) -> None:
        while event := self.dequeue():
            self.handle(event)
