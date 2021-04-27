=================
Module Decorator
=================

moduleinfo
============

.. code-block:: python

    def paddlehub.module.module.moduleinfo(
        name: str,
        version: str,
        author: str = None,
        author_email: str = None,
        summary: str = None,
        type: str = None,
        meta=None) -> Callable:

-----------------

   Mark Module information for a python class, and the class will automatically be extended to inherit HubModule. In other words, python classes marked with moduleinfo can be loaded through hub.Module.

-----------------

* Args:
    * name(str)
        Module name.
    
    * version(str)
        Module name.

    * author(str)
        The author of Module.

    * author_email(str)
        The email of Module author.

    * summary(str)
        Module summary.

    * type(str)
        Module type.


moduleinfo
============

.. code-block:: python

    def paddlehub.module.module.runnable(func: Callable) -> Callable:

-----------------

   Mark a Module method as runnable, when the command `hub run` is used, the method will be called.

-----------------

* Args:
    * func(Callable)
        member function of Module.

serving
============

.. code-block:: python

    def paddlehub.module.module.serving(func: Callable) -> Callable:

-----------------

   Mark a Module method as serving method, when the command `hub serving` is used, the method will be called.

-----------------

* Args:
    * func(Callable)
        member function of Module.