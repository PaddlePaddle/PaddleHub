=======================
LocalModuleManager
=======================

.. code-block:: python

    class paddlehub.module.manager.LocalModuleManager(home: str = MODULE_HOME):

-----------------

   LocalModuleManager is used to manage PaddleHub's local Module, which supports the installation, uninstallation, and search of HubModule. LocalModuleManager is a singleton object related to the path, in other words, when the LocalModuleManager object of the same home directory is generated multiple times, the same object is returned.

-----------------

* Args:
    * home(str)
       The directory where PaddleHub modules are stored, the default is ~/.paddlehub/modules

**member functions**
=====================

install
------------------

   .. code-block:: python

      def install(
         name: str = None,
         directory: str = None,
         archive: str = None,
         url: str = None,
         version: str = None,
         ignore_env_mismatch: bool = False) -> HubModule:

   Install a HubModule from name or directory or archive file or url. When installing with the name parameter, if a module that meets the conditions (both name and version) already installed, the installation step will be skipped. When installing with other parameter, The locally installed modules will be uninstalled.

   * Args:
      * name(str | optional)
         module name to install

      * directory(str | optional)
         directory containing  module code

      * archive(str | optional) 
         archive file containing  module code

      * url(str|optional) 
         url points to a archive file containing module code

      * version(str | optional)
         module version, use with name parameter
            
      * ignore_env_mismatch(str | optional)
         Whether to ignore the environment mismatch when installing the Module.

uninstall
------------------

   .. code-block:: python

      def uninstall(name: str) -> bool:

   Uninstall a HubModule from name.

   * Args:
      * name(str)
         module name to uninstall

   * Return:
      True if uninstall successfully else False

list
------------------

   .. code-block:: python

      def list() -> List[HubModule]:

   List all installed HubModule.

   * Return:
      List of installed HubModule.

search
------------------

   .. code-block:: python

      def search(name: str) -> HubModule:

   search a HubModule with specified name.


   * Args:
      * name(str)
         module name to search.

   * Return:
      None if not HubModule with specified name found else the specified HubModule.