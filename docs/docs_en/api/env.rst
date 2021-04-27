================
Hub Environment
================

.. code-block:: console

    HUB_HOME
    ├── MODULE_HOME
    ├── CACHE_HOME
    ├── DATA_HOME
    ├── CONF_HOME
    ├── THIRD_PARTY_HOME
    ├── TMP_HOME
    └── LOG_HOME


paddlehub.env.HUB_HOME
=========================

    The root directory for storing PaddleHub related data. Default to ~/.paddlehub. Users can change the default value through the HUB_HOME environment variable.

paddlehub.env.MODULE_HOME
=========================

    Directory for storing the installed PaddleHub Module.

paddlehub.env.CACHE_HOME
=========================

    Directory for storing the cached data.

paddlehub.env.DATA_HOME
=========================

    Directory for storing the automatically downloaded datasets.

paddlehub.env.CONF_HOME
=========================

    Directory for storing the default configuration files.

paddlehub.env.THIRD_PARTY_HOME
================================

    Directory for storing third-party libraries.

paddlehub.env.TMP_HOME
=========================

    Directory for storing the temporary files generated during running, such as intermediate products of installing modules, files in this directory will generally be automatically cleared.

paddlehub.env.LOG_HOME
=========================

    Directory for storing the log files generated during operation, including some non-fatal errors. The log will be stored daily.
