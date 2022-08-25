Nexus Forge Integration
=======================

The Nexus Forge integration consists of the following submodules/classes:

* ``bluepysnap.nexus.core.NexusHelper``
   * The class that user would instantiate to begin getting data from Nexus
* ``bluepysnap.nexus.connector.NexusConnector``
   * contains core functionalities used to communicate with Nexus
* ``bluepysnap.nexus.factory.EntityFactory``
   * factory class that creates entities and links them to tools that they are instantiated with
* ``bluepysnap.nexus.entity.Entity``
   * a single item from Nexus wrapped into a class that allows traversal of linked objects
   * can be instantiated with a tool linked by ``EntityFactory``
* ``bluepysnap.nexus.tools``
   * collection of functions used for instantiating nexus entities

For further details, please see :ref:`api-documentation`.

NexusHelper
-----------

To set up SNAP NexusHelper, the following is needed:

* **BUCKET**: Nexus project data is acquired from (as ``"ORGANIZATION/PROJECT"``)
* **TOKEN**: Nexus token copied from `Nexus <https://bbp.epfl.ch/nexus/web/>`_.

Then it's just a matter of instantiating the helper:

.. code-block:: python

    from bluepysnap.nexus import NexusHelper

    BUCKET = "nse/test"
    TOKEN = "<insert_token_here>"
    nexus = NexusHelper(BUCKET, TOKEN)

    entity = nexus.get_entity_by_id('<nexus_id>')

EntityFactory
-------------

As mentioned above, ``EntityFactory`` links tools with entities.
To be more exact, while creating entities, it assigns them to a tool that will be used to open/instantiate them.
SNAP is equipped with several tools but it is also entirely possible to add new tools to the factory.

Tools
~~~~~

Tools are basically functions that take only one ``Entity`` as an argument and returns an instance of a desired tool.
Let's illuminate this with a short example:

.. code-block:: python

    def open_entity_with_some_tool(entity):
        """Example of how to open entities."""
        from pathlib import Path
        from somelib import SomeClass

        # If the entity has a reference to file path, it can be used directly:
        if hasattr(entity, "attribute_containing_path"):
            file_path = entity.path_to_file
        # Otherwise, resort to downloading the resource data files:
        else:
            download_path = Path('/path/to/directory')
            entity.download(items=entity.distribution[0], path=download_path)

            # construct the file path
            file_name = entity.distribution[0].name
            file_path = download_path / file_name

        # Open the entity with the desired class
        return SomeClass(entity.name, file_path)


Registering a Tool
~~~~~~~~~~~~~~~~~~

Before any of the custom tools can be used, they need to be registered to the ``EntityFactory`` which is responsible for creating the Entities.
The factory instance used by ``NexusHelper`` is available through ``NexusHelper.factory``.

E.g., to register a tool to open neuron morphologies:

.. code-block:: python

    nexus.factory.register(
        resource_types = ['NeuronMorphology'], # list of resource types that the function can handle
        tool = 'morph_opener',                 # arbitrary name for the tool
        func = open_entity_with_some_tool,     # the function to open the entity with
    )

Entity
------

Instantiating
~~~~~~~~~~~~~

Now that the tool is registered, it can be linked to entities fetched from Nexus and used to instantiate them:

.. code-block:: python

    entities = nexus.get_entities('NeuronMorphology', tool='morph_opener', limit=1)
    morphology = entities[0].instance

Where to put the tools?
-----------------------

Currently available tools are located in ``bluepysnap.nexus.tools``.
However, these tools are not automatically registered to ``EntityFactory`` unless they are also added to the ``EntityFactory.__init__`` method.
