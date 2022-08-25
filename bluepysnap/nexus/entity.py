# Copyright (c) 2022, EPFL/Blue Brain Project

# This file is part of BlueBrain SNAP library <https://github.com/BlueBrain/snap>

# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 3.0 as published
# by the Free Software Foundation.

# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.

# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Classes that implement the resource and entity handling."""
from functools import partial
from pathlib import Path

from kgforge.core import Resource
from lazy_object_proxy import Proxy
from more_itertools import always_iterable

# user defined or tmp would be better
DOWNLOADED_CONTENT_PATH = Path(".downloaded_content")


class ResolvingResource:
    """Class implementing traversing the resources attributes."""

    def __init__(self, resource, retriever=None):
        """Instantiate a new wrapper class.

        Wraps kgforge Resource object and provides easier access (traversing) to linked objects
        and store metadata.

        Args:
            resource (kgforge.core.Resource): The wrapped resource.
            retriever (callable): A function implementing the communication with Nexus.
        """
        self._wrapped = resource
        self._retriever = retriever

    @property
    def wrapped(self):
        """The wrapped resource."""
        return self._wrapped

    def _sync_resource(self):
        """Retrieve the resource if it's not synchronized."""
        if (
            self._retriever
            and getattr(self._wrapped, "_synchronized", None) is False
            and hasattr(self._wrapped, "id")
        ):
            self._wrapped = self._retriever(self._wrapped.id)
            assert self._wrapped._synchronized is True  # pylint:disable = protected-access
            return True
        return False

    def __getattr__(self, name):
        """Get an attribute from the metadata or from the wrapped resource.

        It could call the retriever if the resource is not synchronized.
        """
        # ignore private attributes to avoid unnecessary lookups (for example using notebooks)
        if name.startswith("_"):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

        # get the attribute from _store_metadata
        _name = "_" + name
        meta = getattr(self._wrapped, "_store_metadata", None)
        if meta and _name in meta:
            return meta[_name]

        if not hasattr(self._wrapped, name):
            self._sync_resource()

        try:
            result = getattr(self._wrapped, name)
        except AttributeError as error:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            ) from error

        # In some cases 'distribution' contains list, in some cases 'DataDownload'.
        # In either case, don't wrap it as it raises an error with kgforge.
        if isinstance(result, Resource) and (
            not hasattr(result, "type") or result.type != "DataDownload"
        ):
            # Wrap the Resource to resolve nested attributes.
            # The retrieved resource is not saved to avoid modifying the wrapped resource.
            result = ResolvingResource(result, retriever=self._retriever)

        return result

    def __repr__(self):
        """Proxies the __repr__ call to the wrapped resource."""
        return self._wrapped.__repr__()


class Entity:
    """Implements the instantiation and downloading of a resource."""

    def __init__(self, resource: Resource, helper=None, connector=None, opener=None):
        """Instantiate a new entity.

        Args:
            resource (kgforge.core.Resource): The resource to be wrapped.
            helper (NexusHelper): NexusHelper instance.
            connector (NexusConnector): Connector instance.
            opener (callable): A function used to open the instance associated to the resource.
        """
        if connector is None:
            retriever = downloader = None
        else:
            retriever = connector.get_resource_by_id
            downloader = connector.download_resource

        self._helper = helper
        self._connector = connector
        self._resolving_resource = ResolvingResource(resource, retriever=retriever)
        self._instance = Proxy(partial(opener, self)) if opener else None
        self._downloader = downloader

    @property
    def resource(self):
        """The wrapped resource."""
        return self._resolving_resource.wrapped

    @property
    def instance(self):
        """The instantiated object."""
        return self._instance

    def download(self, items=None, path=None):
        """Downloads the wrapped resource.

        Args:
            items (list, kgforge.core.Resource): Item(s) to download.
            path (str): The path to the directory into which the data is downloaded.
        """
        path = path or DOWNLOADED_CONTENT_PATH
        items = always_iterable(items or self.resource.distribution)

        for item in items:
            self._downloader(item, path)

    def to_dict(self, store_metadata=True):
        """Returns the entity as a dictionary.

        Args:
            store_metadata(bool): A flag indicating whether or not to include metadata in the
                                  output.

        Returns:
            dict: A dictionary containing the data of the entity.
        """
        return self._helper.to_dict(self, store_metadata=store_metadata)

    def __repr__(self):
        """Overwrite the default __repr__ implementation."""
        resource_id = getattr(self._resolving_resource, "id", None)
        resource_type = getattr(self._resolving_resource, "type", None)
        instance_type = type(self._instance).__name__
        return (
            f"Entity(resource_id=<{resource_id}>,"
            f" resource_type=<{resource_type}>,"
            f" instance_type=<{instance_type}>)"
        )

    def __getattr__(self, name):
        """Proxy getter requests to the wrapped resource."""
        try:
            return getattr(self._resolving_resource, name)
        except AttributeError as error:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            ) from error
