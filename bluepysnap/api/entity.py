from functools import partial
from pathlib import Path

from kgforge.core import Resource
from lazy_object_proxy import Proxy
from more_itertools import always_iterable

# user defined or tmp would be better
DOWNLOADED_CONTENT_PATH = Path(".downloaded_content").absolute()


class ResolvingResource:
    def __init__(self, resource: Resource, retriever=None):
        self._wrapped = resource
        self._retriever = retriever

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
        except AttributeError:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

        # TODO: In some cases 'distribution' contains list, in some cases 'DataDownload'
        # Figure out how to handle this in a cleaner way.
        # if isinstance(result, Resource):
        if isinstance(result, Resource) and (
            not hasattr(result, "type") or result.type != "DataDownload"
        ):
            # Wrap the Resource to resolve nested attributes.
            # The retrieved resource is not saved to avoid modifying the wrapped resource.
            result = ResolvingResource(result, retriever=self._retriever)

        return result

    def _sync_resource(self):
        """Retrieve the resource if it's not synchronized."""
        if (
            self._retriever
            and getattr(self._wrapped, "_synchronized", None) is False
            and hasattr(self._wrapped, "id")
        ):
            # FIXME: is it fine to always retrieve the latest version? if not, which one?
            self._wrapped = self._retriever(self._wrapped.id)
            assert self._wrapped._synchronized is True
            return True
        return False

    def __repr__(self):
        return self._wrapped.__repr__()


class Entity:
    def __init__(self, resource: Resource, retriever=None, opener=None, downloader=None):
        """Return a new entity.

        Args:
            resource: resource to be wrapped.
            retriever: callable used to retrieve nested resources by id.
            opener: callable used to open the instance associated to the resource.
        """
        self._rr = ResolvingResource(resource, retriever=retriever)
        self._instance = Proxy(partial(opener, self)) if opener else None
        self._downloader = downloader

    @property
    def resource(self):
        return self._rr._wrapped

    @property
    def instance(self):
        return self._instance

    def download(self, items=None, path=None):
        path = path or DOWNLOADED_CONTENT_PATH
        items = always_iterable(items or self.resource.distribution)

        for item in items:
            self._downloader(item, path)

    def __repr__(self):
        resource_id = getattr(self._rr, "id", None)
        resource_type = getattr(self._rr, "type", None)
        instance_type = type(self._instance).__name__
        return (
            f"Entity(resource_id=<{resource_id}>,"
            f" resource_type=<{resource_type}>,"
            f" instance_type=<{instance_type}>)"
        )

    def __getattr__(self, name):
        try:
            return getattr(self._rr, name)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
