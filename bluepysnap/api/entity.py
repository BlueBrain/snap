from kgforge.core import Resource


class ResolvingResource:
    def __init__(self, resource, retriever=None):
        self._wrapped = resource
        self._retriever = retriever

    def __getattr__(self, name):
        """Get an attribute from the metadata or from the wrapped resource.

        It could call the retriever if the resource is not synchronized.
        """

        # ignore private attributes to avoid unnecessary lookups (for example using ipython)
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

        if isinstance(result, Resource):
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


class Entity:
    def __init__(self, resource, instance):
        self._resource = resource
        self._instance = instance

    @property
    def resource(self):
        return self._resource

    @property
    def instance(self):
        return self._instance

    def __repr__(self):
        resource_id = getattr(self._resource, "id", None)
        resource_type = getattr(self._resource, "type", None)
        instance_type = type(self._instance).__name__
        return (
            f"Entity(resource_id=<{resource_id}>,"
            f" resource_type=<{resource_type}>,"
            f" instance_type=<{instance_type}>)"
        )

    def __getattr__(self, name):
        try:
            return getattr(self._resource, name)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
