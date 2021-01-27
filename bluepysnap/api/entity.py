class Entity:
    def __init__(self, resource, instance):
        self.resource = resource
        self.instance = instance

    def __getattr__(self, name):
        """Try to get an attribute from resource._store_metadata or from the resource itself."""
        _name = "_" + name
        meta = getattr(self.resource, "_store_metadata", None)
        if meta and _name in meta:
            return meta[_name]
        if hasattr(self.resource, name):
            return getattr(self.resource, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
