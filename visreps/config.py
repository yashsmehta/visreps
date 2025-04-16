class ConfigDict(dict):
    """A dictionary-like class for configuration settings that allows attribute-style access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def copy(self):
        """Return a shallow copy of the ConfigDict."""
        # Ensure the copy is also a ConfigDict instance
        return ConfigDict(super().copy()) 