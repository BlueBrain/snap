# Copyright (c) 2019, EPFL/Blue Brain Project

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

"""SONATA network config parsing."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import libsonata

from bluepysnap.exceptions import BluepySnapError

# List of keys which are expected to have paths
EXPECTED_PATH_KEYS = {
    "morphologies_dir",
    "biophysical_neuron_models_dir",
    "vasculature_file",
    "vasculature_mesh",
    "endfeet_meshes_file",
    "microdomains_file",
    "neurolucida-asc",
    "h5v1",
    "edges_file",
    "nodes_file",
    "edges_type_file",
    "nodes_type_file",
    "node_sets_file",
    "output_dir",
    "network",
    "mechanisms_dir",
}


class Parser:
    """SONATA network config parser.

    See Also:
        https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#network_config
    """

    def __init__(self, config, config_dir):
        """Initializes a Resolver object.

        Args:
            config (dict): Dict containing the config.
            config_dir(str):  Path to the directory containing the config file.

        Returns:
             Parser: A Parser object.
        """
        content = config.copy()

        self.manifest = Parser._resolve_manifest(content.pop("manifest", {}), config_dir)
        self.content = content

    @staticmethod
    def _resolve_manifest(manifest, configdir):
        result = manifest.copy()

        for k, v in result.items():
            if not isinstance(v, str):
                raise BluepySnapError(f"{v} should be a string value.")
            if not Path(v).is_absolute() and not v.startswith("$"):
                result[k] = str(Path(configdir, v).resolve())

        while True:
            update = False
            for k, v in result.items():
                if v.count("$") > 1:
                    raise BluepySnapError(
                        f"{k} is not a valid anchor : contains more than one sub anchor."
                    )
                if v.startswith("$"):
                    tokens = v.split("/", 1)
                    resolved = result[tokens[0]]
                    if "$" not in resolved:
                        result[k] = str(Path(resolved, *tokens[1:]))
                        update = True
            if not update:
                break

        assert "${configdir}" not in result
        result["${configdir}"] = configdir

        return result

    def _resolve_string(self, value, key):
        # not a startswith to detect the badly placed anchors
        if "$" in value:
            vs = [self.manifest[v] if v.startswith("$") else v for v in value.split("/")]
            abs_paths = [v for v in vs[1:] if v.startswith("/")]
            if len(abs_paths) != 0:
                raise BluepySnapError(
                    f"Misplaced anchors in : {value}. Please verify your '$' usage."
                )
            return str(Path(*vs))
        # only way to know if value is a relative path or a normal string
        elif value.startswith(".") or key in EXPECTED_PATH_KEYS:
            return str(Path(self.manifest["${configdir}"], value).resolve())
        else:
            # we cannot know if a string is a path or not if it does not contain anchor or .
            return value

    def _resolve(self, value, key=None):
        if isinstance(value, Mapping):
            return {k: self._resolve(v, k) for k, v in value.items()}
        elif isinstance(value, str):
            return self._resolve_string(value, key)
        elif isinstance(value, Iterable):
            return [self._resolve(v) for v in value]
        else:
            return value

    def resolve(self):
        """Resolve variables in config file paths."""
        return self._resolve(self.content)

    @staticmethod
    def parse(config, configdir):
        """Parse SONATA network config."""
        return Parser(config, configdir).resolve()


class Config:
    """Common config class."""

    def __init__(self, config, config_class):
        """Initializes the Config class.

        Args:
            config (str): Path to the configuration file
            config_class (class): libsonata class corresponding to the configuration file, either
                libsonata.CircuitConfig or libsonata.SimulationConfig
        """
        self._config_dir = str(Path(config).parent.absolute())
        self._libsonata = config_class.from_file(config)

    @property
    def to_libsonata(self):
        """Return the libsonata instance of the config."""
        return self._libsonata

    def to_dict(self):
        """Return the configuration as a dict with absolute paths."""
        return Parser.parse(json.loads(self._libsonata.expanded_json), self._config_dir)


class CircuitConfig(Config):
    """Handle CircuitConfig."""

    @classmethod
    def from_config(cls, config_path):
        """Instantiate the config class from circuit configuration."""
        return cls(config_path, libsonata.CircuitConfig)


class SimulationConfig(Config):
    """Handle SimulationConfig."""

    @classmethod
    def from_config(cls, config_path):
        """Instantiate the config class from simulation configuration."""
        return cls(config_path, libsonata.SimulationConfig)
