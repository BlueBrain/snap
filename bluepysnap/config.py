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

from pathlib import Path
from collections.abc import Mapping, Iterable

from bluepysnap import utils
from bluepysnap.exceptions import BluepySnapError

# List of keys which are expected to have paths
EXPECTED_PATH_KEYS = {'morphologies_dir',
                      'biophysical_neuron_models_dir',
                      'vasculature_file',
                      'vasculature_mesh',
                      'end_feet_area',
                      'neurolucida-asc',
                      'h5v1',
                      'edges_file',
                      'nodes_file',
                      'edges_type_file',
                      'nodes_type_file',
                      'node_sets_file'}


class Config:
    """SONATA network config parser.

    See Also:
        https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#network_config
    """

    def __init__(self, config):
        """Initializes a Config object from a path to the actual config.

        Args:
            config (str/dict): Path to the SONATA configuration file or dict containing the config.

        Returns:
             Config: A Config object.
        """
        if isinstance(config, dict):
            content = config.copy()
            configdir = None
        else:
            configdir = str(Path(config).parent.resolve())
            content = utils.load_json(str(config))
        self.manifest = Config._resolve_manifest(content.pop('manifest', {}), configdir)
        self.content = content

    @staticmethod
    def _resolve_manifest(manifest, configdir):
        result = manifest.copy()

        for k, v in result.items():
            if not isinstance(v, str):
                raise BluepySnapError(f'{v} should be a string value.')
            if not Path(v).is_absolute() and not v.startswith("$"):
                if configdir is None:
                    raise BluepySnapError("Dictionary config with relative paths is not allowed.")
                result[k] = str(Path(configdir, v).resolve())

        while True:
            update = False
            for k, v in result.items():
                if v.count('$') > 1:
                    raise BluepySnapError(
                        f'{k} is not a valid anchor : contains more than one sub anchor.')
                if v.startswith('$'):
                    tokens = v.split('/', 1)
                    resolved = result[tokens[0]]
                    if '$' not in resolved:
                        result[k] = str(Path(resolved, *tokens[1:]))
                        update = True
            if not update:
                break

        assert '${configdir}' not in result
        result['${configdir}'] = configdir

        return result

    def _resolve_string(self, value, key):
        # not a startswith to detect the badly placed anchors
        if '$' in value:
            vs = [
                self.manifest[v] if v.startswith('$') else v
                for v in value.split('/')
            ]
            abs_paths = [v for v in vs[1:] if v.startswith('/')]
            if len(abs_paths) != 0:
                raise BluepySnapError(
                    f"Misplaced anchors in : {value}. Please verify your '$' usage.")
            return str(Path(*vs))
        # only way to know if value is a relative path or a normal string
        elif value.startswith('.') or key in EXPECTED_PATH_KEYS:
            if self.manifest['${configdir}'] is not None:
                return str(Path(self.manifest['${configdir}'], value).resolve())
            raise BluepySnapError("Dictionary config with relative paths is not allowed.")
        else:
            # we cannot know if a string is a path or not if it does not contain anchor or .
            return value

    def _resolve(self, value, key=None):
        if isinstance(value, Mapping):
            return {
                k: self._resolve(v, k) for k, v in value.items()
            }
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
    def parse(filepath):
        """Parse SONATA network config."""
        return Config(filepath).resolve()
