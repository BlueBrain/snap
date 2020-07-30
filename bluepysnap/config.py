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

# TODO: move to `libsonata` library

import collections
from pathlib2 import Path

import six

from bluepysnap import utils
from bluepysnap.exceptions import BluepySnapError


class Config(object):
    """SONATA network config parser.

    See Also:
        https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#network_config
    """

    def __init__(self, filepath):
        """Initializes a Config object from a path to the actual config.

        Args:
            filepath (str): Path the SONATA configuration file.

        Returns:
             Config: A Config object.
        """
        configdir = str(Path(filepath).parent.resolve())
        content = utils.load_json(filepath)
        self.manifest = Config._resolve_manifest(content.pop('manifest', {}), configdir)
        self.content = content

    @staticmethod
    def _resolve_manifest(manifest, configdir):
        result = manifest.copy()

        assert '${configdir}' not in result
        result['${configdir}'] = configdir

        for k, v in six.iteritems(result):
            if v.startswith('.'):
                result[k] = str(Path(configdir, v).resolve())

        while True:
            update = False
            for k, v in six.iteritems(result):
                if v.count('$') > 1:
                    raise BluepySnapError(
                        '{} is not a valid anchor : contains more than one sub anchor.')
                if v.startswith('$'):
                    tokens = v.split('/', 1)
                    resolved = result[tokens[0]]
                    if '$' not in resolved:
                        result[k] = str(Path(resolved, *tokens[1:]))
                        update = True
            if not update:
                break

        for k, v in result.items():
            if not v.startswith('/'):
                raise BluepySnapError("{} cannot be resolved as an abs path.".format(k))

        return result

    def _resolve_string(self, value):
        # not a startswith to detect the badly placed anchors
        if '$' in value:
            vs = [
                self.manifest[v] if v.startswith('$') else v
                for v in value.split('/')
            ]
            abs_paths = [v for v in vs[1:] if v.startswith('/')]
            if len(abs_paths) != 0:
                raise BluepySnapError("Misplaced anchors in : {}."
                                      "Please verify your '$' usage.".format(value, vs))
            return str(Path(*vs))
        elif value.startswith('.'):
            return str(Path(self.manifest['${configdir}'], value).resolve())
        else:
            # we cannot know if a string is a path or not if it does not contain anchor or .
            return value

    def _resolve(self, value):
        if isinstance(value, collections.Mapping):
            return {
                k: self._resolve(v) for k, v in six.iteritems(value)
            }
        elif isinstance(value, six.string_types):
            return self._resolve_string(value)
        elif isinstance(value, collections.Iterable):
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
