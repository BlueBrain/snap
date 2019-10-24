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
import os.path

import six

from bluepysnap import utils


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
        configdir = os.path.abspath(os.path.dirname(filepath))
        content = utils.load_json(filepath)
        self.manifest = Config._resolve_manifest(content.pop('manifest'), configdir)
        self.content = content

    @staticmethod
    def _resolve_manifest(manifest, configdir):
        result = manifest.copy()

        assert '${configdir}' not in result
        result['${configdir}'] = configdir

        for k, v in six.iteritems(result):
            if v == '.':
                result[k] = configdir
            elif v.startswith('./'):
                result[k] = os.path.join(configdir, v[2:])

        while True:
            update = False
            for k, v in six.iteritems(result):
                if v.startswith('$'):
                    tokens = v.split('/', 1)
                    resolved = result[tokens[0]]
                    if '$' not in resolved:
                        result[k] = os.path.join(resolved, *tokens[1:])
                        update = True
            if not update:
                break

        return result

    def _resolve_path(self, value):
        if '$' in value:
            vs = [
                self.manifest[v] if v.startswith('$') else v
                for v in value.split('/')
            ]
            return os.path.join(*vs)
        else:
            return value

    def _resolve(self, value):
        if isinstance(value, collections.Mapping):
            return {
                k: self._resolve(v) for k, v in six.iteritems(value)
            }
        elif isinstance(value, six.string_types):
            return self._resolve_path(value)
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
