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

"""Some wrappers to simplify using external packages."""
# TODO: the emodel wrappers are very tailored to the mechanims we're are storing in the notebook
# example. We need more of the logic from bluepyemodel/model/model.py, but for that we need a
# config, which gets added then deprecated in Nexus, according to Tanguy. Once it's not PoC, then we
# can integrate it.

import logging
import os
import subprocess
from pathlib import Path

from more_itertools import always_iterable

from bluepysnap.nexus.entity import DOWNLOADED_CONTENT_PATH

L = logging.getLogger(__name__)

try:
    from bluepyopt import ephys
except ImportError:
    L.warning("Need to have bluepyopt installed")

try:
    from bluepyemodel.model.model import (
        define_distributions,
        define_mechanisms,
        define_morphology,
        define_parameters,
    )
except ImportError:
    L.warning("Need to have bluepyemodel installed")


def _get_path(p):
    return Path(p.replace("file://", ""))


def _get_path_for_item(item, entity):
    if hasattr(item, "atLocation") and hasattr(item.atLocation, "location"):
        path = _get_path(item.atLocation.location)
        if os.access(path, os.R_OK):
            return path
    if hasattr(item, "contentUrl"):
        entity.download(items=item, path=DOWNLOADED_CONTENT_PATH)
        path = DOWNLOADED_CONTENT_PATH / item.name
        return path

    return None


def open_circuit_snap(entity):
    """Open SNAP circuit.

    Args:
        entity (Entity): Entity to open.

    Returns:
        Circuit: A SNAP Circuit instance.
    """
    import bluepysnap

    if hasattr(entity, "circuitConfigPath"):
        config_path = _get_path(entity.circuitConfigPath.url)
    else:
        # TODO: we should abstain from any hard coded paths (even if partial).
        # This was used for a demo (can also be seen elsewhere). The config path should be a
        # property of the resource.
        config_path = _get_path(entity.circuitBase.url) / "sonata/circuit_config.json"
    return bluepysnap.Circuit(str(config_path))


def open_circuit_bluepy(entity):  # pragma: no cover
    """Open bluepy circuit.

    Args:
        entity (Entity): Entity to open.

    Returns:
        bluepy.Circuit: A bluepy circuit instance.
    """
    import bluepy  # pylint: disable=import-error

    config_path = _get_path(entity.circuitBase.url) / "CircuitConfig"
    return bluepy.Circuit(str(config_path))


def open_simulation_snap(entity):
    """Open SNAP simulation.

    Args:
        entity (Entity): Entity to open.

    Returns:
        Simulation: A SNAP simulation instance.
    """
    import bluepysnap

    # TODO: Same as with open_circuit_snap: should abstain from hard coded paths
    config_path = _get_path(entity.path) / "sonata/simulation_config.json"
    return bluepysnap.Simulation(str(config_path))


def open_simulation_bluepy(entity):  # pragma: no cover
    """Open bluepy simulation.

    Args:
        entity (Entity): Entity to open.

    Returns:
        bluepy.Simulation: A bluepy simulation instance.
    """
    import bluepy  # pylint: disable=import-error

    config_path = _get_path(entity.path) / "BlueConfig"
    return bluepy.Simulation(str(config_path))


def open_simulation_bglibpy(entity):  # pragma: no cover
    """Open bluepy simulation with bglibpy.

    Args:
        entity (Entity): Entity to open.

    Returns:
        bglibpy.SSim: A bglibpy SSim instance.
    """
    from bglibpy import SSim  # pylint: disable=import-error

    config_path = _get_path(entity.path) / "BlueConfig"
    return SSim(str(config_path))


def open_morphology_release(entity):
    """Open morphology release with morph-tool.

    Args:
        entity (Entity): Entity to open.

    Returns:
        morph_tool.morphdb.MorphDB: A morphology release as a MorpDB instance.
    """
    from morph_tool.morphdb import MorphDB

    config_path = _get_path(entity.morphologyIndex.distribution.url)
    return MorphDB.from_neurondb(str(config_path))


def open_emodelconfiguration(entity, connector):  # pragma: no cover
    """Open emodel configuration.

    Args:
        entity (Entity): Entity to open.
        connector (NexusConnector): Nexus connector instance.

    Returns:
        EModelConfiguration: EModel configuration wrapper.
    """
    # TODO: we need the connector here, since the
    # morphology/SubCellularModelScript (mod file) only exists as text;
    # it's not 'connected'/'linked' to anything in nexus

    def _get_entity_by_filter(type_, filter_):
        resources = connector.get_resources(type_, filter_)
        assert len(resources) == 1, f"Wanted 1 entity, got {len(resources)}"
        ret = resources[0]

        def download(path):
            connector.download_resource(ret.distribution, path)
            return Path(path) / ret.distribution.name

        ret.download = download

        return ret

    def _get_named_entity(type_, name):
        return _get_entity_by_filter(type_, {"name": name})

    def _get_distribution(type_, name):
        return _get_entity_by_filter(type_, {"channelDistribution": name})

    def _get_distribution_for_parameter(param):
        if param.location.startswith("distribution_"):
            return _get_distribution(
                "ElectrophysiologyFeatureOptimisationChannelDistribution",
                param.location.split("distribution_")[1],
            )

        return None

    morphology = _get_named_entity("NeuronMorphology", name=entity.morphology.name)
    mod_file = _get_named_entity("SubCellularModelScript", name=entity.mechanisms.name)
    distributions = {p.name: _get_distribution_for_parameter(p) for p in entity.parameters}

    return EModelConfiguration(
        entity.parameters, entity.mechanisms, distributions, morphology, mod_file
    )


def open_morphology_neurom(entity):
    """Open morphology with NeuroM.

    Args:
        entity (Entity): Entity to open.

    Returns:
        neurom.core.morphology.Morphology: A neurom Morphology instance.
    """
    import neurom

    supported_formats = {"text/plain", "application/swc", "application/h5"}
    unsupported_formats = set()

    for item in always_iterable(entity.distribution):
        if item.type == "DataDownload":
            encoding_format = getattr(item, "encodingFormat", "").lower()
            if encoding_format in supported_formats:
                path = _get_path_for_item(item, entity)
                if path:
                    return neurom.io.utils.load_morphology(path)
            if encoding_format:
                unsupported_formats.add(encoding_format)

    if unsupported_formats:
        raise RuntimeError(f"Unsupported morphology formats: {unsupported_formats}")

    raise RuntimeError("Missing morphology location")


def open_atlas_voxcell(entity):  # pragma: no cover
    """Open atlas with voxcell.

    Args:
        entity (Entity): Entity to open.

    Returns:
        voxcell.nexus.voxcelbrain.Atlas: A voxcell Atlas instance.
    """
    from voxcell.nexus.voxelbrain import Atlas  # pylint: disable=import-error

    path = _get_path(entity.distribution.url)
    return Atlas.open(str(path))


def wrap_morphology_dataframe_as_entities(df, helper, tool=None):
    """Wraps morphology dataframe as entities.

    Args:
        df (pandas.DataFrame): A morphology dataframe.
        helper (NexusHelper): NexusHelper instance.
        tool (callable): The function to use to open the morphologies.

    Returns:
        tuple: An array of entities (:py:class:`~bluepysnap.nexus.entity.Entity`).
    """
    from kgforge.core import Resource

    from bluepysnap.nexus.entity import Entity

    df = df[["name", "path"]]
    return tuple(
        helper.reopen(
            Entity(
                Resource(
                    type="NeuronMorphology",
                    distribution=[
                        Resource(
                            type="DataDownload",
                            name=name,
                            atLocation=Resource(location=str(path)),
                            encodingFormat=f"application/{path.suffix.lstrip('.')}",
                        )
                    ],
                )
            ),
            tool=tool,
        )
        for name, path in df.values
    )


class DistrWrapper:  # pragma: no cover
    """Wrapper for distributions."""

    def __init__(self, parameter, distribution):
        """Instantiate a new DistrWrapper.

        Args:
            parameter (kgforge.core.Resource): A parameter resource.
            distribution (kgforge.core.Resource): A distribution resource.
        """
        self.parameters = parameter if isinstance(parameter, list) else [parameter]

        if distribution is None:
            self.name = "uniform"
            self.function = None
            self.soma_ref_location = None
        else:
            self.name = distribution.channelDistribution
            self.function = distribution.function
            self.soma_ref_location = distribution.somaReferenceLocation


class ParamWrapper:  # pragma: no cover
    """Wrapper for parameters."""

    def __init__(self, parameter):
        """Instantiate a new ParamWrapper.

        Args:
            parameter (kgforge.core.Resource): A parameter resource.
        """
        self.param = parameter

    def __getattr__(self, name):
        """Wrap the attribute getter for the parameters."""
        if name == "distribution" and not hasattr(self.param, "distribution"):
            # check which distribution should be returned but default to 'uniform'
            if self.param.location.startswith("distribution_"):
                return self.param.location.split("distribution_")[1]
            return "uniform"
        return getattr(self.param, name)


class EmodelMorphWrapper:  # pragma: no cover
    """Wrapper for emodels."""

    MORPHOLOGY_PATH = DOWNLOADED_CONTENT_PATH / "morphologies"

    class DummyMorph:
        """Dummy Morphology class."""

        def __init__(self, path):
            """Instantiate a new DummyMorph.

            Args:
                path (str): The path to the morphology file.
            """
            self.path = path

    def __init__(self, morphology):
        """Instantiate a new EmodelMorphWrapper.

        Args:
            morphology (Entity): A morphology entity.
        """
        self.path = morphology.download(self.MORPHOLOGY_PATH)
        self.morphology = self.DummyMorph(str(self.path))


class EModelConfiguration:  # pragma: no cover
    """EModelConfiguration wrapper class."""

    MECHANISM_PATH = DOWNLOADED_CONTENT_PATH / "mechanisms_source"

    def __init__(self, parameters, mechanisms, distributions, morphology, mod_file):
        """Instantiate a new EModelConfiguration.

        Args:
            parameters (list): An array of parameters.
            mechanisms (list): An array of mechanisms.
            distributions (list): An array of distributions.
            morphology (Entity): A morphology entity.
            mod_file (object): Entity-like downloadable resource
        """
        self._parameters = [ParamWrapper(p) for p in parameters]
        self._mechanisms = mechanisms if isinstance(mechanisms, list) else [mechanisms]
        self._distributions = [DistrWrapper(p, d) for p, d in distributions.items()]
        self._morphology = EmodelMorphWrapper(morphology)
        self._mod_file = mod_file

    def _compile_mod_file(self):
        """Get the mod file, and compile it.

        Returns:
            str: path to the downloaded mod file
        """
        path = self._mod_file.download(self.MECHANISM_PATH)
        subprocess.check_call(
            ["nrnivmodl", str(self.MECHANISM_PATH)],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        return path

    def build_cell_model(self):
        """Build the cell model.

        Returns:
            bluepyopt.ephys.models.CellModel: Cell model
        """
        self._compile_mod_file()
        morphology = define_morphology(self._morphology)
        mechanisms = define_mechanisms(self._mechanisms, None)
        distributions = define_distributions(self._distributions)
        parameters = define_parameters(self._parameters, distributions, None)

        cell_model = ephys.models.CellModel(
            name=self._morphology.path.stem,
            morph=morphology,
            mechs=mechanisms,
            params=parameters,
        )

        return cell_model
