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
import subprocess

from bluepysnap.nexus.factory import DOWNLOADED_CONTENT_PATH

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


class DistrWrapper:
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


class ParamWrapper:
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


class EmodelMorphWrapper:
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


class EModelConfiguration:
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
