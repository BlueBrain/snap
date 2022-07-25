"""Some wrappers to simplify using external packages."""
# TODO: the emodel wrappers are very tailored to the mechanims we're are storing in the notebook
# example. We need more of the logic from bluepyemodel/model/model.py, but for that we need a
# config, which gets added then deprecated in Nexus, according to Tanguy. Once it's not PoC, then we
# can integrate it.

import logging
import subprocess

from bluepysnap.api.factory import DOWNLOADED_CONTENT_PATH

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


def wrap_morphology_dataframe_as_entities(df, api, tool=None):
    """Wraps morphology dataframe as entities.

    Args:
        df (pd.DataFrame): morphology dataframe
        api (bluepysnap.api.core.Api): API instance
        tool (callable): function used to open the morphologies

    Returns:
        (tuple): array of entities (bluepysnap.api.entity.Entity)
    """
    from kgforge.core import Resource

    from bluepysnap.api.entity import Entity

    df = df[["name", "path"]]
    return tuple(
        api.reopen(
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
        """Initializes the DistrWrapper."""
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
        """Initializes the ParamWrapper."""
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
            """Initializes the DummyMorph."""
            self.path = path

    def __init__(self, morphology):
        """Initializes the EmodelMorphWrapper."""
        self.path = morphology.download(self.MORPHOLOGY_PATH)
        self.morphology = self.DummyMorph(str(self.path))


class EModelConfiguration:
    """EModelConfiguration wrapper class."""

    MECHANISM_PATH = DOWNLOADED_CONTENT_PATH / "mechanisms_source"

    def __init__(self, parameters, mechanisms, distributions, morphology, mod_file):
        """Initializes EModelConfiguration.

        Args:
            parameters (list): array of parameters
            mechanisms (list): array of mechanisms
            distributions (list): array of distributions
            morphology (Entity): morphology entity
            mod_file (object): Entity-like downloadable resource
        """
        self._parameters = [ParamWrapper(p) for p in parameters]
        self._mechanisms = mechanisms if isinstance(mechanisms, list) else [mechanisms]
        self._distributions = [DistrWrapper(p, d) for p, d in distributions.items()]
        self._morphology = EmodelMorphWrapper(morphology)
        self._mod_file = mod_file

    def _compile_mod_file(self):
        """Get the mod file, and compile it."""
        path = self._mod_file.download(self.MECHANISM_PATH)
        subprocess.check_call(["nrnivmodl", str(self.MECHANISM_PATH)])
        return path

    def build_cell_model(self):
        """Build the cell model.

        Returns:
            (bluepyopt.ephys.models.CellModel): Cell model
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
