'''some wrappers to simplify using external packages'''
import logging
import subprocess

from pathlib import Path

from bluepysnap.api.factory import DOWNLOADED_CONTENT_PATH

L = logging.getLogger(__name__)

try:
    from bluepyopt import ephys
    from bluepyopt.ephys.morphologies import NrnFileMorphology
except:
    L.warning("Need to have bluepyopt installed")


class EModelConfiguration:
    MECHANISM_PATH = DOWNLOADED_CONTENT_PATH / 'mechanisms_source'
    MORPHOLOGY_PATH = DOWNLOADED_CONTENT_PATH / 'morphologies'

    def __init__(self, parameters, mechanisms, morphology, mod_file):
        self._parameters = parameters
        self._mechanisms = mechanisms
        self._morphology = morphology
        self._mod_file = mod_file

    def _compile_mod_file(self):
        '''get the mod file, and compile it'''
        path = self._mod_file.download(self.MECHANISM_PATH)
        subprocess.check_call(['nrnivmodl', str(self.MECHANISM_PATH)])
        return path

    def build_cell_model(self):
        morphology_path = self._morphology.download(self.MORPHOLOGY_PATH)

        nrn_morph = NrnFileMorphology(str(morphology_path), do_replace_axon=True)

        # setup mechanisms {
        mechanism_name = self._mechanisms.name
        mechanism_location = self._mechanisms.location
        mod_path = self._compile_mod_file()
        mechanisms_locations = [ephys.locations.NrnSeclistLocation(
            f"{mechanism_name}.{mechanism_location}",
            seclist_name=mechanism_location
        )]

        natg_mech_instance = ephys.mechanisms.NrnMODMechanism(
            f"{mechanism_name}.{mechanism_location}",
            suffix=mechanism_name,
            mod_path=str(mod_path),
            locations=mechanisms_locations,
            deterministic=not self._mechanisms.stochastic)
        # }

        # TODO: this is very tailored to the mechanims we're are storing in the example;
        #   need more of the logic from bluepyemodel/model/model.py, but for that we
        #   need a config, which gets added then deprcated in Nexus, according to Tanguy
        #   once it's not PoC, then we can integrate it
        decay_param, natg_param = self._parameters

        scaler = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
            name=decay_param.location,
            distribution="math.exp({distance}*{constant})*{value}", # from bluepyemodel tests
            dist_param_names=[decay_param.name],
            soma_ref_location=0
        )
        decay_param_instance = ephys.parameters.MetaParameter(
            name=f'{decay_param.name}.{decay_param.location}',
            obj=scaler,
            attr_name=decay_param.name,
            frozen=False,
            bounds=decay_param.value
        )

        natg_param_instance = ephys.parameters.NrnSectionParameter(
            f"{natg_param.name}.{natg_param.location}",
            param_name=natg_param.name,
            locations=mechanisms_locations,
            frozen=False,
            bounds=natg_param.value
        )

        cell_model = ephys.models.CellModel(
            name=morphology_path.stem,
            morph=nrn_morph,
            mechs=[natg_mech_instance],
            params=[natg_param_instance, decay_param_instance])

        return cell_model
