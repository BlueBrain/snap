"""
The purpose of this module is to test Ipython Notebook files.

IpythonTester is a class that takes as an entry an already run notebook (i.e.: all outputs
are saved in the notebook after a 'Restart & Run All' action.). It runs all cells in a new
IPython kernel and check for errors. If the code can be run then it checks cell outputs
(text and png outputs only) and compares the just processed cell outputs with the reference ones.
If a disagreement is found, then an exception is raised.
"""
from __future__ import print_function
import os
import base64
import re
from difflib import Differ
from contextlib import contextmanager
from queue import Empty

import six
import imageio
import nbformat
import numpy as np
import jupyter_client as jc
from jupyter_client import KernelManager

PNG_SAVE_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'png_diff')


class IpynbTesterError(Exception):
    """Generic IpynbTester exception """


class IpynbKernel(object):
    """ Ipython notebook container with a notebook and a jupyter kernel manager client. """

    def __init__(self, notebook, kernel_client):
        """ Constructor for IpynbClient

        Inputs:
            notebook : a NotebookNode
            kernel_client : a BlockingKernelClient
        """
        if isinstance(notebook, nbformat.notebooknode.NotebookNode):
            self.notebook = notebook
        else:
            raise IpynbTesterError('Notebook should be a NotebookNode object')
        if isinstance(kernel_client, jc.blocking.client.BlockingKernelClient):
            self.kernel_client = kernel_client
        else:
            raise IpynbTesterError('Kernel_client should be a BlockingKernelClient object')


@contextmanager
def setup_notebook_kernel(nb_path):
    """ Context manager for notebook kernels

    Loads the notebook, creates and starts a KernelManager, yields a IpynbClient and finally stops
    channels and kernel.

    Inputs :
        nb_path : path to the tested notebook

    Yields:
        A notebook kernel
    """
    with open(nb_path, 'r') as fd:
        nb = nbformat.reads(fd.read(), nbformat.current_nbformat)
        km = KernelManager()
        km.start_kernel(extra_arguments=['--pylab=inline'],
                        stderr=open(os.devnull, 'w'))
        kc = km.client()
        kc.start_channels()
        nb_kernel = IpynbKernel(nb, kc)
        try:
            yield nb_kernel
        finally:
            kc.stop_channels()
            km.shutdown_kernel()
            del km


class IpynbTester(object):
    """ Ipython notebook tester """

    def __init__(self, cell_timeout=60, query_message_timeout=0.5):
        super(IpynbTester, self).__init__()
        self.cell_timeout = cell_timeout
        self.query_message_timeout = query_message_timeout

    @staticmethod
    def _fuzzy_compare(value1, value2):
        """ Fuzzy compares two values (apply sanitize if values are strings)

        If values are strings, they are sanitized to avoid problems with random values
        like hex addresses or uids. Direct compare if values are something else than strings.

        Inputs:
            value1 : first value to compare
            value2 : second value to compare

        Returns:
            boolean value for the comparison
        """

        def _sanitize(s):
            """ Sanitizes string for comparison """
            # ignore trailing newlines
            s = s.rstrip('\r\n')
            # normalize hex addresses:
            s = re.sub(r'0x[A-Fa-f0-9]+', '0xFFFFFFFF', s)
            # normalize UUIDs:
            s = re.sub(r'[a-f0-9]{8}(\-[a-f0-9]{4}){3}\-[a-f0-9]{12}', 'U-U-I-D', s)
            # normalize <th> and <tr> (pandas changed this multiple times for df outputs)
            s = re.sub(r'<td|<th', '<t-', s)
            s = re.sub(r'</td>|</th>', '</t->', s)
            return s

        if not type(value1) is type(value2):
            return False
        if isinstance(value1, six.string_types):
            return _sanitize(value1) == _sanitize(value2)
        return value1 == value2

    @staticmethod
    def _is_diff_png(b64_ref, b64_test, save=True, output_dir=None):
        """ Compares the pixels of two PNGs using numpy.

        Computes the difference pixel-wise between two images and saves it in output_dir if needed.

        Args:
            b64_ref: base 64 png string for the reference image
            b64_test:  base 64 png string for the tested image
            save: boolean value to save the files and the differences between them
            output_dir: directory to store the reference/tested file and difference. If
            save=True and outpout_dir=None then the cwd is used tp

        Returns:
            True if images are different, False otherwise
        """

        def _png_b64_to_ndarray(b64_png):
            """ convert PNG output into a np.ndarray using imageio """
            decoded_string = base64.decodebytes(b64_png.encode('utf-8'))
            s_io = six.BytesIO(decoded_string)
            return imageio.get_reader(s_io).get_data(0)

        reference, tested = map(_png_b64_to_ndarray, (b64_ref, b64_test))
        if reference.shape != tested.shape:
            return True
        difference = np.abs(reference - tested)
        res = np.count_nonzero(difference) > 0
        if res and save:
            output_dir = '.' if output_dir is None else output_dir
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            prefix = os.path.join(output_dir, 'ipynb_tester-{}'.format(b64_ref[:4]))
            imageio.imwrite(prefix + '_reference.png', reference)
            imageio.imwrite(prefix + '_tested.png', tested)
            imageio.imwrite(prefix + '_difference.png', 255 - difference)
        return res

    def _compare_outputs(self, reference_output, tested_output, cell,
                         skip_compare=('metadata', 'traceback', 'latex', 'prompt_number')):
        """ Compares the output of two different runs of a same cell

        Notes:
            Only address text, numbers and png images, other are skipped. If a problem is found
            then an Exception is raised.

        Inputs:
            reference_output : the reference dictionary
            tested_output : the tested dictionary
            cell : the current cell from notebook

        Raises:
            if a difference has been found
        """
        cell_id = cell['execution_count']
        for key in reference_output:
            reference = reference_output[key]
            tested = tested_output.get(key)
            if key not in tested_output:
                raise IpynbTesterError("Execution id {}, the '{}' key is present in ref's "
                                       "cell output but not in test cell".format(cell_id, key))
            elif key == 'data':
                self._compare_outputs(reference, tested, cell)
            elif key == 'image/png':
                if self._is_diff_png(reference, tested, save=True, output_dir=PNG_SAVE_DIRECTORY):
                    raise IpynbTesterError(
                        "Execution id {}, png images differ. "
                        "Images diff are stored here: {}".format(cell_id, PNG_SAVE_DIRECTORY))
            elif key not in skip_compare:
                if self._fuzzy_compare(reference, tested):
                    continue
                if isinstance(reference, six.string_types) and isinstance(tested, six.string_types):
                    diff = Differ().compare(reference.splitlines(True), tested.splitlines(True))
                    msg = 'Execution id {}, text outputs differ:\n'.format(cell_id) + ''.join(diff)
                    raise IpynbTesterError(msg)
                raise IpynbTesterError('Execution id {}, {} != {}'.format(cell_id, reference,
                                                                          tested))

    @staticmethod
    def _add_cell_output(outputs, msg_type, properties):
        """ Add an output to a given cell output dictionary """
        out = nbformat.NotebookNode(output_type=msg_type)
        out.update(properties)
        outputs.append(out)

    def _run_cell(self, shell, iopub, cell, kc):
        """ Runs cell and creates the output list of dictionaries containing all
        useful information about the cell execution and results.

        Inputs:
            shell : kernel client's shell channel
            iopub : kernel client's iopub channel
            cell : cell to run
            kc : the kernel client

        Returns:
            list of dictionary : output of cells with all information and figures in base64
        """
        kc.execute(cell.source)
        shell.get_msg(timeout=self.cell_timeout)
        outs = []
        in_stream_mode = False
        last_name = ''
        while True:
            try:
                # for the kernel to publish results to frontends
                msg = iopub.get_msg(timeout=self.query_message_timeout)
            except Empty:
                break
            msg_type = msg['msg_type']
            content = msg.get('content', dict())
            name = content.get('name', '')
            # need to be careful with the stream name
            # streams from two sources are in a different cell outputs
            if msg_type != 'stream' or last_name != name:
                in_stream_mode = False
            if msg_type == 'clear_output':
                return []
            elif msg_type in ('status', 'execute_input'):
                pass
            elif msg_type == 'stream':
                if in_stream_mode:
                    outs[-1]['text'] += content['text']
                else:
                    props = {'text': content['text'], 'name': name}
                    self._add_cell_output(outs, msg_type, props)
                    in_stream_mode = True
            elif msg_type in ('display_data', 'pyout', 'execute_result'):
                props = {'data': content['data'],
                         'metadata': content['metadata']}
                if msg_type in ('execute_result', 'pyout'):
                    props.update(
                        {'execution_count': content['execution_count']})
                self._add_cell_output(outs, msg_type, props)
            elif msg_type in ('pyerr', 'error'):
                try:
                    stack = msg['content']['traceback']
                except ValueError:
                    stack = 'Cannot retrieve stacktrace'
                raise IpynbTesterError(
                    'Execution id {}, error while running code:\n{}'.format(cell['execution_count'],
                                                                            ''.join(stack)))
            else:
                print("Unhandled iopub msg:", msg_type)
            last_name = name
        return outs

    def test_notebook(self, nb_path, dry_run=False):
        """ Launches a Ipython kernel manager and runs the ipython notebook from path.

        Notes:
            A dry run will only execute cells and check if the message type is known.

        Inputs:
            nb_path : path to the tested notebook
            dry_run : boolean for comparing cell outputs with reference outputs
        """
        with setup_notebook_kernel(nb_path) as nbc:
            for cell in nbc.notebook.cells:
                if cell.cell_type != 'code':
                    continue
                outs = self._run_cell(nbc.kernel_client.shell_channel,
                                      nbc.kernel_client.iopub_channel,
                                      cell,
                                      nbc.kernel_client)
                if dry_run:
                    continue
                for out, ref in zip(outs, cell.outputs):
                    self._compare_outputs(ref, out, cell)
