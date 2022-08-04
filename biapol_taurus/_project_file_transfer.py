import os
import warnings
import tempfile
from pathlib import Path
from skimage.io import imread, imsave
from taurus_datamover import Datamover, waitfor


class ProjectFileTransfer:
    """
    Connects a project space on the cluster with a fileserver share via an export node.

    After initializing, source mount and target project space, you can
    * `get_file`s to the project space,
    * `list_files` in the project space and
    * `remove_file`s from the project space.

    """

    def __init__(self, source_mount: str, target_project_space: str,
                 dm_path: str = '/sw/taurus/tools/slurmtools/default/bin/'):
        """
        Sets up a project-space - fileserver-mount connection.

        Parameters
        ----------
        source_mount : str
            Fileserver mount on the export node, e.g. /grp/g_my_group/
        target_project : str
            Project space on the cluster, e.g. /projects/p_my_project/
        """
        self.source_mount = Path(source_mount)
        self.target_project_space = Path(target_project_space)
        self.dm = Datamover(path_to_exe=dm_path)

    def get_file(self, filename: str, timeout_in_s: float = -1,
                 wait_for_finish: bool = True):
        """
        Transfers a file from a mounted fileserver share to the project space.

        Parameters
        ----------
        filename: str
            filename as on the fileserver, if the file is stored on the fileserver under
            \\fileserver\\mount\folder\\data.txt
            You need to pass 'folder/data.txt' here.
        timeout_in_s: float, optional (default: endless)
            Timeout in seconds. This process will wait a bit and check repeatedly if the
            file arrived. Waiting will be interrupted when the timeout is reached.
        wait_for_finish : bool, optional (default: True)
            If True, will wait until the requested file arrived.

        See also
        --------
        .. [0] https://doc.zih.tu-dresden.de/data_transfer/datamover/
        """
        filename = filename.replace("\\", "/")
        source_file = self.source_mount / filename
        filename_only = filename.split("/")[-1]
        target_file = self.target_project_space / filename_only

        if Path(target_file).is_file():
            warnings.warn("\nFile exists already: " + str(target_file))
            return

        # start a process, submitting the copy-job
        proc = self.dm.dtcp('-r', str(source_file),
                            str(self.target_project_space))
        exit_code = waitfor(proc)
        if exit_code > 0:
            raise IOError(
                'Could not get file from fileserver: {}'.format(
                    str(source_file)))

    def list_files(self):
        """
        Get a list of files located in the project space

        Returns
        -------
        List of strings
        """
        return [str(f) for f in sorted(self.target_project_space.glob('**/*'))]

    def list_fileserver_files(self):
        """
        Get a list of files located in the project space

        Returns
        -------
        List of strings
        """
        proc = self.dm.dtls('-R1', str(self.target_project_space))
        exit_code = waitfor(proc)
        out, err = proc.communicate()
        if exit_code > 0:
            warnings.warn('list operation exited with error: {}'.format(err))
        return out.decode('utf-8').split("\n")

    def remove_file(self, filename, timeout_in_s: float = 20,
                    wait_for_finish: bool = False):
        """
        Removes a given file from the project space.

        Parameters
        ----------
        filename : str
            The file that should be removed from the given project space
        timeout_in_s : float, optional (default: 20s)
            Time we wait until the file might be deleted.
            Will not do anything as long as wait_for_finish = False
        wait_for_finish : bool, optional (default: False)
            Wait for file remove operation to be finished.

        Returns
        -------
        True if the file was deleted successfully (or we didn't wait)
        False if the timeout was reached

        """
        if not filename.startswith(self.target_project_space):
            filename = self.target_project_space / filename
        else:
            filename = Path(filename)

        proc = self.dm.dtrm('-r', str(filename))

        if not wait_for_finish:
            return True

        exit_code = waitfor(proc)
        if exit_code > 0:
            raise IOError('Could not remove file: {}'.format(str(filename)))

    def imread(self, filename, *args, **kw):
        """
        Load an image from a file.

        First we look for the file on the project space. If it is not found there, we try to copy it over from the fileserver and then open it.

        Parameters
        ----------
        filename : str
            The file that should be loaded. The path should be an absolute path to a readable file, or relative either to target_project_space or to source_mount.
        all other arguments are passed down to [scikit-image.io.imread](https://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imread)

        Returns
        -------
        ndarray containing the image data

        """
        full_path = Path(filename)
        if not full_path.is_file():
            full_path = self.target_project_space / filename
            if not full_path.is_file():
                self.get_file(filename=filename)
                full_path = self.target_project_space / filename
        return imread(str(full_path), *args, **kw)

    def imsave(self, filename, data, *args, project: bool = True, **kw):
        """
        Save an image to a file on the project space or fileserver.

        First we look for the file on the project space. If it is not found there, we try to copy it over from the fileserver and then open it.

        Parameters
        ----------
        filename : str
            The filename where the image should be saved. The path should be an absolute path to a writable file, or relative either to target_project_space or to source_mount.
        all other arguments are passed down to [scikit-image.io.imsave](https://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imsave)
        data : ndarray containing the image data

        Returns
        -------
        result of skimage.io.imsave

        """
        full_path = Path(filename)

        if os.access(full_path.parent, os.W_OK):
            return imsave(str(full_path), data, *args, **kw)
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_file = Path(tmpdirname) / filename.name
                output = imsave(str(temp_file), data, *args, **kw)
                if project:
                    proc = self.dm.dtmv(
                        str(temp_file), str(
                            self.target_project_space / filename))
                else:
                    proc = self.dm.dtmv(
                        str(temp_file), str(
                            self.source_mount / filename))
                waitfor(proc)
            return output
