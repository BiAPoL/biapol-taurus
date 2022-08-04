from multiprocessing.connection import wait
from pathlib import Path
from taurus_datamover import Datamover, waitfor
import warnings


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
            out, err = proc.communicate()
            raise IOError(
                'copy operation exited with error: {}\nand output: {}'.format(
                    err, out))

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
            out, err = proc.communicate()
            raise IOError(
                'delete operation exited with error: {}\nand output: {}'.format(
                    err, out))
