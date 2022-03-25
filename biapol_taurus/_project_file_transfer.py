from pathlib import Path
import warnings
import subprocess
import os


class ProjectFileTransfer:
    """
    Connects a project space on the cluster with a fileserver share via an export node.

    After initializing, source mount and target project space, you can
    * `get_file`s to the project space,
    * `list_files` in the project space and
    * `remove_file`s from the project space.

    """

    def __init__(self, source_mount: str, target_project_space: str):
        """
        Sets up a project-space - fileserver-mount connection.

        Parameters
        ----------
        source_mount : str
            Fileserver mount on the export node, e.g. /grp/g_my_group/
        target_project : str
            Project space on the cluster, e.g. /projects/p_my_project/
        """
        if not source_mount.endswith("/"):
            source_mount = source_mount + "/"
        if not target_project_space.endswith("/"):
            target_project_space = target_project_space + "/"
        self.source_mount = source_mount
        self.target_project_space = target_project_space

        # todo: check if those folders exist

        self.dtcp = "/sw/taurus/tools/slurmtools/default/bin/dtcp"
        self.dtrm = "/sw/taurus/tools/slurmtools/default/bin/dtrm"

    def save_file(self, filename: str, timeout_in_s: float = -1,
                  wait_for_finish: bool = False):
        """
        Transfer a filename from the project space to a mounted fileserver.

        Parameters
        ----------
        filename : str
            DESCRIPTION.
        timeout_in_s : float, optional
            DESCRIPTION. The default is -1.
        wait_for_finish : bool, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.
        """
        filename = os.path.normpath(filename)
        source = os.path.join(self.target_project_space, filename)
        target = os.path.join(self.source_mount, filename)
        print(source, target)

        if os.path.exists(target):
            warnings.warn("\nFile exists already: " + target)
            return

        # start a process, submitting the copy-job
        self._run_command([self.dtcp, '-r', source, target])

        if wait_for_finish:
            self._check_arrival(source, target, timeout_in_s=timeout_in_s)

    def get_file(self, filename: str, timeout_in_s: float = -1, wait_for_finish: bool = True):
        """
        Transfers a file from a mounted fileserver share to the project space.

        Parameters
        ----------
        filename: str
            filename as on the fileserver, if the file is stored on the fileserver under
            \\fileserver\mount\folder\data.txt
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
        source_file = self.source_mount + filename
        filename_only = filename.split("/")[-1]
        target_file = self.target_project_space + filename_only

        if Path(target_file).is_file():
            warnings.warn("\nFile exists already: " + target_file)
            return

        # start a process, submitting the copy-job
        self._run_command([self.dtcp, '-r', source_file,
                           self.target_project_space])

        if wait_for_finish:
            new_location = self._check_arrival(source_file, target_file,
                                               timeout_in_s=timeout_in_s)
            return new_location

    def list_files(self):
        """
        Get a list of files located in the project space

        Returns
        -------
        List of strings
        """
        # print(self._run_command(["ls", self.target_project_space, "-l"]))
        from os import listdir
        from os.path import isfile, join

        return [f for f in listdir(self.target_project_space) if isfile(join(self.target_project_space, f))]


    def remove_file(self, filename, timeout_in_s: float = 20, wait_for_finish: bool = False):
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
            filename = self.target_project_space + filename

        self._run_command([self.dtrm, filename])

        if not wait_for_finish:
            return True

        print("Waiting .", end='', flush=True)
        import time
        start_time = time.time()

        while (True):
            if not Path(filename).is_file():
                print("")
                return True  # successfully transferred file

            if timeout_in_s > 0 and (time.time() - start_time) > timeout_in_s:
                print("")
                warnings.warn("\nTimeout while deleting file:\n" + filename)
                return False

            print(".", end='', flush=True)
            time.sleep(0.5)  # sleep for half a second

    def _check_arrival(self,
                       source_file: str,
                       target_file: str,
                       timeout_in_s: float = -1):

        # wait and check repeatedly if the file arrived
        print("Waiting .", end='', flush=True)
        import time
        start_time = time.time()
        while (True):
            if Path(target_file).is_file():
                print("")
                return target_file  # successfully transferred file

            if timeout_in_s > 0 and (time.time() - start_time) > timeout_in_s:
                print("")
                warnings.warn("\nTimeout while transferring file:\n" + source_file + "\n->\n" + target_file)
                return None

            print(".", end='', flush=True)
            time.sleep(0.5)  # sleep for half a second

    def _run_command(self, command):
        """
        Execute a command on the terminal and return its output.

        Parameters
        ----------
        command : str or list of str
            The command to be executed, e.g. "ls -l" or ["ls", "-l"]

        Returns
        -------
        str, stdout output of the command
        """
        if isinstance(command, str):
            command = command.split(" ")

        # print(command)

        proc = subprocess.Popen(command, stdout=subprocess.PIPE)

        # collect std output from the process
        output = []
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            # the real code does filtering her
            output.append((line.rstrip()).decode("utf-8"))
        return "\n".join(output)
