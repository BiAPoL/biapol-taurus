import os
import warnings
import tempfile
from pathlib import Path
from skimage.io import imread, imsave
from taurus_datamover import Datamover, waitfor


class DangerousOperationException(IOError):
    '''
    Exception that is raised when a dangerous operatino is called without im_sure=True
    '''
    pass


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
            Fileserver mount on the export node, e.g. /grp/g_my_group/userdir/
        target_project : str
            Project space on the cluster, e.g. /projects/p_my_project/userdir/
        dm_path: str, optional
            the path where the datamover tools reside, by default /sw/taurus/tools/slurmtools/default/bin/
        """
        self.source_mount = Path(source_mount)
        self.target_project_space = Path(target_project_space)
        self.dm = Datamover(path_to_exe=dm_path)
        self.tmp = tempfile.TemporaryDirectory()

    def sync_with_fileserver(self, direction: str = 'from fileserver', delete: bool = False,
                             overwrite_newer: bool = False, im_sure: bool = False, dry_run: bool = False, background: bool = True):
        '''Synchronize a whole directory tree with the fileserver (using rsync). Does not delete files, but overwrites existing files.

        By default, we sync from the fileserver to the project space (direction='from fileserver'). If you want to synchronize from the project space to the fileserver, use direction='to fileserver'.

        Beware that this recursively copies _all_ the data. So if you have an error in source mount or target project space, this might create a mess. If you are unsure, first call the method with dry_run=True. That way, no data will be transferred, and you can check the output. This behavior is enforced for dangerous operations that might cause data loss. Dangerous operations are: setting delete=true or overwrite_newer=true and syncing the entire fileserver with the entire project space.

        Parameters
        ----------
        direction : str, optional
            The direction in which to sync, by default 'from fileserver'
        delete : bool, optional
            Delete files that do not exist on the fileserver on the target project space (add rsync --delete flag), by default False
        overwrite_newer : bool, optional
            Overwrite files on the target project space even if they are newer (removes rsync -u flag), by default False
        im_sure : bool, optional
            Confirm that you are sure and skip the dry-run for dangerous operations, by default False
        dry_run : bool, optional
            Enforce a dry-run (add rsunc -n flag), by default False
        background : bool, optional
            Run in the background and do not wait until the sync is complete, by default True

        Returns
        -------
        subprocess.CompletedProcess object (if bacground=True (default))
            the CompletedProcess object created by subprocess.Popen. This can be used to retrieve the command output with the communicate() method: https://docs.python.org/3/library/subprocess.html
        tuple of strings (if background=False)
            the first element of the tuple is the standard output (stdout) of the process, the second element is the error (stderr).
        '''
        if overwrite_newer:
            options = ['-av']
        else:
            options = ['-auv']
        if delete:
            options.append('--delete')
        if direction == 'from_fileserver':
            options.append(str(self.source_mount))
            options.append(str(self.target_project_space))
        else:
            options.append(str(self.target_project_space))
            options.append(str(self.source_mount))
        dangerous = delete or overwrite_newer
        if self.target_project_space.parent == self.target_project_space.parents[
                -2] and self.source_mount.parent == self.source_mount.parents[-2]:
            # syncing the entire fileserver directly into target_project_space
            # is dangerous because it might affect data of other users of the
            # same project space
            dangerous = True
        if dry_run:
            dangerous = False  # dry runs are never dangerous
            options[0] += 'n'
        if dangerous and not im_sure:
            warnings.warn(
                'What you are trying to do is dangerous. Enforcing dry-run...')
            options[0] += 'n'
            proc = self.dm.dtrsync(*options)
            waitfor(proc, discard_output=False)
            out, _ = proc.communicate()
            raise DangerousOperationException(
                'If you are sure you know what you are doing, call this method again with te keyword argument "im_sure=True".\nBut before you do that, please carefully check the output of the dry-run and make sure that is what you intended: {}'.format(out))
        proc = self.dm.dtrsync(*options)
        waitfor(proc, discard_output=False)
        return proc.communicate()

    def get_file(self, filename: str, timeout_in_s: float = -1,
                 wait_for_finish: bool = True) -> Path:
        '''Ensures that the computing node has access to a file. If necessary, the file is retrieved from a mounted fileserver share.

        Before transferring the file, local directories are checked in the following order: 1. filename (in case the user gave a path to an accessible file) 2. /tmp/(temporary directory)/file.name, 3. /target_project_space/filename (in case the user gave a path relative to the target project space). Only if no file of the same name is found, the file is retrieved from the fileserver.

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

        Returns
        -------
        Path
            The path where the file is accessible to the computing node

        '''
        # first check if the file as given by the user exists locally
        full_path = Path(filename)
        if full_path.is_file():
            return full_path
        else:
            # then check, if the file exists on /tmp
            full_path = Path(self.tmp) / full_path.name
            if full_path.is_file():
                return full_path
            else:
                # then check if the user meant a path relative to the target
                # project
                full_path = self.target_project_space / filename
                if full_path.is_file():
                    return full_path
        # if we can't find the file locally, retrieve it from the fileserver
        # (into /tmp)
        filename = filename.replace("\\", "/")
        source_file = self.source_mount / filename
        # copy the file into /tmp
        target_file = Path(self.tmp) / source_file.name

        # start a process, submitting the copy-job
        proc = self.dm.dtcp('-r', str(source_file),
                            str(target_file))
        exit_code = waitfor(proc)
        if exit_code > 0:
            raise IOError(
                'Could not get file from fileserver: {}'.format(
                    str(source_file)))
        return target_file

    def list_files(self):
        """
        Get a list of files located in the project space

        Returns
        -------
        List of strings
        """
        return [str(f) for f in sorted(self.target_project_space.glob('**/*'))]

    def list_fileserver_files(self, timeout_in_s: float = 30):
        """
        Get a list of files located in the project space

        Returns
        -------
        List of strings
        """
        proc = self.dm.dtls('-R1', str(self.source_mount))
        exit_code = waitfor(
            proc,
            timeout_in_s=timeout_in_s,
            discard_output=False)
        out, err = proc.communicate()
        return out.decode('utf-8').split("\n")

    def remove_file(self, filename, wait_for_finish: bool = False):
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
        if not str(filename).startswith(str(self.target_project_space)):
            filename = self.target_project_space / filename
        else:
            filename = Path(filename)

        proc = self.dm.dtrm('-r', str(filename))

        if not wait_for_finish:
            return True

        exit_code = waitfor(proc)
        if exit_code > 0:
            raise IOError('Could not remove file: {}'.format(str(filename)))

    def cleanup_tmp(self):
        '''Delete all temporary data and create a new, empty temp directory.
        '''
        self.tmp.cleanup()
        self.tmp = tempfile.TemporaryDirectory()

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
        full_path = self.get_file(filename)
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

    def __del__(self):
        '''Clean up the temporary directory when the object is deleted
        '''
        self.tmp.cleanup()
