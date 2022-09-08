import os
import warnings
import tempfile
from pathlib import Path
from taurus_datamover import Datamover, CacheWorkspace, waitfor, save_to_project


class ConfirmationRequiredException(IOError):
    '''Exception that is raised when a dangerous operation is called without im_sure=True
    '''
    pass


class ProjectFileTransfer:
    """
    Connects a project space on the cluster with a fileserver share via an export node.

    After initializing, source mount and target project space, you can:
    * `imread` an image file from the fileserver
    * `imsave` an image file to the fileserver
    * `sync_with_fileserver` Synchronize a whole directory tree with the fileserver (using rsync).
    * `get_file`s to the project space,
    * `list_files` in the project space and
    * `list_fileserver_files` list files on the fileserver
    * `remove_file`s from the project space.

    See also
    --------
    .. [0] https://doc.zih.tu-dresden.de/data_transfer/datamover/
    .. [1] https://gitlab.mn.tu-dresden.de/bia-pol/taurus-datamover
    """

    def __init__(self, source_fileserver_dir: str, target_project_space_dir: str,
                 datamover_path: str = '/sw/taurus/tools/slurmtools/default/bin/',
                 workspace_exe_path: str = '/usr/bin/'):
        """
        Sets up a connection between a directory on the fileserver and a directory on the project space.

        Parameters
        ----------
        source_fileserver_dir : str
            Fileserver mount on the export node, e.g. /grp/g_my_group/userdir/
        target_project : str
            Project space on the cluster, e.g. /projects/p_my_project/userdir/
        datamover_path: str, optional
            the path where the datamover tools reside, by default /sw/taurus/tools/slurmtools/default/bin/
        """
        self.source_fileserver_dir = Path(source_fileserver_dir)
        self.target_project_space_dir = Path(target_project_space_dir)
        self.datamover = Datamover(path_to_exe=datamover_path)
        self.workspace_exe_path = workspace_exe_path
        self.cache = None
        self.temporary_directory = None
        self._initialize_tmp()
        self._ensure_project_dir_exists()

    def imread(self, filename, *args, **kw):
        """
        Load an image from a file.

        First we look for the file on the project space. If it is not found there, we try to copy it over from the fileserver and then open it.

        Parameters
        ----------
        filename : str
            The file that should be loaded. The path should be an absolute path to a readable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        all other arguments are passed down to [scikit-image.io.imread](https://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imread)

        Returns
        -------
        ndarray containing the image data

        """
        from skimage.io import imread
        full_path = self.get_file(filename)
        return imread(str(full_path), *args, **kw)

    def imsave(self, filename, data, *args, target: str = 'project', **kw):
        """
        Save an image to a file on the project space, fileserver or any other location you have write access from a node.

        Be aware that on taurus, unlike the login node, computing nodes don't have write access to the project space. Therefore, you need to use this function to save to the project space rather than just `skimage.io.imsave`.

        Parameters
        ----------
        filename : str
            The filename where the image should be saved. The path should be an absolute path to a writable file, or relative either to `target_project_space_dir` or to `source_fileserver_dir`.
        data : ndarray
            image data
        target : str, optional, by default 'project'
            where the file should be saved. 'fileserver' means the file will be saved to 'source_fileserver_dir' otherwise, 'target project_space_dir'
        all other arguments are passed down to [scikit-image.io.imsave](https://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imsave)

        Returns
        -------
        result of skimage.io.imsave

        """
        from skimage.io import imsave
        return self.save_file(imsave, filename, data, *args, target=target, **kw)

    def numpy_load(self, filename, *args, **kw):
        """
        Load a numpy array from a file.

        First we look for the file on the project space. If it is not found there, we try to copy it over from the fileserver and then open it.

        Parameters
        ----------
        filename : str
            The file that should be loaded. The path should be an absolute path to a readable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        all other arguments are passed down to [numpy.load](https://numpy.org/doc/stable/reference/generated/numpy.load.html)

        Returns
        -------
        ndarray containing the numpy data

        """
        from numpy import load as np_load
        full_path = self.get_file(filename)
        return np_load(str(full_path), *args, **kw)

    def numpy_save(self, filename, data, *args, target: str = 'project', **kw):
        """
        Save a numpy array to a file on the project space, fileserver or any other location you have write access from a node.

        Be aware that on taurus, unlike the login node, computing nodes don't have write access to the project space. Therefore, you need to use this function to save to the project space rather than just `numpy.save`.

        Parameters
        ----------
        filename : str
            The filename where the data should be saved. The path should be an absolute path to a writable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        data : ndarray
            numpy data
        target : str, optional, by default 'project'
            where the file should be saved. 'fileserver' means the file will be saved to 'source_fileserver_dir' otherwise, 'target project_space_dir'
        all other arguments are passed down to [numpy.save](https://numpy.org/doc/stable/reference/generated/numpy.save.html)

        Returns
        -------
        result of numpy.save

        """
        from numpy import save as np_save
        return self.save_file(np_save, filename, data, *args, target=target, **kw)

    def numpy_savez_compressed(self, filename, data, *args, target: str = 'project', **kw):
        """
        Save several numpy arrays into a single file in compressed .npz format to the project space, fileserver or any other location you have write access from a node.

        Be aware that on taurus, unlike the login node, computing nodes don't have write access to the project space. Therefore, you need to use this function to save to the project space rather than just `numpy.savez_compressed`.

        Parameters
        ----------
        filename : str
            The filename where the data should be saved. The path should be an absolute path to a writable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        data : ndarray
            numpy data
        target : str, optional, by default 'project'
            where the file should be saved. 'fileserver' means the file will be saved to 'source_fileserver_dir' otherwise, 'target project_space_dir'
        all other arguments are passed down to [numpy.savez_compressed](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html)

        Returns
        -------
        result of numpy.savez_compressed

        """
        from numpy import savez_compressed as np_savez_compressed
        return self.save_file(np_savez_compressed, filename, data, *args, target=target, **kw)

    def numpy_loadtxt(self, filename, *args, **kw):
        """
        Load a numpy array from a text file.

        First we look for the file on the project space. If it is not found there, we try to copy it over from the fileserver and then open it.

        Parameters
        ----------
        filename : str
            The file that should be loaded. The path should be an absolute path to a readable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        all other arguments are passed down to [numpy.loadtxt](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html)

        Returns
        -------
        ndarray containing the numpy data

        """
        from numpy import loadtxt as np_loadtxt
        full_path = self.get_file(filename)
        return np_loadtxt(str(full_path), *args, **kw)

    def numpy_savetxt(self, filename, data, *args, target: str = 'project', **kw):
        """
        Save a numpy array to a text file on the project space, fileserver or any other location you have write access from a node.

        Be aware that on taurus, unlike the login node, computing nodes don't have write access to the project space. Therefore, you need to use this function to save to the project space rather than just `numpy.save`.

        Parameters
        ----------
        filename : str
            The filename where the data should be saved. The path should be an absolute path to a writable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        data : ndarray
            numpy data
        target : str, optional, by default 'project'
            where the file should be saved. 'fileserver' means the file will be saved to 'source_fileserver_dir' otherwise, 'target project_space_dir'
        all other arguments are passed down to [numpy.savetxt](https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html)

        Returns
        -------
        result of numpy.save

        """
        from numpy import savetxt as np_savetxt
        return self.save_file(np_savetxt, filename, data, *args, target=target, **kw)

    def pandas_read_csv(self, filename, *args, **kw):
        """
        Load a pandas dataframe from a csv file.

        First we look for the file on the project space. If it is not found there, we try to copy it over from the fileserver and then open it.

        Parameters
        ----------
        filename : str
            The file that should be loaded. The path should be an absolute path to a readable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        all other arguments are passed down to [pandas.read_csv](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table)

        Returns
        -------
        pandas dataframe containing the csv data

        """
        from pandas import read_csv as pd_read_csv
        full_path = self.get_file(filename)
        return pd_read_csv(str(full_path), *args, **kw)

    def pandas_to_csv(self, filename, data, *args, target: str = 'project', **kw):
        """
        Save a pandas dataframe to a csv file on the project space, fileserver or any other location you have write access from a node.

        Be aware that on taurus, unlike the login node, computing nodes don't have write access to the project space. Therefore, you need to use this function to save to the project space rather than just `dataframe.to_csv`.

        Parameters
        ----------
        filename : str
            The filename where the data should be saved. The path should be an absolute path to a writable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        data : pandas.DataFrame
        target : str, optional, by default 'project'
            where the file should be saved. 'fileserver' means the file will be saved to 'source_fileserver_dir' otherwise, 'target project_space_dir'
        all other arguments are passed down to [pandas.DataFrame.to_csv](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-store-in-csv)

        Returns
        -------
        result of pandas.DataFrame.to_csv

        """
        return self.save_file(data.to_csv, filename, *args, target=target, **kw)

    def pandas_read_json(self, filename, *args, **kw):
        """
        Load a pandas dataframe from a json text file.

        First we look for the file on the project space. If it is not found there, we try to copy it over from the fileserver and then open it.

        Parameters
        ----------
        filename : str
            The file that should be loaded. The path should be an absolute path to a readable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        all other arguments are passed down to [pandas.read_json](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-json-reader)

        Returns
        -------
        pandas dataframe containing the json data

        """
        from pandas import read_json as pd_read_json
        full_path = self.get_file(filename)
        return pd_read_json(str(full_path), *args, **kw)

    def pandas_to_json(self, filename, data, *args, target: str = 'project', **kw):
        """
        Save a pandas dataframe to a json file on the project space, fileserver or any other location you have write access from a node.

        Be aware that on taurus, unlike the login node, computing nodes don't have write access to the project space. Therefore, you need to use this function to save to the project space rather than just `dataframe.to_json`.

        Parameters
        ----------
        filename : str
            The filename where the data should be saved. The path should be an absolute path to a writable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        data : pandas.DataFrame
        target : str, optional, by default 'project'
            where the file should be saved. 'fileserver' means the file will be saved to 'source_fileserver_dir' otherwise, 'target project_space_dir'
        all other arguments are passed down to [pandas.DataFrame.to_json](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-json-writer)

        Returns
        -------
        result of pandas.DataFrame.to_json

        """
        return self.save_file(data.to_json, filename, *args, target=target, **kw)

    def pandas_read_excel(self, filename, *args, **kw):
        """
        Load a pandas dataframe from a excel file.

        First we look for the file on the project space. If it is not found there, we try to copy it over from the fileserver and then open it.

        Parameters
        ----------
        filename : str
            The file that should be loaded. The path should be an absolute path to a readable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        all other arguments are passed down to [pandas.read_excel](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html)

        Returns
        -------
        pandas dataframe containing the excel data

        """
        from pandas import read_excel as pd_read_excel
        full_path = self.get_file(filename)
        return pd_read_excel(str(full_path), *args, **kw)

    def pandas_to_excel(self, filename, data, *args, target: str = 'project', **kw):
        """
        Save a pandas dataframe to an excel file on the project space, fileserver or any other location you have write access from a node.

        Be aware that on taurus, unlike the login node, computing nodes don't have write access to the project space. Therefore, you need to use this function to save to the project space rather than just `dataframe.to_excel`.

        Parameters
        ----------
        filename : str
            The filename where the data should be saved. The path should be an absolute path to a writable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        data : pandas.DataFrame
        target : str, optional, by default 'project'
            where the file should be saved. 'fileserver' means the file will be saved to 'source_fileserver_dir' otherwise, 'target project_space_dir'
        all other arguments are passed down to [pandas.DataFrame.to_excel](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html#pandas.DataFrame.to_excel)

        Returns
        -------
        result of pandas.DataFrame.to_excel

        """
        return self.save_file(data.to_excel, filename, *args, target=target, **kw)

    def pandas_read_hdf(self, filename, *args, **kw):
        """
        Load a pandas dataframe from a binary hdf file.

        First we look for the file on the project space. If it is not found there, we try to copy it over from the fileserver and then open it.

        Parameters
        ----------
        filename : str
            The file that should be loaded. The path should be an absolute path to a readable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        all other arguments are passed down to [pandas.read_hdf](https://pandas.pydata.org/docs/reference/api/pandas.read_hdf.html)

        Returns
        -------
        pandas dataframe containing the hdf data

        """
        from pandas import read_hdf as pd_read_hdf
        full_path = self.get_file(filename)
        return pd_read_hdf(str(full_path), *args, **kw)

    def pandas_to_hdf(self, filename, data, *args, target: str = 'project', **kw):
        """
        Save a pandas dataframe to a hdf file on the project space, fileserver or any other location you have write access from a node.

        Be aware that on taurus, unlike the login node, computing nodes don't have write access to the project space. Therefore, you need to use this function to save to the project space rather than just `dataframe.to_hdf`.

        Parameters
        ----------
        filename : str
            The filename where the data should be saved. The path should be an absolute path to a writable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        data : pandas.DataFrame
        target : str, optional, by default 'project'
            where the file should be saved. 'fileserver' means the file will be saved to 'source_fileserver_dir' otherwise, 'target project_space_dir'
        all other arguments are passed down to [pandas.DataFrame.to_hdf](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_hdf.html#pandas.DataFrame.to_hdf)

        Returns
        -------
        result of pandas.DataFrame.to_json

        """
        return self.save_file(data.to_hdf, filename, 'data', *args, target=target, **kw)

    csv_load = pandas_read_csv

    csv_save = pandas_to_csv

    def save_file(self, save_function: callable, filename: str, *args, target: str = 'project', **kw):
        """
        Save data to a file on the project space, fileserver or any other location you have write access from a node.

        Be aware that, unlike the login node, computing nodes don't have write access to the project space. Therefore, you need to use this function to save to the project space rather than just the save_function directly.

        Parameters
        ----------
        save_function : callable
            the function that saves the data to disk
        filename : str
            The filename where the image should be saved. The path should be an absolute path to a writable file, or relative either to target_project_space_dir or to source_fileserver_dir.
        data : ndarray
            image data
        target : str, optional
            where the file should be saved. 'fileserver' means the file will be saved to 'source_fileserver_dir' otherwise, 'target project_space_dir'
        all other arguments are passed down to save_function

        Returns
        -------
        result of save_function

        """
        filename = filename.replace("\\", "/")
        full_path = Path(filename)
        if os.access(full_path.parent, os.W_OK) and str(full_path.parent) != '.':
            return save_function(str(full_path), *args, **kw)
        if str(filename).startswith(str(self.target_project_space_dir)):
            target_path = str(self.target_project_space_dir / filename)
        elif str(filename).startswith(str(self.source_fileserver_dir)):
            target_path = str(self.source_fileserver_dir / filename)
        else:
            if target != 'fileserver':
                target_path = str(self.target_project_space_dir / filename)
            else:
                target_path = str(self.source_fileserver_dir / filename)
        return save_to_project(save_function, str(target_path), *args, cache_workspace=self.cache,
                               path_to_datamover=self.datamover.path_to_exe, path_to_workspace_tools=self.workspace_exe_path, **kw)

    def sync_with_fileserver(self, direction: str = 'from fileserver', delete: bool = False,
                             overwrite_newer: bool = False, im_sure: bool = False, dry_run: bool = False, background: bool = True):
        '''Synchronize a whole directory tree with the fileserver (using rsync). By default, Does not delete files, but overwrites existing files if they are older.

        By default, we sync from the fileserver to the project space (direction='from fileserver'). If you want to synchronize from the project space to the fileserver, use direction='to fileserver'.

        Beware that this recursively copies _all_ the data. So if you have an error in source mount or target project space, this might create a mess.
        If you are unsure, first call the method with dry_run=True. That way, no data will be transferred, and you can check the output.
        This behavior is enforced for dangerous operations that might cause data loss.
        Dangerous operations are:
        1. setting delete=True
        2. setting overwrite_newer=True
        3. syncing the entire fileserver with the entire project space.

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
        if direction == 'from_fileserver' or direction == 'from fileserver':
            options.append(str(self.source_fileserver_dir) + '/')
            options.append(str(self.target_project_space_dir))
        else:
            options.append(str(self.target_project_space_dir) + '/')
            options.append(str(self.source_fileserver_dir))
        confirmation_required = delete or overwrite_newer
        if self.target_project_space_dir.parent == list(self.target_project_space_dir.parents)[
                -2] and self.source_fileserver_dir.parent == list(self.source_fileserver_dir.parents)[-2]:
            # syncing the entire fileserver directly into target_project_space_dir
            # requires confirmation because it might affect data of other users of the
            # same project space
            confirmation_required = True
        if dry_run:
            confirmation_required = False  # dry runs are never dangerous
            options[0] += 'n'
        if confirmation_required and not im_sure:
            warnings.warn(
                'What you are trying to do requires confirmation. Enforcing dry-run...')
            options[0] += 'n'
            process = self.datamover.dtrsync(*options)
            waitfor(process, discard_output=False, quiet=False)
            out, _ = process.communicate()
            raise ConfirmationRequiredException(
                'If you are sure you know what you are doing, call this method again with te keyword argument "im_sure=True".\nBut before you do that, please carefully check the output of the dry-run and make sure that is what you intended: {}'.format(out))
        process = self.datamover.dtrsync(*options)
        waitfor(process, discard_output=False, quiet=False)
        return process.communicate()

    def get_file(self, filename: str, timeout_in_s: float = -1,
                 wait_for_finish: bool = True) -> Path:
        '''Ensures that the computing node has access to a file. If necessary, the file is retrieved from a mounted fileserver share.

        Before transferring the file, local directories are checked in the following order:
        1. filename (in case the user gave a path to an accessible file)
        2. (temporary directory)/file.name,
        3. /target_project_space_dir/filename (in case the user gave a path relative to the target project space).
        4. Only if no file of the same name is found, the file is retrieved from the fileserver.

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
            # then check, if the file exists in tmp
            full_path = self.temporary_directory_path / full_path.name
            if full_path.is_file():
                return full_path
            else:
                # then check if the user meant a path relative to the target
                # project
                full_path = self.target_project_space_dir / filename
                if full_path.is_file():
                    return full_path
        # if we can't find the file locally, retrieve it from the fileserver
        # (into tmp)
        filename = filename.replace("\\", "/")
        source_file = self.source_fileserver_dir / filename
        # copy the file into tmp
        target_file = self.temporary_directory_path / source_file.name

        # start a process, submitting the copy-job
        process = self.datamover.dtcp('-r', str(source_file),
                                      str(target_file))
        exit_code = waitfor(process, quiet=False)
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
        return [str(f)
                for f in sorted(self.target_project_space_dir.glob('**/*'))]

    def list_fileserver_files(self, timeout_in_s: float = 30):
        """
        Get a list of files located in the project space

        Returns
        -------
        List of strings
        """
        process = self.datamover.dtls('-R1', str(self.source_fileserver_dir))
        exit_code = waitfor(process, timeout_in_s=timeout_in_s, discard_output=False, quiet=False)
        out, err = process.communicate()
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
        pathlib.Path object : if we didn't wait
        int : exit code of rm operation if we waited

        """
        if not str(filename).startswith(str(self.target_project_space_dir)):
            filename = self.target_project_space_dir / filename
        else:
            filename = Path(filename)

        process = self.datamover.dtrm('-r', str(filename))

        if not wait_for_finish:
            return process

        exit_code = waitfor(process)
        if exit_code > 0:
            raise IOError('Could not remove file: {}'.format(str(filename)))

        return exit_code

    def _initialize_tmp(self):
        '''Delete all temporary data and create a new, empty temp directory.
        '''
        self.cache = CacheWorkspace(path_to_exe=self.workspace_exe_path)
        self.temporary_directory = tempfile.TemporaryDirectory(prefix=self.cache.name + '/')
        self.temporary_directory_path = Path(self.temporary_directory.name)

    def _ensure_project_dir_exists(self):
        if not self.target_project_space_dir.exists():
            with tempfile.TemporaryDirectory() as temporary_dir:
                i = 0
                while self.target_project_space_dir.parents[i].exists() == False:
                    i += 1
                cached_target_dir = Path(temporary_dir).joinpath(*self.target_project_space_dir.parts[-(i + 1):])
                cached_target_dir.mkdir(parents=True, exist_ok=True)
                process = self.datamover.dtrsync('-a', temporary_dir + '/',
                                                 str(self.target_project_space_dir.parents[i]) + '/')
                exit_code = waitfor(process)
            if exit_code > 0:
                raise IOError('Could not create target project space dir: {}'.format(
                    str(self.target_project_space_dir)))

    def __del__(self):
        '''Clean up the cache when the object is deleted
        '''
        try:
            self.temporary_directory.cleanup()
            self.cache.cleanup()
        except FileNotFoundError:
            pass
