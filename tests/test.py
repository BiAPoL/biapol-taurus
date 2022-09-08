from ctypes.wintypes import tagRECT
from time import sleep
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from skimage.io import imsave, imread

from taurus_datamover._mock import MockCluster

my_path = Path(__file__)
try:
    import biapol_taurus
except ModuleNotFoundError:
    import sys
    sys.path.append(str(my_path.parent.parent))
    import biapol_taurus


class TestProjectFileTransfer(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_cluster = MockCluster(commands=['dtls', 'dtcp', 'dtrm', 'dtmv', 'dtrsync'])
        self.project = self.mock_cluster.temp_path / 'project'
        self.project.mkdir()
        self.fileserver = self.mock_cluster.temp_path / 'fileserver'
        self.fileserver.mkdir()
        self.fileserver_userdir = self.fileserver / 'userdir'
        self.fileserver_userdir.mkdir()
        self.project_userdir = self.project / 'userdir'
        self.testdata = np.random.randn(64, 64)
        np.save(self.fileserver_userdir / 'testdata.npy', self.testdata, allow_pickle=False)
        imsave(self.fileserver_userdir / 'testimage.tif', self.testdata)
        self.mock_cluster.__enter__()
        self.pft = biapol_taurus.ProjectFileTransfer(
            source_fileserver_dir=str(self.fileserver_userdir),
            target_project_space_dir=str(self.project_userdir),
            datamover_path=self.mock_cluster.bin_path,
            workspace_exe_path=self.mock_cluster.bin_path,
            quiet=True)
        self.pandas_dataframe = pd.DataFrame.from_dict(
            {'test column': {'test row': 'a', 'test row2': 'b'}, 'test col2': {'test row': 1, 'test row2': 2}})
        self.pandas_dataframe.to_csv(self.fileserver_userdir / 'testdata.csv')
        return super().setUp()

    def tearDown(self) -> None:
        self.mock_cluster.tempdir.cleanup()
        return super().tearDown()

    def test_ensure_project_dir_exists(self):
        self.project_userdir.rmdir()
        pft = biapol_taurus.ProjectFileTransfer(
            source_fileserver_dir=str(self.fileserver_userdir),
            target_project_space_dir=str(self.project_userdir),
            datamover_path=self.mock_cluster.bin_path,
            workspace_exe_path=self.mock_cluster.bin_path)
        self.assertTrue(self.project_userdir.exists())

    def test_sync_from_fileserver(self):
        self.pft.sync_with_fileserver()
        self.assertTrue((self.project_userdir / 'testdata.npy').exists())
        self.assertTrue((self.project_userdir / 'testimage.tif').exists())
        numpy_data = np.load(self.project_userdir / 'testdata.npy')
        self.assertTrue(np.array_equal(self.testdata, numpy_data))

    def test_sync_to_fileserver(self):
        np.save(self.project_userdir / 'testdata_new.npy', self.testdata, allow_pickle=False)
        self.pft.sync_with_fileserver(direction='to fileserver')
        self.assertTrue((self.fileserver_userdir / 'testdata_new.npy').exists())
        self.assertTrue((self.fileserver_userdir / 'testdata.npy').exists())
        self.assertTrue((self.fileserver_userdir / 'testimage.tif').exists())
        numpy_data = np.load(self.fileserver_userdir / 'testdata_new.npy')
        self.assertTrue(np.array_equal(self.testdata, numpy_data))

    def test_sync_delete_not_confirmed(self):
        self.assertRaises(
            biapol_taurus.ConfirmationRequiredException,
            self.pft.sync_with_fileserver,
            direction='to fileserver',
            delete=True)
        self.assertTrue((self.fileserver_userdir / 'testdata.npy').exists())
        self.assertTrue((self.fileserver_userdir / 'testimage.tif').exists())

    def test_sync_delete_confirmed(self):
        self.pft.sync_with_fileserver(direction='to fileserver', delete=True, im_sure=True)
        self.assertFalse((self.fileserver_userdir / 'testdata.npy').exists())
        self.assertFalse((self.fileserver_userdir / 'testimage.tif').exists())

    def test_sync_overwrite_older(self):
        new_testdata = np.random.randn(64, 64)
        sleep(1)  # ensure that the source testdata file is at least 1s newer than the target file
        np.save(self.project_userdir / 'testdata.npy', new_testdata, allow_pickle=False)
        self.pft.sync_with_fileserver(direction='to fileserver')
        self.assertTrue((self.fileserver_userdir / 'testdata.npy').exists())
        self.assertTrue((self.fileserver_userdir / 'testimage.tif').exists())
        numpy_data = np.load(self.fileserver_userdir / 'testdata.npy')
        self.assertTrue(np.array_equal(new_testdata, numpy_data))
        self.assertFalse(np.array_equal(self.testdata, numpy_data))

    def test_sync_not_overwrite_newer(self):
        new_testdata = np.random.randn(64, 64)
        np.save(self.project_userdir / 'testdata.npy', new_testdata, allow_pickle=False)
        sleep(1)  # ensure that the target testdata file is at least 1s newer than the source file
        np.save(self.fileserver_userdir / 'testdata.npy', self.testdata, allow_pickle=False)
        self.pft.sync_with_fileserver(direction='to fileserver')
        self.assertTrue((self.fileserver_userdir / 'testdata.npy').exists())
        self.assertTrue((self.fileserver_userdir / 'testimage.tif').exists())
        numpy_data = np.load(self.fileserver_userdir / 'testdata.npy')
        self.assertFalse(np.array_equal(new_testdata, numpy_data))
        self.assertTrue(np.array_equal(self.testdata, numpy_data))

    def test_sync_not_overwrite_newer_unconfirmed(self):
        new_testdata = np.random.randn(64, 64)
        np.save(self.project_userdir / 'testdata.npy', new_testdata, allow_pickle=False)
        self.assertRaises(
            biapol_taurus.ConfirmationRequiredException,
            self.pft.sync_with_fileserver,
            direction='to fileserver',
            overwrite_newer=True)
        self.assertTrue((self.fileserver_userdir / 'testdata.npy').exists())
        self.assertTrue((self.fileserver_userdir / 'testimage.tif').exists())
        numpy_data = np.load(self.fileserver_userdir / 'testdata.npy')
        self.assertFalse(np.array_equal(new_testdata, numpy_data))
        self.assertTrue(np.array_equal(self.testdata, numpy_data))

    def test_sync_overwrite_newer_confirmed(self):
        new_testdata = np.random.randn(64, 64)
        np.save(self.project_userdir / 'testdata.npy', new_testdata, allow_pickle=False)
        sleep(1)  # ensure that the target testdata file is at least 1s newer than the source file
        np.save(self.fileserver_userdir / 'testdata.npy', self.testdata, allow_pickle=False)
        self.pft.sync_with_fileserver(direction='to fileserver', overwrite_newer=True, im_sure=True)
        self.assertTrue((self.fileserver_userdir / 'testdata.npy').exists())
        self.assertTrue((self.fileserver_userdir / 'testimage.tif').exists())
        numpy_data = np.load(self.fileserver_userdir / 'testdata.npy')
        self.assertTrue(np.array_equal(new_testdata, numpy_data))
        self.assertFalse(np.array_equal(self.testdata, numpy_data))

    def test_get_file_fileserver(self):
        cached_file = self.pft.get_file('testdata.npy')
        self.assertRegex(str(cached_file), r'.*/scratch/cache/.*')
        self.assertTrue(cached_file.exists())
        numpy_data = np.load(cached_file)
        self.assertTrue(np.array_equal(self.testdata, numpy_data))

    def test_get_file_project(self):
        new_testdata = np.random.randn(64, 64)
        np.save(self.project_userdir / 'testdata.npy', new_testdata, allow_pickle=False)
        cached_file = self.pft.get_file('testdata.npy')
        self.assertRegex(str(cached_file), r'.*/project/.*')
        self.assertTrue(cached_file.exists())
        numpy_data = np.load(cached_file)
        self.assertTrue(np.array_equal(new_testdata, numpy_data))

    def test_get_file_again(self):
        cached_file = self.pft.get_file('testdata.npy')
        # now we overwrite the original file to make sure that we get the cached file that still contains the old data
        new_testdata = np.random.randn(64, 64)
        np.save(self.fileserver_userdir / 'testdata.npy', new_testdata, allow_pickle=False)
        cached_file_again = self.pft.get_file('testdata.npy')
        self.assertEqual(cached_file, cached_file_again)
        self.assertTrue(cached_file_again.exists())
        numpy_data = np.load(cached_file)
        self.assertTrue(np.array_equal(self.testdata, numpy_data))

    def test_get_file_local(self):
        new_testdata = np.random.randn(64, 64)
        local_dir = self.mock_cluster.temp_path / 'local'
        local_dir.mkdir()
        local_file = local_dir / 'testdata.npy'
        np.save(local_file, new_testdata, allow_pickle=False)
        got_file = self.pft.get_file(local_file)
        self.assertEqual(local_file, got_file)
        self.assertTrue(got_file.exists())
        numpy_data = np.load(got_file)
        self.assertTrue(np.array_equal(new_testdata, numpy_data))

    def test_get_file_missing_file(self):
        self.assertRaises(IOError, self.pft.get_file, 'testdata_missing.npy')

    def test_remove_file(self):
        new_testdata = np.random.randn(64, 64)
        project_file = self.project_userdir / 'testdata.npy'
        np.save(project_file, new_testdata, allow_pickle=False)
        self.assertTrue(project_file.exists())
        self.pft.remove_file('testdata.npy', wait_for_finish=True)
        self.assertFalse(project_file.exists())

    def test_remove_file_nowait(self):
        new_testdata = np.random.randn(64, 64)
        project_file = self.project_userdir / 'testdata.npy'
        np.save(project_file, new_testdata, allow_pickle=False)
        self.assertTrue(project_file.exists())
        process = self.pft.remove_file('testdata.npy', wait_for_finish=False)
        self.assertTrue(project_file.exists())
        process.communicate()
        self.assertFalse(project_file.exists())

    def test_remove_file_full_path(self):
        new_testdata = np.random.randn(64, 64)
        project_file = self.project_userdir / 'testdata.npy'
        np.save(project_file, new_testdata, allow_pickle=False)
        self.assertTrue(project_file.exists())
        self.pft.remove_file(project_file, wait_for_finish=True)
        self.assertFalse(project_file.exists())

    def test_remove_file_missing_file(self):
        self.assertRaises(IOError, self.pft.remove_file, 'testdata_missing.npy', wait_for_finish=True)

    def test_save_file(self):
        project_file = self.project_userdir / 'saved_data.npy'
        self.pft.save_file(np.save, 'saved_data.npy', self.testdata, allow_pickle=False)
        self.assertTrue(project_file.exists())
        numpy_data = np.load(project_file)
        self.assertTrue(np.array_equal(self.testdata, numpy_data))

    def test_save_file_fileserver(self):
        fileserver_file = self.fileserver_userdir / 'saved_data.npy'
        self.pft.save_file(np.save, 'saved_data.npy', self.testdata, target='fileserver', allow_pickle=False)
        self.assertTrue(fileserver_file.exists())
        numpy_data = np.load(fileserver_file)
        self.assertTrue(np.array_equal(self.testdata, numpy_data))

    def test_save_file_absolute_path(self):
        project_file = self.project_userdir / 'saved_data.npy'
        self.pft.save_file(np.save, str(project_file), self.testdata, allow_pickle=False)
        self.assertTrue(project_file.exists())
        numpy_data = np.load(project_file)
        self.assertTrue(np.array_equal(self.testdata, numpy_data))

    def test_csv(self):
        with self.subTest('save csv'):
            test_file = self.fileserver_userdir / 'saved.csv'
            self.pft.csv_save('saved.csv', self.pandas_dataframe, target='fileserver')
            self.assertTrue(test_file.exists())
        with self.subTest('load csv'):
            df = self.pft.csv_load('saved.csv', index_col=0)
            from pandas.testing import assert_frame_equal
            assert_frame_equal(df, self.pandas_dataframe)

    def test_pandas_json(self):
        with self.subTest('to json'):
            test_file = self.fileserver_userdir / 'saved.json'
            self.pft.pandas_to_json('saved.json', self.pandas_dataframe, target='fileserver')
            self.assertTrue(test_file.exists())
        with self.subTest('read json'):
            df = self.pft.pandas_read_json('saved.json')
            from pandas.testing import assert_frame_equal
            assert_frame_equal(df, self.pandas_dataframe)

    def test_pandas_hdf(self):
        ext = 'h5'
        filename = 'saved.' + ext
        with self.subTest('to ' + ext):
            test_file = self.fileserver_userdir / filename
            self.pft.pandas_to_hdf(filename, self.pandas_dataframe, target='fileserver')
            self.assertTrue(test_file.exists())
        with self.subTest('read ' + ext):
            df = self.pft.pandas_read_hdf(filename)
            from pandas.testing import assert_frame_equal
            assert_frame_equal(df, self.pandas_dataframe)

    def test_pandas_excel(self):
        ext = 'xls'
        filename = 'saved.' + ext
        with self.subTest('to ' + ext):
            test_file = self.fileserver_userdir / filename
            self.pft.pandas_to_excel(filename, self.pandas_dataframe, target='fileserver')
            self.assertTrue(test_file.exists())
        with self.subTest('read ' + ext):
            df = self.pft.pandas_read_excel(filename, index_col=0)
            from pandas.testing import assert_frame_equal
            assert_frame_equal(df, self.pandas_dataframe)

    def test_numpy_txt(self):
        ext = 'csv'
        filename = 'saved.' + ext
        with self.subTest('to ' + ext):
            test_file = self.fileserver_userdir / filename
            self.pft.numpy_savetxt(filename, self.testdata, target='fileserver')
            self.assertTrue(test_file.exists())
        with self.subTest('read ' + ext):
            loaded = self.pft.numpy_loadtxt(filename)
            self.assertTrue(np.array_equal(self.testdata, loaded))

    def test_numpy(self):
        ext = 'npy'
        filename = 'saved.' + ext
        with self.subTest('to ' + ext):
            test_file = self.fileserver_userdir / filename
            self.pft.numpy_save(filename, self.testdata, target='fileserver')
            self.assertTrue(test_file.exists())
        with self.subTest('read ' + ext):
            loaded = self.pft.numpy_load(filename)
            self.assertTrue(np.array_equal(self.testdata, loaded))

    def test_numpy_compressed(self):
        ext = 'npz'
        filename = 'saved.' + ext
        with self.subTest('to ' + ext):
            test_file = self.fileserver_userdir / filename
            self.pft.numpy_savez_compressed(filename, self.testdata, target='fileserver')
            self.assertTrue(test_file.exists())
        with self.subTest('read ' + ext):
            loaded = self.pft.numpy_load(filename)
            self.assertTrue(np.array_equal(self.testdata, loaded['arr_0']))

    def test_image(self):
        ext = 'tif'
        filename = 'saved.' + ext
        with self.subTest('to ' + ext):
            test_file = self.fileserver_userdir / filename
            self.pft.imsave(filename, self.testdata, target='fileserver')
            self.assertTrue(test_file.exists())
        with self.subTest('read ' + ext):
            loaded = self.pft.imread(filename)
            self.assertTrue(np.array_equal(self.testdata, loaded))


if __name__ == '__main__':
    unittest.main()
