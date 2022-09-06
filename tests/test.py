import unittest
from pathlib import Path
import numpy as np
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
        self.userdir = self.fileserver / 'userdir'
        self.userdir.mkdir()
        self.testdata = np.random.randn(64, 64)
        np.save(self.userdir / 'testdata.npy', self.testdata, allow_pickle=False)
        imsave(self.userdir / 'testimage.tif', self.testdata)
        return super().setUp()

    def tearDown(self) -> None:
        self.mock_cluster.tempdir.cleanup()
        return super().tearDown()

    def test_ensure_project_dir_exists(self):
        with self.mock_cluster as mock_cluster:
            target = self.project / 'userdir'
            self.assertFalse(target.exists())
            pft = biapol_taurus.ProjectFileTransfer(
                source_fileserver_dir=str(self.userdir),
                target_project_space_dir=str(target),
                datamover_path=mock_cluster.bin_path,
                workspace_exe_path=mock_cluster.bin_path)
            self.assertTrue(target.exists())

    def test_sync_with_fileserver(self):
        with self.mock_cluster as mock_cluster:
            pft = biapol_taurus.ProjectFileTransfer(
                source_fileserver_dir=str(self.userdir),
                target_project_space_dir=str(self.project / 'userdir'),
                datamover_path=mock_cluster.bin_path,
                workspace_exe_path=mock_cluster.bin_path)
            pft.sync_with_fileserver()
            numpy_data = np.load(self.project / 'userdir' / 'testdata.npy')
            self.assertTrue(np.array_equal(self.testdata, numpy_data))


if __name__ == '__main__':
    unittest.main()
