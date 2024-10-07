import os
import requests
import zipfile

from types import SimpleNamespace
from dswx_sar.dswx_runconfig import RunConfig
from dswx_sar import generate_log

from dswx_sar import dswx_s1

DATASET_URL = ('https://github.com/opera-adt/DSWX-SAR/releases/download/'
               'sample_dataset/dswx_s1_test_data.zip')


ZIP_FILENAME =  os.path.basename(DATASET_URL)
def _download_test_dataset_from_github():
    with requests.get(DATASET_URL, stream=True) as rq:
        rq.raise_for_status()
        with open(ZIP_FILENAME, 'wb') as fout:
            for chunk in rq.iter_content(chunk_size=1048576): 
                fout.write(chunk)

    with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
        zip_ref.extractall('./')

print('')

def test_workflow():
    test_data_directory = 'data'
    if not os.path.isdir('data'):
        os.makedirs('data', exist_ok=True)

    os.chdir(test_data_directory)

    if not os.path.isdir('input_dir'):
        _download_test_dataset_from_github()

    runconfig_path = 'dswx_s1_test.yaml'
    log_path = 'dswx_s1.log'

    args = SimpleNamespace(input_yaml=runconfig_path,
                           debug_mode=False,
                           log_file=log_path)

    cfg = RunConfig.load_from_yaml(runconfig_path, 'dswx_s1', args)
    generate_log.configure_log_file(cfg.groups.log_file)

    dswx_s1.dswx_s1_workflow(cfg)

