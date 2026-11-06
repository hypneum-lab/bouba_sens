# Studyforrest sample manifest

- Source: OpenNeuro ds000113 (Studyforrest, Hanke et al. 2014) via S3 REST
- Frames: 9000 at 10 Hz

## Per-channel provenance

- `force.pt`  : REAL ds000113 cardiac(pulse-ox)+respiration physio, 500 Hz [cardiac,d,d2,resp,d,d2]
- `gravity.pt`: REAL ds000113 derivatives/motion *_bold_moco.txt rotation regressors
- `tactile.pt`: REAL ds000113 scenes.csv (scene/DAY-NIGHT/INT-EXT) one-hot -> 32-D proj
- `labels.pt` : REAL ds000113 scenes.csv scene index, quantised to 4
- `audio.pt`  : DERIVED PROXY (NOT soundtrack; copyrighted/absent from ds000113) from REAL german_audio_description.csv narration timing -> 128-D
- `vision.pt` : DERIVED PROXY (NOT movie; copyrighted/absent from ds000113) from REAL scenes.csv cut structure -> (16,16)
- 
- ### Real S3 object URLs
- - `scenes.csv`: https://s3.amazonaws.com/openneuro.org/ds000113/stimuli/annotations/scenes.csv
- - `german_audio_description.csv`: https://s3.amazonaws.com/openneuro.org/ds000113/stimuli/annotations/german_audio_description.csv
- - `physio_run-01.tsv.gz`: https://s3.amazonaws.com/openneuro.org/ds000113/sub-01/ses-forrestgump/func/sub-01_ses-forrestgump_task-forrestgump_run-01_physio.tsv.gz
- - `moco_run-01.txt`: https://s3.amazonaws.com/openneuro.org/ds000113/derivatives/motion/sub-01/ses-forrestgump/func/sub-01_ses-forrestgump_acq-dico_run-01_bold_moco.txt
- - `physio_run-02.tsv.gz`: https://s3.amazonaws.com/openneuro.org/ds000113/sub-01/ses-forrestgump/func/sub-01_ses-forrestgump_task-forrestgump_run-02_physio.tsv.gz
- - `moco_run-02.txt`: https://s3.amazonaws.com/openneuro.org/ds000113/derivatives/motion/sub-01/ses-forrestgump/func/sub-01_ses-forrestgump_acq-dico_run-02_bold_moco.txt
- - `physio_run-03.tsv.gz`: https://s3.amazonaws.com/openneuro.org/ds000113/sub-01/ses-forrestgump/func/sub-01_ses-forrestgump_task-forrestgump_run-03_physio.tsv.gz
- - `moco_run-03.txt`: https://s3.amazonaws.com/openneuro.org/ds000113/derivatives/motion/sub-01/ses-forrestgump/func/sub-01_ses-forrestgump_acq-dico_run-03_bold_moco.txt
- - `physio_run-04.tsv.gz`: https://s3.amazonaws.com/openneuro.org/ds000113/sub-01/ses-forrestgump/func/sub-01_ses-forrestgump_task-forrestgump_run-04_physio.tsv.gz
- - `moco_run-04.txt`: https://s3.amazonaws.com/openneuro.org/ds000113/derivatives/motion/sub-01/ses-forrestgump/func/sub-01_ses-forrestgump_acq-dico_run-04_bold_moco.txt
- - `physio_run-05.tsv.gz`: https://s3.amazonaws.com/openneuro.org/ds000113/sub-01/ses-forrestgump/func/sub-01_ses-forrestgump_task-forrestgump_run-05_physio.tsv.gz
- - `moco_run-05.txt`: https://s3.amazonaws.com/openneuro.org/ds000113/derivatives/motion/sub-01/ses-forrestgump/func/sub-01_ses-forrestgump_acq-dico_run-05_bold_moco.txt

## SHA256

- `audio.pt`: `37e12a14dcb39950b3442e243141ca8b05f74eb1130a81c2756b66ac6908f049`
- `vision.pt`: `924ee146c9669250a128387ed85fad047cbd91a183a63f4c58dbc529a4cef9d5`
- `tactile.pt`: `a553c11be9ee76eb4340ad5f9bba959449d34b000e0e2520a8f9f53654fb1921`
- `gravity.pt`: `38b6bc7ed4c125c605010a28b20680b24925af403d5665832d9c7b7e8f508964`
- `force.pt`: `890b90aea31adb93f160609a74ce5a2da50942fa82ff8802a7645f9ef15cd773`
- `labels.pt`: `5de7e69b6a7b96406fbec839b83b073a6924fde34702ebe00ee154edaf67c3b2`
