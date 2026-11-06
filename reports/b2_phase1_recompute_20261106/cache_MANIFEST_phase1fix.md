# Studyforrest sample manifest

- Source: OpenNeuro ds000113 (Studyforrest, Hanke et al. 2014) via S3 REST
- Frames: 9000 at 10 Hz

## Per-channel provenance

- `force.pt`  : REAL ds000113 ses-forrestgump physio, 4 cols [trigger,resp,cardiac,SpO2] @ 100 Hz (Hanke et al. 2014 p. 11; sidecar used if present) -> [cardiac,d,d2,resp,d,d2] @ 10 Hz
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

## SHA256

- `audio.pt`: `37e12a14dcb39950b3442e243141ca8b05f74eb1130a81c2756b66ac6908f049`
- `vision.pt`: `924ee146c9669250a128387ed85fad047cbd91a183a63f4c58dbc529a4cef9d5`
- `tactile.pt`: `a553c11be9ee76eb4340ad5f9bba959449d34b000e0e2520a8f9f53654fb1921`
- `gravity.pt`: `38b6bc7ed4c125c605010a28b20680b24925af403d5665832d9c7b7e8f508964`
- `force.pt`: `d96ad636e1d0d7c21c33d79c47556c43936bf259ce69ed77b51366c62ad5b9e9`
- `labels.pt`: `5de7e69b6a7b96406fbec839b83b073a6924fde34702ebe00ee154edaf67c3b2`
