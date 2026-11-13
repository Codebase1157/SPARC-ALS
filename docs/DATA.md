# Data access

SPARC-ALS v1.0.0 is evaluated on three public corpora that do **not**
require a data-use agreement. Each has its own license and must be
obtained separately from its upstream provider.

## VOC-ALS

* **Reference.** Dubbioso et al., "Voice signals database of ALS
  patients with different dysarthria severity and healthy controls",
  *Scientific Data* 11, 800 (2024).
* **Location.** `https://doi.org/10.7303/syn53009474` (Synapse).
* **Size.** 1,224 smartphone recordings at 8 kHz from 153 Italian
  speakers (102 ALS, 51 HC). ALSFRS-R speech item 1 grades the patients
  as severe (5), moderate (13), mild (31) or non-dysarthric (53).
* **Tokens kept.** `/a, e, i, o, u, pa, ta, ka/`.
* **Place at:** `data/raw/voc_als/<speaker_id>/<token>.wav`.

## Minsk2020

* **Reference.** Vashkevich & Rushkevich, "Classification of ALS
  patients based on acoustic analysis of sustained vowel phonations",
  *Biomedical Signal Processing and Control* 65, 102350 (2021).
* **Location.** `https://github.com/Mak-Sim/Minsk2020_ALS_database`.
* **Size.** Sustained /a/ and /i/ from 31 ALS speakers and 33 controls
  (44.1 kHz; SPARC-ALS resamples to 16 kHz and then band-limits to 4
  kHz for the analysis pipeline).
* **Place at:** `data/raw/minsk2020/<speaker_id>/<token>.wav`.

## TORGO

* **Reference.** Rudzicz et al., "The TORGO database of acoustic and
  articulatory speech from speakers with dysarthria", *LREC* 46,
  523–541 (2012).
* **Location.** `https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html`.
* **Size.** 8 dysarthric speakers (7 with cerebral palsy, 1 with ALS)
  and 7 controls. 150 array-microphone words and sentences per speaker
  are drawn with a fixed seed (42).
* **Caveat.** TORGO results concern dysarthria of mixed etiology; they
  are never reported as ALS detection in the paper.
* **Place at:** `data/raw/torgo/<speaker_id>/session<n>/wav_arrayMic/*.wav`.

## Reproducibility note

The three corpora are not redistributed with this release. Once placed
under `data/raw/`, running `bash run.sh` reproduces every preprocessing
artifact, feature cache, prediction CSV and metric table deterministically
from the committed splits and seeds.
