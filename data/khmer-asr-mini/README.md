# Khmer ASR mini dataset

Short Khmer speech clips with transcripts, 16 kHz mono FLAC, packed for fine-tuning Whisper.
Columns: `id`, `source`, `text`, `duration` (seconds), `audio` (FLAC bytes).

| Split | Clips | Hours |
| --- | --- | --- |
| train | 20,758 | 15.99 |
| validation | 1,150 | 0.93 |
| test | 1,179 | 0.92 |
| ddd_test | 2,034 | 4.66 |

## Sources

| Source | Clips | Hours | License | Origin |
| --- | --- | --- | --- | --- |
| openslr42 | 2,906 | 3.97 | CC-BY-SA-4.0 | OpenSLR 42 (https://openslr.org/42/), 1 male speaker, word-segmented text |
| km_speech_corpus | 14,943 | 10.41 | CC-BY-4.0 | https://huggingface.co/datasets/seanghay/km-speech-corpus |
| kheng_info | 3,097 | 1.46 | research use only | https://huggingface.co/datasets/seanghay/khmer_kheng_info_speech (single words, kheng.info) |
| mpwt | 2,058 | 1.9 | research use only | https://huggingface.co/datasets/seanghay/khmer_mpwt_speech (Ministry of Public Works and Transport app) |
| rfi_manual | 83 | 0.11 | own collection (RFI Khmer audio) | Clips cut and transcribed by hand from RFI Khmer news |

`ddd_test` is a 2,034-clip sample of the Digital Divide Data Khmer ASR Cultural Dataset (https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset, CC-BY-SA-4.0): two speakers and paragraphs held out, used as an out-of-domain test set only.

Some sources are licensed for research use only, so keep this dataset private.
