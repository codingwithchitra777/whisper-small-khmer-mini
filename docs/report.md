# Khmer Speech-to-Subtitles

**Fine-tuning Whisper-small to generate timestamped Khmer subtitles for YouTube videos**

**Final Project Report**

| | |
| --- | --- |
| University | Royal University of Phnom Penh |
| Programme | Master of Science in Data Science and Engineering |
| Subject | Introduction to Large Language Models |
| Lecturer | Dr. Chen Sovann |
| Students | Sem Chitra, Sar Sakal, Earn Pisey, Sam Reaksmey |
| Class | MDSE Cohort 3, October 2026 |
| Code | [github.com/codingwithchitra777/whisper-small-khmer-mini](https://github.com/codingwithchitra777/whisper-small-khmer-mini) |
| Model | [huggingface.co/chitra168/whisper-small-khmer-mini](https://huggingface.co/chitra168/whisper-small-khmer-mini) |

## Abstract

Khmer is spoken by about 17 million people, yet almost no Khmer video on YouTube has subtitles, and
general-purpose speech recognition handles Khmer poorly. This project builds a working web application that
takes a YouTube link and returns timestamped Khmer subtitles that can be exported as SRT. At its core is
OpenAI's Whisper-small, a 241-million-parameter Transformer encoder–decoder, fully fine-tuned on about 16 hours
of Khmer speech collected from five sources, including clips we transcribed ourselves. We customise the
training in three ways: timestamp-aware labels so the model keeps producing subtitle timings, checkpoint
selection by character error rate without spaces (a metric suited to Khmer, which has no consistent word
spacing), and leak-free data splits hashed by sentence. We compare the fine-tuned model with the zero-shot
Whisper-small baseline and with an ablation trained on a single-speaker dataset, on an in-domain test set and
an out-of-domain set of unseen speakers. On unseen speakers, fine-tuning lowers the character error rate
without spaces from 147.6% (zero-shot) to 18.9%, and training on all five sources halves the error of the
single-speaker ablation (39.4% → 18.9%).

## 1. Introduction

### 1.1 The problem

A growing share of news, education and entertainment in Cambodia is published as online video, most of it on
YouTube and Facebook. Nearly all of it is published without subtitles. Writing Khmer subtitles by hand is slow:
a transcriber has to listen, type in a complex script, and align each line to the video. Automatic tools that
exist for English and other major languages either do not support Khmer or produce transcripts too inaccurate
to publish.

Missing subtitles exclude several groups of people: viewers who are deaf or hard of hearing, people watching
without sound, and learners of Khmer who benefit from reading along. They also make video content
unsearchable, since search engines index text rather than speech.

### 1.2 Target users

- Khmer content creators and small media outlets who want subtitles but cannot pay for manual transcription.
- Deaf and hard-of-hearing viewers in Cambodia and the Khmer diaspora.
- Teachers and students, who can turn recorded lessons into readable text.
- Learners of Khmer who want to read along with native speech.

### 1.3 Our solution

We built a web application where a user pastes a YouTube link and receives Khmer subtitles with start and end
times, shown next to the playing video and downloadable as an SRT or plain-text file. The speech recognition
model is Whisper-small, fine-tuned by us on Khmer speech. This report describes the data (Section 3), the model
and how we customised its fine-tuning (Section 4), the full system (Section 5), and how the model compares with
a baseline and an ablation (Section 6).

## 2. Existing Solutions and What Is Missing

| Existing option | What it offers | What is missing for Khmer subtitles |
| --- | --- | --- |
| Manual transcription | High accuracy | Slow and expensive; hours of work per hour of video |
| YouTube automatic captions | Free, built into the platform; "Central Khmer" is on YouTube's list of automatic-caption languages [8] | Only for videos on YouTube and only if captions are generated for the video; YouTube publishes no accuracy figures and itself warns that "the quality of the captions may vary" [8] |
| Commercial speech APIs (Google Cloud Speech-to-Text) | Khmer (`km-KH`) is supported, including the Chirp 2 and Chirp 3 models with automatic punctuation [9] | Paid per minute of audio; a general-purpose model with no Khmer accuracy figures published; no subtitle workflow (the user must download audio, call the API and build subtitle files themselves) |
| OpenAI Whisper, used as is (zero-shot) | Free, open-source, supports Khmer | Khmer was a tiny part of its training data, so accuracy is low (measured in Section 6 as our baseline) |
| Community Khmer ASR models on Hugging Face (Table 1b) | Free, open models fine-tuned on public Khmer data | Released as models only: none provides an end-user tool from a video link to subtitle files |

*Table 1. Existing options for getting Khmer subtitles.*

Table 1b lists the most-downloaded Khmer speech recognition models on the Hugging Face Hub (October 2026) with
the results their authors report. Each is measured on a different test set, so the numbers cannot be compared
directly with each other or with ours.

| Model | Base model | Training data | Reported result |
| --- | --- | --- | --- |
| [gagan3012/wav2vec2-xlsr-khmer](https://huggingface.co/gagan3012/wav2vec2-xlsr-khmer) | wav2vec2-large-XLSR-53 | Common Voice, OpenSLR 42 | WER 24.96% on an OpenSLR 42 split |
| [seanghay/whisper-small-khmer-v2](https://huggingface.co/seanghay/whisper-small-khmer-v2) | Whisper-small (same as ours) | OpenSLR 42, Google FLEURS, km-speech-corpus | WER 61.65% on Google FLEURS |
| [seanghay/Qwen3-ASR-0.6B-Khmer](https://huggingface.co/seanghay/Qwen3-ASR-0.6B-Khmer) | Qwen3-ASR-0.6B | ~700 h of the Digital Divide Data Khmer dataset | CER (space-insensitive) 1.96% in-domain, 7.91% out-of-domain |

*Table 1b. Community Khmer speech recognition models (reported by their authors, not re-measured by us).*

The strongest of these, Qwen3-ASR-0.6B-Khmer, was trained on roughly 40 times more audio than our model (about
700 hours against our 16), and on the same Digital Divide Data dataset our out-of-domain test set comes from, so
its scores show what more data can achieve rather than a like-for-like comparison. None of these models comes
with a way for a non-programmer to subtitle a video.

The gap is a free tool that goes all the way from a video link to subtitle timings, backed by a model trained
specifically for Khmer. Our project addresses both parts: a Khmer-tuned model and a complete subtitle workflow
around it.

## 3. Dataset

No single public Khmer speech dataset is large and varied enough on its own, so we combined five sources into
one dataset, `khmer-asr-mini`. Table 2 lists them; their licenses are recorded in the code (`src/config.py`).

| Source | Clips | Hours | Content | License |
| --- | --- | --- | --- | --- |
| km-speech-corpus (seanghay) | 14,943 | 10.41 | Read sentences, many speakers | CC-BY-4.0 |
| OpenSLR 42 | 2,906 | 3.97 | Read news sentences, one male speaker | CC-BY-SA-4.0 |
| khmer_kheng_info_speech | 3,097 | 1.46 | Single words (kheng.info dictionary) | Research use only |
| khmer_mpwt_speech | 2,058 | 1.90 | Traffic-rule questions (Ministry of Public Works and Transport) | Research use only |
| RFI Khmer (our own) | 83 | 0.11 | Radio news clips we cut and transcribed by hand | Own collection |
| **Total** | **23,087** | **17.85** | | |

*Table 2. Sources combined into the khmer-asr-mini dataset.*

### 3.1 Data preparation

The script `src/build_dataset.py` turns the raw collections into one consistent dataset:

- **Audio format.** The sources were recorded at 16, 22.05, 44.1 and 48 kHz. Every clip is converted to 16 kHz
  mono (Whisper's input format) using polyphase resampling, and stored losslessly as FLAC inside Parquet files.
- **Length filtering.** Clips shorter than 0.3 s or longer than 30 s (Whisper's input window) are dropped, as
  are transcripts longer than Whisper's 448-token decoder limit. (No clip in the current sources hit either
  limit.)
- **Text normalisation.** Zero-width spaces, common in Khmer text, are removed and repeated whitespace is
  collapsed.
- **Consistent spacing.** Khmer is written without spaces between words. OpenSLR 42 inserts a space between
  every word, which the other sources do not, so these spaces are removed; otherwise the model would learn two
  conflicting spacing styles.
- **Leak-free splits.** Each sentence is assigned to train, validation or test by hashing its text, so the same
  sentence (even read by different speakers) never appears in two splits. This matters because several sources
  repeat sentences across speakers; a random split would put test sentences into training and inflate the
  scores.
- **Out-of-domain test set.** We add a separate test set from the Digital Divide Data Khmer ASR dataset
  (CC-BY-SA-4.0): two speakers and paragraphs not seen anywhere in training. Any training sentence that also
  appears in this set is removed.

| Split | Clips | Hours | Purpose |
| --- | --- | --- | --- |
| train | 20,758 | 15.99 | Fine-tuning |
| validation | 1,150 | 0.93 | Choosing the best checkpoint during training |
| test | 1,179 | 0.92 | In-domain test: same sources, sentences never seen in training |
| ddd_test | 2,034 | 4.66 | Out-of-domain test: unseen speakers and unseen text |

*Table 3. Dataset splits.*

Because two sources are licensed for research use only, the dataset is kept private and is not
redistributed.

## 4. Model and Fine-tuning Method

### 4.1 Base model

We use `openai/whisper-small`, a Transformer encoder–decoder with 240,582,912 trainable parameters (241 M), well
above the project's 10-million minimum. The encoder turns an 80-channel log-Mel spectrogram of up to 30 seconds
of audio into a sequence of hidden states; the decoder generates the transcript token by token while attending
to them. Whisper was pre-trained on 680,000 hours of multilingual speech, but only a very small fraction of it
was Khmer, which is why its zero-shot Khmer output is poor. We chose the small size because it can be fully
fine-tuned on a single consumer GPU in under an hour and runs acceptably fast for a demo.

![Whisper-small encoder–decoder architecture: the encoder turns an audio chunk, as an 80-channel log-Mel
spectrogram, into hidden states through 12 Transformer blocks; the decoder, prompted with Khmer transcription
tokens, attends to them through cross-attention and writes Khmer text with timestamp tokens one token at a
time.](figures/whisper-architecture.png)

*Figure 1. Whisper-small as used in this project, after Radford et al. (2023) [1]. All 241 million weights are
fine-tuned; the highlighted parts are what our fine-tuning sets for Khmer: the task prompt (language `km`,
task `transcribe`, timestamps on) and timestamped Khmer output, capped at 448 tokens.*

### 4.2 How we customised the fine-tuning

We fine-tune all 241 million parameters (no frozen layers or adapters), using the Hugging Face
`Seq2SeqTrainer`. On top of standard Whisper fine-tuning, we made the following choices, implemented in
`src/train.py` and `src/speech.py`:

- **Timestamp-aware labels.** Standard fine-tuning trains only on plain transcripts, and the model gradually
  forgets how to predict timestamps, which a subtitle tool depends on. For a random 50% of training clips we
  instead write the target as `<|0.00|> transcript <|end time|>`, with the end time rounded to Whisper's 0.02 s
  steps. The model therefore keeps producing time tokens, so the web app can still request timestamped output.
- **Language and task fixed to Khmer transcription.** The language (Khmer) and task (transcribe) are set in the
  model's generation configuration, so the model never drifts into translating to English or into a wrong
  language.
- **Model selection by CER without spaces.** During training the model regularly transcribes up to 400
  validation clips, and the checkpoint with the lowest character error rate with all spaces removed is kept.
  Word error rate is unreliable for Khmer because word boundaries are not written consistently, so a correct
  transcript with different spacing would be counted as many word errors.
- **Training on a single affordable GPU.** bf16 mixed precision and gradient checkpointing keep memory low, and
  gradient accumulation (16 clips × 2 steps) gives an effective batch of 32 on one GPU. The same code also runs
  on Kaggle's free T4 GPUs (fp16, two GPUs with `torchrun`), with a callback that checkpoints and stops before
  Kaggle's 12-hour session limit so the next session can resume.

| Setting | Value |
| --- | --- |
| Optimiser | AdamW (Trainer default) |
| Learning rate | 1 × 10⁻⁵, 200 warm-up steps, then linear decay |
| Epochs | 5 (3,245 optimiser steps for the main model, 410 for the ablation) |
| Effective batch size | 32 (16 per step × 2 gradient-accumulation steps, 1 GPU) |
| Precision | bf16 mixed precision |
| Memory | Gradient checkpointing |
| Timestamp label share | 50% |
| Evaluation / checkpoint interval | Every 500 steps (main), every 100 steps (ablation); up to 400 validation clips |
| Best-model metric | CER without spaces (lower is better) |
| Hardware | RunPod, 1 × NVIDIA RTX 4090 (24 GB), $0.74/hour |
| Measured training speed | ~1.45 optimiser steps/s (smoke test) |
| Training time | 44 min 24 s for the main model (2,664 s, 3,245 steps); 7 min 35 s for the ablation (455 s, 410 steps); evaluations included |
| Total GPU cost | [about $X, from the RunPod billing page] |

*Table 4. Fine-tuning hyperparameters.*

## 5. System Architecture

The application has two parts: a Next.js web front end and a FastAPI back end that hosts the fine-tuned model.
The front end forwards requests to the API, which runs on the same machine.

![System architecture: the viewer's browser sends a YouTube URL to the Next.js web app, which proxies it to the
FastAPI back end; the back end downloads the audio, splits it at pauses, transcribes it with the fine-tuned
Whisper-small model and returns timestamped segments.](figures/system-architecture.png)

*Figure 2. System architecture. The fine-tuned model (highlighted) is trained offline on RunPod and loaded by
the back end at start-up.*

When a user submits a YouTube link, the back end (`api/main.py`) runs these steps:

1. **Download.** yt-dlp and ffmpeg download the video's audio track and convert it to 16 kHz mono WAV.
2. **Split at pauses.** Khmer is expensive in Whisper's vocabulary: about 400 tokens for 15 seconds of read
   speech, and more for fast news delivery, against a decoder limit of 448 tokens. A standard 30-second window
   would therefore be cut off mid-sentence. We split the audio into chunks of at most 10 seconds, each ending at
   a pause (at least ~200 ms of low energy) so that words are not split in half. We first used 15-second
   chunks, but on a real news video the endings of fast sentences still came out garbled; 10-second chunks
   removed that and gave shorter, easier-to-read subtitle lines. Silent chunks are skipped, which prevents the
   model from inventing text for silence.
3. **Transcribe.** The fine-tuned model transcribes the chunks eight at a time, with timestamps enabled, in
   half precision on the GPU. Batching matters on a laptop GPU, where the fixed cost of each decoding step
   dominates: it made transcription about 2.5 times faster.
4. **Assemble.** Each segment's timestamps are shifted by its chunk's position in the video and returned as a
   list of (start, end, text) segments.

The front end (`webapp/`) shows the embedded YouTube video next to a scrollable subtitle timeline and lets the
user export the subtitles as an SRT file (usable in YouTube Studio and video editors) or as plain text.

> [Add 2–3 screenshots of the web app here: the input page, the results with the video and subtitle timeline,
> and an exported SRT file.]

## 6. Experiments and Results

### 6.1 Metrics

- **WER (word error rate):** the standard ASR metric, reported for completeness. For Khmer it is unreliable
  because spacing is inconsistent.
- **CER (character error rate):** the share of characters substituted, deleted or inserted.
- **CER without spaces:** CER after removing all spaces from both the prediction and the reference. This is our
  main metric, because it scores what was said and ignores how the words were spaced.

### 6.2 Models compared

| Model | Training data | Role |
| --- | --- | --- |
| Whisper-small, zero-shot | None (original OpenAI model) | Baseline |
| whisper-small-khmer-mini | All five sources, 16 h (20,758 clips) | Main model |
| whisper-small-khmer-openslr-only | OpenSLR 42 only, ~3.6 h (2,606 clips), one speaker | Ablation |

*Table 5. Models in the comparison.*

The ablation uses exactly the same training recipe but only the single-speaker OpenSLR 42 data. It tests
whether more data from more speakers, which is most of our data-collection effort, actually makes the model
better, particularly on the out-of-domain test set with speakers the model has never heard.

### 6.3 Results

**Training progress.** Table 6 shows the main model's score on 400 held-out validation clips during
training. CER without spaces fell from 46.8% after the first 500 steps to 22.6% at step 3,000, the best
checkpoint, which is the one kept as the final model. The gains shrank with every evaluation, and validation
loss stopped improving after step 2,500 (0.243 → 0.246) while CER kept falling slightly, so five epochs was
about the right amount of training for this data. WER stayed near 86%: a Khmer "word" between spaces is
usually a whole phrase, so a single wrong character marks the entire phrase as an error.

| Step | Epoch | Validation loss | WER | CER | CER no space |
| --- | --- | --- | --- | --- | --- |
| 500 | 0.77 | 0.460 | 97.3% | 47.2% | 46.8% |
| 1,000 | 1.54 | 0.304 | 91.6% | 35.1% | 34.5% |
| 1,500 | 2.31 | 0.262 | 86.7% | 27.8% | 27.1% |
| 2,000 | 3.08 | 0.257 | 87.5% | 25.1% | 24.4% |
| 2,500 | 3.85 | 0.243 | 86.7% | 24.2% | 23.5% |
| **3,000** | **4.62** | **0.246** | **86.1%** | **23.4%** | **22.6%** |

*Table 6. Main model on the validation set during training (best checkpoint in bold). Training loss fell from
3.15 at the start to about 0.10 at the end of epoch 5 (0.35 averaged over the whole run).*

The ablation model, trained only on OpenSLR 42, was evaluated every 100 steps on the 165 OpenSLR validation
clips (Table 7). Its best score, 27.1% CER without spaces, is worse than the main model's 22.6% even though
its validation set contains only the speaker and sentence style it was trained on, while the main model's
validation set mixes all five sources. Its curve was still falling steeply when training ended (34.4% →
27.1% over the last 100 steps): with only 410 optimiser steps, 200 of them in learning-rate warm-up, the
ablation is somewhat undertrained, so part of the gap comes from fewer training steps and not only from less
data. Its WER stays near 100% because OpenSLR's spaces were removed, so each reference sentence counts as a
single "word".

| Step | Epoch | Validation loss | CER no space |
| --- | --- | --- | --- |
| 100 | 1.22 | 1.534 | 131.7% |
| 200 | 2.44 | 1.134 | 79.1% |
| 300 | 3.66 | 0.427 | 34.4% |
| **400** | **4.88** | **0.320** | **27.1%** |

*Table 7. Ablation model (OpenSLR 42 only) on its validation set during training (best checkpoint in bold).
A CER above 100% means the model inserted more characters than the reference has, typical of an untrained
model that rambles.*

**Test results.** All three models transcribed both test sets with the same decoding settings (greedy,
Khmer, up to 444 new tokens). Table 8 gives the results; the main model is best on every metric and both sets.
In plain terms, the main model gets roughly four characters in five right: a CER without spaces of 18.9% on
speakers it has never heard corresponds to about 81% character accuracy (approximate, because CER also counts
inserted characters), and 24.9% on held-out sentences to about 75%.

| Model | Test set | Clips | WER | CER | CER no space |
| --- | --- | --- | --- | --- | --- |
| Baseline (zero-shot) | test | 1,179 | 3,381% | 564.0% | 383.1% |
| Baseline (zero-shot) | ddd_test | 2,034 | 2,472% | 211.9% | 147.6% |
| Ablation (OpenSLR only) | test | 1,179 | 98.8% | 59.2% | 59.1% |
| Ablation (OpenSLR only) | ddd_test | 2,034 | 100.0% | 40.1% | 39.4% |
| **Main (all sources)** | **test** | **1,179** | **79.7%** | **25.6%** | **24.9%** |
| **Main (all sources)** | **ddd_test** | **2,034** | **102.3%** | **20.0%** | **18.9%** |

*Table 8. Results on the in-domain (test) and out-of-domain (ddd_test) sets. Error rates can exceed 100%
because inserted words or characters count as errors on top of the reference length.*

**Results by source.** Table 9 breaks the in-domain `test` result down by data source.

| Source (`test`) | Clips | Baseline | Ablation | Main |
| --- | --- | --- | --- | --- |
| openslr42 | 135 | 350.5% | 27.2% | **12.0%** |
| mpwt | 120 | 355.1% | 50.6% | **14.5%** |
| kheng_info | 132 | 1,478.3% | 40.1% | **15.3%** |
| rfi_manual | 4 | 229.6% | 49.8% | **18.7%** |
| km_speech_corpus | 788 | 365.0% | 68.3% | **30.0%** |

*Table 9. CER without spaces on `test`, per source. `rfi_manual` has only 4 test clips, so its score is not
reliable on its own.*

### 6.4 Discussion

- **Fine-tuning turns an unusable model into a usable one.** Zero-shot Whisper-small does not produce Khmer
  transcripts at all in practice: its CER without spaces is 383.1% on `test` and 147.6% on `ddd_test`, meaning it
  outputs far more wrong characters than the reference contains (typical of a model that rambles or repeats
  itself in a language it barely knows). After fine-tuning, the main model gets about four out of five
  characters right: 24.9% on `test` and 18.9% on `ddd_test`.
- **More, more varied data clearly helps (the ablation).** Trained with the same recipe on OpenSLR 42 alone,
  the ablation model scores 59.1% on `test` and 39.4% on `ddd_test`. Adding the other four sources cuts the
  error by 58% (relative) on `test` and by 52% on `ddd_test`. The `ddd_test` result matters most: its two
  speakers and its sentences appear nowhere in either model's training data, so it measures how well each model
  handles new voices. The single-speaker model generalises much worse, which is what we expected from training
  on one voice. As noted in Section 6.3, the ablation also had fewer training steps (410 versus 3,245), so part
  of the gap reflects less training, not only less data.
- **The main model wins on every source, even the ablation's own.** On OpenSLR 42 test sentences, the source
  the ablation was trained on, the main model scores 12.0% against the ablation's 27.2% (Table 9). Seeing
  many voices and sentence styles made it better even on the single speaker.
- **km-speech-corpus is the hardest source, and it explains why `test` scores worse than `ddd_test`.** We
  expected the one-word kheng.info clips to be hardest, but the main model handles them well (15.3%). The
  hardest is km-speech-corpus (30.0%): its clips are short fragments (about 2.5 s on average) that often start
  or end in the middle of a sentence, for example ដែលនោះទើបជាការគោរព… ("which then is the respect…"), so the
  model has little context to resolve similar-sounding words. It also makes up 67% of `test` (788 of 1,179
  clips), which pulls the overall `test` score up to 24.9%; on every other source the main model scores 12–19%.
  `ddd_test`, by contrast, consists of complete, clearly read sentences, so it ends up easier (18.9%) despite
  the unseen speakers.
- **WER is not a useful measure for Khmer.** The main model's WER is 79.7% on `test` and above 100% on
  `ddd_test`, even though its character error is around 20%. Khmer has no consistent word spacing, so a
  "word" between spaces is often a whole phrase, and any spacing difference or a single wrong character makes the
  whole phrase an error. This is why CER without spaces is our main metric.
- **Typical errors are near-misses in spelling.** The main model's mistakes are mostly similar-sounding
  spellings rather than unrelated words (examples below, from the evaluation reports). The zero-shot baseline,
  in contrast, often falls into a loop: on the first `ddd_test` sentence it output the single letter ឍ followed by
  a space, over and over, until it hit the 444-token limit, which is why its error rates exceed 100%.

| Test set | Reference | Main model output | Error |
| --- | --- | --- | --- |
| test (OpenSLR) | សិស្សវិទ្យាល័យទួលទំពូង**ដេញ**វាយគ្នាពេញសាលា | សិស្សវិទ្យាល័យទួលទំពូង**ដិញ**វាយគ្នាពេញសាលា | one vowel |
| test (km-speech-corpus) | ស្គាល់អត្តសញ្ញាណនៃ**ជាតិសាសន៏**របស់ខ្លួន | ស្គាល់អត្តសញ្ញាណនៃ**ជាតិសាស្ត្រ**របស់ខ្លួន | similar-sounding word (ethnicity → science) |
| test (MPWT) | ត្រូវ**ប្រុង**ប្រយ័ត្ន និងឈប់ | ត្រូវ**ព្រោង**ប្រយ័ត្ន និងឈប់ | one syllable |
| ddd_test | ក្នុង**ចំណោម**នោះ កោះរ៉ុង និងកោះរ៉ុងសន្លឹម ដែលជាកោះដ៏**ស្រស់**ស្អាតជាងកោះដទៃ**ទៀត**នៅក្នុងប្រទេសកម្ពុជា។ | ក្នុង**ច្រណោម**នោះ កោះរ៉ុង និង កោះរ៉ុងសន្លឹម ដែលជាកោះដ៏**ស្រា**ស្អាតជាងកោះដទៃ**ទាន់**នៅក្នុងប្រទេសកម្ពុជា | three syllables, spacing; unseen speaker |

*Table 10. Example transcripts from the main model (differences in bold).*

## 7. Demonstration

The complete system runs locally: `start-api.ps1` starts the FastAPI back end with the fine-tuned model on port
8000, and `start-webapp.ps1` starts the Next.js front end on port 3000. In the live demo we paste a Khmer
YouTube video link, show the progress steps, play the video next to the generated subtitles, and export the SRT
file.

For the demo, the web app is shared with a public link through a Cloudflare quick tunnel, while the model runs on
a laptop with an NVIDIA RTX 4060 Laptop GPU (8 GB). Table 11 shows the end-to-end time for two real Khmer news
videos, from submitting the link to receiving all subtitles, including the YouTube download.

| Video | Length | Processing time | Subtitle lines | Speed |
| --- | --- | --- | --- | --- |
| *The One News*: Korean DMZ incident report (`g5w3-ViyvPI`) | 2 min 44 s | 65 s | 23 | about 2.5× faster than real time |
| News commentary on the Thai–Cambodian border (`AzJBYI82L6Q`) | 8 min 34 s | 146 s | 62 | about 3.5× faster than real time |

*Table 11. End-to-end processing time on an RTX 4060 Laptop GPU (batched fp16 transcription, 10-second chunks).
The second video was run through the public link, which is only possible with the background job system,
because Cloudflare closes any single request after 100 seconds.*

Longer videos are relatively faster because the one-off costs (the YouTube download and GPU warm-up) are spread
over more audio. On the first video, the subtitles were readable and followed the story, with errors concentrated in
foreign names (for example, the South Korean president's name) and fast speech, consistent with Section 6.

## 8. Limitations and Future Work

- **Mostly read speech.** Nearly all training data is clearly read sentences or single words, while YouTube
  videos contain spontaneous speech, music and background noise. More real-world, transcribed video audio would
  help most.
- **Coarse timestamps.** Timestamp labels mark only the start and end of each training clip, so timing within a
  long chunk is approximate. Training on sentence-level timings would give tighter subtitles.
- **Small model.** Whisper-small was chosen to train quickly and cheaply; larger Whisper models, or longer
  training, should be more accurate.
- **Licensing.** Two sources are research-use-only, so the model cannot be released for commercial use as is.
- **Speed on CPU.** Transcribing long videos without a GPU is slow; a hosted GPU server or model quantisation
  would make the tool usable by the public.
- **Future features:** subtitle editing in the browser before export, uploading local video files, and Facebook
  video links.

## 9. Conclusion

We built a working web application that turns Khmer speech in YouTube videos into timestamped, exportable
subtitles, addressing the lack of Khmer subtitles on online video. We assembled a 16-hour Khmer speech dataset
from five sources with careful normalisation and leak-free splits, and fully fine-tuned the 241-million-parameter
Whisper-small with timestamp-aware labels and Khmer-appropriate model selection. The zero-shot model cannot
transcribe Khmer (CER without spaces above 100% on both test sets), while our fine-tuned model reaches 24.9%
on held-out sentences and 18.9% on speakers it has never heard. The ablation shows that our data collection
paid off: the same recipe trained on a single speaker reaches only 39.4% on unseen speakers, twice the error of
the main model. Training took 44 minutes on one rented RTX 4090. The code is available at
github.com/codingwithchitra777/whisper-small-khmer-mini.

## References

1. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech
   recognition via large-scale weak supervision. *Proceedings of the 40th International Conference on Machine
   Learning (ICML).*
2. OpenSLR 42: Khmer speech data. https://openslr.org/42/
3. seanghay/km-speech-corpus. Hugging Face. https://huggingface.co/datasets/seanghay/km-speech-corpus
4. seanghay/khmer_kheng_info_speech. Hugging Face. https://huggingface.co/datasets/seanghay/khmer_kheng_info_speech
5. seanghay/khmer_mpwt_speech. Hugging Face. https://huggingface.co/datasets/seanghay/khmer_mpwt_speech
6. Digital Divide Data. Khmer speech dataset. Hugging Face.
   https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset
7. Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of EMNLP
   2020: System Demonstrations.*
8. YouTube Help. Use automatic captioning (list of automatic-caption languages). Accessed October 2026.
   https://support.google.com/youtube/answer/6373554
9. Google Cloud. Speech-to-Text supported languages (Khmer, km-KH). Accessed October 2026.
   https://docs.cloud.google.com/speech-to-text/docs/speech-to-text-supported-languages
10. Hugging Face Hub. Khmer automatic-speech-recognition models (search "khmer", sorted by downloads).
    Accessed October 2026. https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&search=khmer
