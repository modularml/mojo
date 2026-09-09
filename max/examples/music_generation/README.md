# Generating music with MiniMax-Music3

[MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) writes and
sings a song from two texts: a caption describing how the music should sound,
and lyrics with section tags such as `[verse]` and `[chorus]`. This example
renders one, either in your own process or through a running MAX server.

The [audio generation guide](https://docs.modular.com/serve/audio-generation)
covers the endpoint and the prompt format in more depth; this example is the
script that goes with it.

The interesting case is a full-length song. The model denoises 8 second
windows that overlap by half, blends them where they meet and crops the
result, so a three-minute song is dozens of windows joined at dozens of seams
rather than one long generation. The bundled song is 2 minutes 45 seconds, and
`--check-seams` measures those joins.

## Requirements

A [MAX-compatible GPU](https://docs.modular.com/faq/#gpu-requirements). The
checkpoint is roughly 28 GiB of weights across its autoregressive, diffusion
and vocoder stages, which is more than a 22 GiB card holds at once — the model
loads and frees each stage in turn, so a card that size is enough, and the
weights download on first use.

Expect the render to take several times longer than the song. On an A10G a
20 second excerpt takes about 5.7x realtime and a 60 second clip about 4.7x, so
the bundled 2:45 song is roughly a quarter of an hour. The first render on a
machine pays for compiling the graphs on top of that, which took another ten
minutes here; later runs replay it from the compilation cache.

## Rendering a song

In your own process, with the `max` package installed:

```sh
python generate_song.py --out song.wav
```

Or through a server, which is worth starting if you plan to render more than
one song, since it keeps its compiled graphs between requests:

```sh
max serve --model-path MiniMaxAI/MiniMax-Music3
python generate_song.py --out song.wav --server http://localhost:8000
```

Both paths send the same request. In process it goes through the audio
generation pipeline directly; over HTTP it goes to `/v1/audio/speech`, whose
OpenAI schema puts the lyrics in `input` and the caption in `instructions`. A
server started this way also answers `/v1/responses` if you set
`MAX_SERVE_API_TYPES='["openai","responses"]'`, which returns a URL for the
audio rather than the bytes.

A server holds its memory between requests — around 18 GiB on a 22 GiB card —
so give it the GPU to itself. A second server, or an in-process render
alongside it, fails to allocate.

While you are working on a song, render an excerpt rather than the whole
thing:

```sh
python generate_song.py --out excerpt.wav --duration 30
```

`--duration` sets the length in seconds, `--steps` the denoising steps per
window (30 by default; fewer is faster and rougher), and `--seed` fixes the
sampling so that a render repeats exactly.

## Writing your own song

A song is a JSON file. Lyrics are a list of lines so that the file stays
readable:

```json
{
  "caption": "Global Metadata: dream pop, 92 BPM, A minor, ... Vocal Details: female lead, airy breathy timbre, ... Arrangement: shimmering electric guitar and analog pad, ...",
  "lyrics": ["[intro]", "", "[verse]", "Headlights paint the empty road"],
  "duration": 165.0,
  "seed": 1235
}
```

Pass it with `--song my_song.json`.

The model card asks for the caption in three sections — Global Metadata, Vocal
Details, Arrangement — and reads the section tags in the lyrics. Both are worth
following: at full length the arrangement is where you say how the song should
develop, and a song that says nothing about its development has no reason to
develop. Tempo in the caption is a hint the model may not take literally.

## Checking the seams

A join that went wrong sounds like a click or a lurch in level. Both are cheap
to look for, because the offsets follow from the model's constants rather than
from the audio:

```sh
python generate_song.py --out song.wav --check-seams
```

Neither measurement means anything on its own — music is full of transients —
so each seam is scored against thousands of arbitrary offsets in the same
render, and the verdict comes from how many seams clear a high quantile
against how many chance predicts. To check a WAV you already have, without
paying for the render again:

```sh
python generate_song.py --analyze song.wav --duration 165
```

## Running with Bazel

From a checkout of the repository:

```sh
./bazelw run //max/examples/music_generation:generate_song -- --out song.wav
```
