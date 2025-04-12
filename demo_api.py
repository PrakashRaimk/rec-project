import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import google.generativeai as genai

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder

def generate_speech(text, reference_audio_path, args):
        in_fpath = Path(reference_audio_path.replace("\"", "").replace("\'", ""))
        preprocessed_wav = encoder.preprocess_wav(in_fpath)
        original_wav, sampling_rate = librosa.load(str(in_fpath))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        print("Loaded file succesfully")
        embed = encoder.embed_utterance(preprocessed_wav)
        print("Created the embedding")
        synthesizer = Synthesizer(args.syn_model_fpath)
        if args.seed is not None:
                torch.manual_seed(args.seed)
        texts = [text]
        embeds = [embed]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        print("Created the mel spectrogram")
        print("Synthesizing the waveform:")

            # If seed is specified, reset torch seed and reload vocoder
        if args.seed is not None:
            torch.manual_seed(args.seed)
            vocoder.load_model(args.voc_model_fpath)

            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
        generated_wav = vocoder.infer_waveform(spec)


            ## Post-generation
            # There's a bug with sounddevice that makes the audio cut one second earlier, so we
            # pad it.
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

            # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        generated_wav = encoder.preprocess_wav(generated_wav)
        return generated_wav,synthesizer.sample_rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    parser.add_argument("--reference_audio", type=str, required=True,
                        help="Path to the reference audio file for voice cloning.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)


    ## Run a test
    print("Testing your configuration with small inputs.")
    # Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
    # sampling rate, which may differ.
    # If you're unfamiliar with digital audio, know that it is encoded as an array of floats
    # (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
    # The sampling rate is the number of values (samples) recorded per second, it is set to
    # 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond
    # to an audio of 1 second.
    print("\tTesting the encoder...")
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))

    # Create a dummy embedding. You would normally use the embedding that encoder.embed_utterance
    # returns, but here we're going to make one ourselves just for the sake of showing that it's
    # possible.
    embed = np.random.rand(speaker_embedding_size)
    # Embeddings are L2-normalized (this isn't important here, but if you want to make your own
    # embeddings it will be).
    embed /= np.linalg.norm(embed)
    # The synthesizer can handle multiple inputs with batching. Let's create another embedding to
    # illustrate that
    embeds = [embed, np.zeros(speaker_embedding_size)]
    texts = ["test 1", "test 2"]
    print("\tTesting the synthesizer... (loading the model will output a lot of text)")
    mels = synthesizer.synthesize_spectrograms(texts, embeds)

    # The vocoder synthesizes one waveform at a time, but it's more efficient for long ones. We
    # can concatenate the mel spectrograms to a single one.
    mel = np.concatenate(mels, axis=1)
    # The vocoder can take a callback function to display the generation. More on that later. For
    # now we'll simply hide it like this:
    no_action = lambda *args: None
    print("\tTesting the vocoder...")
    # For the sake of making this test short, we'll pass a short target length. The target length
    # is the length of the wav segments that are processed in parallel. E.g. for audio sampled
    # at 16000 Hertz, a target length of 8000 means that the target audio will be cut in chunks of
    # 0.5 seconds which will all be generated together. The parameters here are absurdly short, and
    # that has a detrimental effect on the quality of the audio. The default parameters are
    # recommended in general.
    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

    print("All test passed! You can now synthesize speech.\n\n")


    ## Interactive speech generation
    print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
          "show how you can interface this project easily with your own. See the source code for "
          "an explanation of what is happening.\n")
    genai.configure(api_key="AIzaSyBZd6UjwDnQ0mXGiJU---pvl1vgr1tlM5s") # Replace with your API key
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        
    print("Interactive generation loop with Gemini AI...")
    num_generated = 0
    while True:
        try:
            # Get the user's message
            user_message = input("Enter your message (or type 'exit' to quit):\n")
            if user_message.lower() == "exit":
                break

            # Generate a response using Gemini AI
            response = model.generate_content(user_message)
            gemini_response_text = response.text
            print(f"Gemini AI Response: {gemini_response_text}")
            

            # Generate speech from the Gemini response
            generated_wav, sample_rate = generate_speech(gemini_response_text, args.reference_audio, args)
            if generated_wav is not None:  # Check if speech generation was successful
                # Play the audio (non-blocking)
                if not args.no_sound:
                    import sounddevice as sd
                    try:
                        sd.stop()
                        sd.play(generated_wav, sample_rate)
                    except sd.PortAudioError as e:
                        print(f"\nCaught exception: {repr(e)}")
                        print(
                            "Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n"
                        )
                    except Exception as e:
                        raise e

                # Save the audio to a file
                filename = "gemini_output_%02d.wav" % num_generated
                print(generated_wav.dtype)
                sf.write(filename, generated_wav.astype(np.float32), sample_rate)
                num_generated += 1
                print("\nSaved output as %s\n\n" % filename)

        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")