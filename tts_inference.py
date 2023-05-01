import os
import torch
import librosa
import argparse
import numpy as np
import torchcrepe
import subprocess
import re
import librosa
from scipy.io import wavfile
import numpy as np


from omegaconf import OmegaConf
from scipy.io.wavfile import write
from vits.models import SynthesizerInfer


def load_svc_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model_g"]
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = saved_state_dict[k]
    model.load_state_dict(new_state_dict)
    return model


def compute_f0_nn(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    # Load audio
    audio = torch.tensor(np.copy(audio))[None]
    # Here we'll use a 20 millisecond hop length
    hop_length = 320
    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 1000
    # Select a model capacity--one of "tiny" or "full"
    model = "full"
    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 512
    # Compute pitch using first gpu
    pitch, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
    )
    pitch = np.repeat(pitch, 2, -1)  # 320 -> 160 * 2
    periodicity = np.repeat(periodicity, 2, -1)  # 320 -> 160 * 2
    # CREPE was not trained on silent audio. some error on silent need filter.
    periodicity = torchcrepe.filter.median(periodicity, 9)
    pitch = torchcrepe.filter.mean(pitch, 9)
    pitch[periodicity < 0.1] = 0
    pitch = pitch.squeeze(0)
    return pitch


def tts_func(_text,_rate,_voice):
    # edge_tts
    voice = "en-CA-ClaraNeural"
    output_file = _text[0:10]+".wav"
    # communicate = edge_tts.Communicate(_text, voice)
    # await communicate.save(output_file)
    if _rate>=0:
        ratestr="+{:.0%}".format(_rate)
    elif _rate<0:
        ratestr="{:.0%}".format(_rate)

    p=subprocess.Popen("edge-tts "+
                        " --text "+_text+
                        " --write-media "+output_file+
                        " --voice "+voice+
                        " --rate="+ratestr
                        ,shell=True,
                        stdout=subprocess.PIPE,
                        stdin=subprocess.PIPE)
    p.wait()
    return output_file

def text_clear(text):
    return re.sub(r"[\n\,\(\) ]", "", text)

def main(args):
    # edge tts
    text2tts = text_clear(args.text)
    output_file = tts_func(text2tts, args.rate, args.voice)

    if (args.ppg == None):
        args.ppg = "svc_tmp.ppg.npy"
        print(
            f"Auto run : python whisper/inference.py -w {output_file} -p {args.ppg}")
        os.system(f"python whisper/inference.py -w {output_file} -p {args.ppg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = OmegaConf.load(args.config)
    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp)
    load_svc_model(args.model, model)
    model.eval()
    model.to(device)

    spk = np.load(args.spk)
    spk = torch.FloatTensor(spk)

    ppg = np.load(args.ppg)
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)

    pit = compute_f0_nn(output_file, device)
    if (args.statics == None):
        print("don't use pitch shift")
    else:
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        print(f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}")
        singer_ave, singer_min, singer_max = np.load(args.statics)
        print(f"singer pitch statics: mean={singer_ave:0.1f}, \
                min={singer_min:0.1f}, max={singer_max:0.1f}")

        shift = np.log2(singer_ave/source_ave) * 12
        if (singer_ave >= source_ave):
            shift = np.floor(shift)
        else:
            shift = np.ceil(shift)
        shift = 2 ** (shift / 12)
        pit = pit * shift

    pit = torch.FloatTensor(pit)

    len_pit = pit.size()[0]
    len_ppg = ppg.size()[0]
    len_min = min(len_pit, len_ppg)
    pit = pit[:len_min]
    ppg = ppg[:len_min, :]

    with torch.no_grad():
        spk = spk.unsqueeze(0).to(device)
        ppg = ppg.unsqueeze(0).to(device)
        pit = pit.unsqueeze(0).to(device)
        len_min = torch.LongTensor([len_min]).to(device)
        audio = model(ppg, pit, spk, len_min)
        audio = audio[0, 0].data.cpu().detach().numpy()

    write(f"{args.text}.wav", hp.data.sampling_rate, audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="path of model for evaluation")
    parser.add_argument('-t', '--text', type=str, required=True,
                        help="Text to use for tts.")
    parser.add_argument('-s', '--spk', type=str, required=True,
                        help="Path of speaker.")
    parser.add_argument('-p', '--ppg', type=str,
                        help="Path of content vector.")
    parser.add_argument('--statics', type=str,
                        help="Path of pitch statics.")
    parser.add_argument('--voice', type=str, default="en-CA-ClaraNeural")
    parser.add_argument('--rate', type=float, default=0.0)
    args = parser.parse_args()

    main(args)

