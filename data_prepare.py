import os
import sys
import yaml
import json
import shutil

import gc
import torch

import ipywidgets as widgets

# 音频分割
import librosa  # Optional. Use any library you like to read audio files.
import soundfile  # Optional. Use any library you like to write audio files.
sys.path.append('./audio_slicer')
from slicer2 import Slicer

# 音频标注
import whisper

def slice_audio(in_file, out_path, out_name, base_index=0):
    audio, sr = librosa.load(in_file, sr=None, mono=False)  # Load an audio file with librosa.
    slicer = Slicer(
        sr=sr,
        threshold=-40,
        min_length=2000,
        min_interval=300,
        hop_size=10,
        max_sil_kept=500
    )
    chunks = slicer.slice(audio)
    total = 0
    for i, chunk in enumerate(chunks):
        total = total + 1
        if len(chunk.shape) > 1:
            chunk = chunk.T  # Swap axes if the audio is stereo.
        out_file = os.path.join(out_path, '%s_%03d.wav' % (out_name, base_index+i))
        soundfile.write(out_file, chunk, sr)  # Save sliced audio files with soundfile.

    return total

def slice_dir(input_dir, output_dir, model_name):
    # 列出当前工作目录中的所有文件和文件夹
    files = os.listdir(input_dir)

    base_index = 0
    # 筛选出所有以 ".wav" 结尾的文件
    wav_files = [f for f in files if f.endswith(".wav")]

    # 打印所有 wav 文件的名称
    for f in wav_files:
        base_index = base_index + slice_audio(os.path.join(input_dir, f), output_dir, model_name, base_index)


class DataPrepare:
    def __init__(self, root_path="/root/Bert-VITS2"):
        self.root_path = root_path
        # path
        self.data_dir = os.path.join(root_path, 'Data')
        self.input_dir = os.path.join(root_path, 'inputs')
        self.speaker_dir = ''
        self.audios = ''
        self.audios_raw = ''
        self.audios_wavs = ''
        self.filelists = ''
        self.models_dir = ''
        self.whisper_model = os.path.join(root_path, 'whisper')
        self.base_model = os.path.join(root_path, 'base_model')
        # widgets
        self.speaker = widgets.Text(
            value='paimeng',
            placeholder='训练的人物名称',
            description='String:',
            disabled=False   
        )
        self.log_interval = widgets.IntText(
            value=50,
            description='log_interval:',
            disabled=False
        )
        self.eval_interval = widgets.IntText(
            value=100,
            description='eval_interval:',
            disabled=False
        )
        self.epochs = widgets.IntText(
            value=100,
            description='epochs:',
            disabled=False
        )
        self.batch_size = widgets.IntText(
            value=4,
            description='batch_size:',
            disabled=False
        )
        self.bert_device = widgets.Dropdown(
            options=['cuda', 'cpu'],
            value='cuda',
            description='bert_device:',
            disabled=False,
        )
        self.bert_processs = widgets.IntText(
            value=2,
            description='bert_processs:',
            disabled=False
        )

    def mkdir(self, path):
        if os.path.exists(path):
            print(f'{path} exists')
        else:
            os.mkdir(path)

    def free_up_memory(self):
    # Prior inference run might have large variables not cleaned up due to exception during the run.
    # Free up as much memory as possible to allow this run to be successful.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def copy_base_models(self):
        src_dur = os.path.join(self.base_model, "DUR_0.pth")
        src_d = os.path.join(self.base_model, "D_0.pth")
        src_g = os.path.join(self.base_model, "G_0.pth")
        src_wd = os.path.join(self.base_model, "WD_0.pth")

        dst_dur = os.path.join(self.models_dir, "DUR_0.pth")
        dst_d = os.path.join(self.models_dir, "D_0.pth")
        dst_g = os.path.join(self.models_dir, "G_0.pth")
        dst_wd = os.path.join(self.models_dir, "WD_0.pth")

        print("复制 DUR_0.pth")
        if not os.path.exists(dst_dur):
            shutil.copy(src_dur, dst_dur)
        print("复制 D_0.pth")
        if not os.path.exists(dst_d):
            shutil.copy(src_d, dst_d)
        print("复制 G_0.pth")
        if not os.path.exists(dst_g):
            shutil.copy(src_g, dst_g)
        print("复制 Wd_0.pth")
        if not os.path.exists(dst_wd):
            shutil.copy(src_wd, dst_wd)

    def prepare_dir(self):
        self.mkdir(self.data_dir)
        self.mkdir(self.input_dir)
        self.speaker_dir = os.path.join(self.data_dir, self.speaker.value)
        self.mkdir(self.speaker_dir)
        self.audios = os.path.join(self.speaker_dir, 'audios')
        self.mkdir(self.audios)
        self.audios_raw = os.path.join(self.audios, 'raw')
        self.mkdir(self.audios_raw)
        self.audios_wavs = os.path.join(self.audios, 'wavs')
        self.mkdir(self.audios_wavs)
        self.filelists = os.path.join(self.speaker_dir, 'filelists')
        self.mkdir(self.filelists)
        self.models_dir = os.path.join(self.speaker_dir, 'models')
        self.mkdir(self.models_dir)
        self.copy_base_models()

    def show_widgets(self):
        display(self.speaker, self.log_interval,
                self.eval_interval, self.epochs,
                self.batch_size, self.bert_device,
                self.bert_processs)

    def audio_slice(self):
        speaker = self.speaker.value
        if len(self.input_dir)>0 and len(self.audios_raw)>0 and len(speaker)>0:
            slice_dir(self.input_dir, self.audios_raw, speaker)
        else:
            print("error input")

        print("分割完成")

    def audio_transcribe(self):
        lang2token = {
                    'zh': "ZH|",
                    'ja': "JP|",
                    "en": "EN|",
                }

        def transcribe(audio_path):
            # load audio and pad/trim it to fit 30 seconds
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # detect the spoken language
            _, probs = model.detect_language(mel)
            print(f"Detected language: {max(probs, key=probs.get)}")
            lang = max(probs, key=probs.get)
            # decode the audio
            options = whisper.DecodingOptions(beam_size=5)
            result = whisper.decode(model, mel, options)

            # print the recognized text
            print(result.text)
            return lang, result.text


        # assert (torch.cuda.is_available()), "Please enable GPU in order to run Whisper!"
        model = whisper.load_model("medium", download_root=self.whisper_model)
        speaker_annos = []
        total_files = sum([len(files) for r, d, files in os.walk(self.audios_raw)])
        processed_files = 0

        for i, wavfile in enumerate(list(os.walk(self.audios_raw))[0][2]):
            # try to load file as audio
            try:
                lang, text = transcribe(os.path.join(self.audios_raw, wavfile))
                if lang not in list(lang2token.keys()):
                    print(f"{lang} not supported, ignoring\n")
                    continue

                text = f"{self.audios_wavs}/{wavfile}|" + f"{self.speaker.value}|" +lang2token[lang] + text + "\n"
                speaker_annos.append(text)

                processed_files += 1
                print(f"Processed: {processed_files}/{total_files}")
            except Exception as e:
                print(e)
                continue

        if len(speaker_annos) == 0:
            print("Warning: no short audios found, this IS expected if you have only uploaded long audios, videos or video links.")
            print("this IS NOT expected if you have uploaded a zip file of short audios. Please check your file structure or make sure your audio language is supported.")
        with open(os.path.join(self.filelists, 'esd.list'), 'w', encoding='utf-8') as f:
            for line in speaker_annos:
                f.write(line)

        print("标注完成")
    
    def generate_config(self):
        with open('configs/config.json') as fp:
            configs = json.load(fp)

        configs['train']['log_interval'] = self.log_interval.value
        configs['train']['eval_interval'] = self.eval_interval.value
        configs['train']['epochs'] = self.epochs.value
        configs['train']['batch_size'] = self.batch_size.value

        configs['data']['training_files'] = os.path.join(self.filelists, 'train.list')
        configs['data']['validation_files'] = os.path.join(self.filelists, 'val.list')
        configs['data']['n_speakers'] = 1
        configs['data']['spk2id'] = {self.speaker.value : 0}

        configs_path = os.path.join(self.speaker_dir, 'config.json')
        with open(configs_path, 'w') as fp:
            json.dump(configs, fp, indent=4)
            print(f'生成配置文件: {configs_path}')

        configs = None
        with open('default_config.yml') as fp:
            configs = yaml.load(fp, Loader=yaml.FullLoader)

        configs['dataset_path'] = self.speaker_dir
        configs['preprocess_text']['transcription_path'] = 'filelists/esd.list'
        configs['bert_gen']['device'] = self.bert_device.value
        configs['bert_gen']['num_processes'] = self.bert_processs.value

        with open('config.yml', 'w') as fp:
            yaml.dump(configs, fp)
            print(f'生成配置文件: config.yml')

    def process(self):
        print('音频分割...')
        self.audio_slice()
        print('音频标注...')
        self.audio_transcribe()
        self.free_up_memory()

    def get_latest_model(self):
        models = os.listdir(self.models_dir)
        models = [m for m in models if m.startswith('G_') and m.endswith('.pth')]
        if len(models) == 0:
            return ""
        
        models.sort()
        return models[-1]
    
    def prepare_infer(self, model=""):
        if len(model) == 0:
            model = self.get_latest_model()
        
        if len(model) == 0:
            print('未找到模型，请先训练，并确保已经生成模型文件')
            return

        print('使用模型: ', model)
        configs = None
        with open('config.yml') as fp:
            configs = yaml.load(fp, Loader=yaml.FullLoader)
            configs['webui']['model'] = f'models/{model}'
            configs['webui']['port'] = 6006

        with open('config.yml', 'w') as fp:
            yaml.dump(configs, fp)
            print(f'更新配置文件: config.yml') 
