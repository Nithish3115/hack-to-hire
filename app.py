import os
import subprocess
import base64
import wave
import io
import zipfile
import shutil
import wget  # Added for downloading models
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import traceback

# # --- AUTO-DOWNLOAD MISSING MODELS ---
# def download_required_models():
#     """
#     Checks for the existence of the large RVC model weights.
#     If missing (e.g. on a fresh git clone), downloads them automatically.
#     """
#     # Define the local path where the file should be
#     model_path = "rvc/models/embedders/contentvec/pytorch_model.bin"
    
#     # Define the URL to download from (Standard RVC Hubert/ContentVec model)
#     url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"

#     if not os.path.exists(model_path):
#         print(f"\n[INFO] Model not found at {model_path}")
#         print("[INFO] Downloading model weights (approx 360MB)... this may take a moment.")
        
#         try:
#             # Ensure the directory exists
#             os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
#             # Download
#             wget.download(url, model_path)
#             print("\n[INFO] Download complete! Model saved.")
#         except Exception as e:
#             print(f"\n[ERROR] Failed to download model: {e}")
#             print("Please manually download 'hubert_base.pt' and place it at:", model_path)

# Call this function immediately when app starts
# download_required_models()
# ------------------------------------

import os
import gdown
import zipfile
import shutil

def download_and_extract_models():
    """
    Downloads the 'models.zip' from Drive and extracts it to 'rvc/models'.
    """
    target_models_dir = "rvc/models"
    check_file = os.path.join(target_models_dir, "predictors", "rmvpe.pt")
    
    # --- PASTE YOUR GOOGLE DRIVE FILE ID HERE ---
    zip_file_id = "1xHdwAJ8CZngfKnyD_dTZ3d_VFXLUyRLX" 
    # --------------------------------------------
    
    if not os.path.exists(check_file):
        # flush=True forces this message to appear in logs IMMEDIATELY
        print(f"\n[INFO] Models not found. Starting download...", flush=True)
        
        zip_output_path = "models.zip"
        
        try:
            # 1. Download
            print(f"[INFO] Downloading from Drive (ID: {zip_file_id})...", flush=True)
            output = gdown.download(id=zip_file_id, output=zip_output_path, quiet=False)
            
            if not output:
                print("[ERROR] Download failed! gdown returned None.", flush=True)
                return

            # 2. Extract
            print("[INFO] Download finished. Extracting...", flush=True)
            os.makedirs("rvc", exist_ok=True)
            
            with zipfile.ZipFile(zip_output_path, 'r') as zip_ref:
                first_item = zip_ref.namelist()[0]
                if first_item.startswith('models/') or first_item.startswith('models\\'):
                    zip_ref.extractall("rvc")
                else:
                    os.makedirs(target_models_dir, exist_ok=True)
                    zip_ref.extractall(target_models_dir)
                    
            print("[INFO] Extraction complete. App is ready!", flush=True)
            
            # 3. Cleanup
            if os.path.exists(zip_output_path):
                os.remove(zip_output_path)
                
        except Exception as e:
            # This will print the exact error to the logs
            print(f"\n[CRITICAL ERROR] Failed to setup models: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
    else:
        print("[INFO] Models already present.", flush=True)
def generate_training_filelist(model_name, model_path, sample_rate, version):
    """
    Generate filelist.txt following the Colab approach
    """
    try:
        # Directory paths following RVC structure
        gt_wavs_dir = os.path.join(model_path, 'sliced_audios')  # Our sliced_audios = gt_wavs in Colab
        feature_dir = os.path.join(model_path, 'extracted')      # Our extracted = feature in Colab  
        f0_dir = os.path.join(model_path, 'f0')                 # F0 directory
        f0nsf_dir = os.path.join(model_path, 'f0_voiced')       # F0 voiced directory
        
        # Get common files across all directories
        gt_files = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir) if name.endswith('.wav')])
        feature_files = set([name.split(".")[0] for name in os.listdir(feature_dir) if name.endswith('.npy')])
        f0_files = set([name.split(".")[0] for name in os.listdir(f0_dir) if name.endswith('.wav.npy')])
        f0nsf_files = set([name.split(".")[0] for name in os.listdir(f0nsf_dir) if name.endswith('.wav.npy')])
        
        # Get intersection of all files
        common_names = gt_files & feature_files & f0_files & f0nsf_files
        
        print(f"Found {len(gt_files)} gt_wavs, {len(feature_files)} features, {len(f0_files)} f0, {len(f0nsf_files)} f0nsf")
        print(f"Common files across all directories: {len(common_names)}")
        
        if len(common_names) == 0:
            print("ERROR: No common files found across all directories!")
            return False
        
        # Generate filelist entries
        opt = []
        spk_id = 0  # Speaker ID
        
        for name in sorted(common_names):
            # Format: gt_wavs_path|feature_path|f0_path|f0nsf_path|speaker_id
            entry = "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s" % (
                gt_wavs_dir.replace("\\", "\\\\"),
                name,
                feature_dir.replace("\\", "\\\\"),
                name,
                f0_dir.replace("\\", "\\\\"),
                name,
                f0nsf_dir.replace("\\", "\\\\"),
                name,
                spk_id
            )
            opt.append(entry)
        
        # Add mute files for training stability (following Colab approach)
        # This helps with training stability
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mute_base = os.path.join(current_dir, 'logs', 'mute')
        if os.path.exists(mute_base):
            fea_dim = 768 if version == "v2" else 256
            for _ in range(2):  # Add 2 mute entries
                mute_entry = "%s/sliced_audios/mute%s.wav|%s/extracted/mute.npy|%s/f0/mute.wav.npy|%s/f0_voiced/mute.wav.npy|%s" % (
                    mute_base, sample_rate, mute_base, mute_base, mute_base, spk_id
                )
                opt.append(mute_entry)
        
        # Shuffle and write filelist
        from random import shuffle
        shuffle(opt)
        
        filelist_path = os.path.join(model_path, 'filelist.txt')
        with open(filelist_path, 'w') as f:
            f.write('\n'.join(opt))
        
        print(f"Generated filelist with {len(opt)} entries")
        return True
        
    except Exception as e:
        print(f"Error generating filelist: {str(e)}")
        return False

def setup_training_config(model_path, sample_rate, version):
    """
    Setup config.json for training following Colab approach
    """
    try:
        import json
        import pathlib
        
        # Determine config path based on version and sample rate
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if version == "v1" or sample_rate == "40k":
            config_path = os.path.join(current_dir, 'rvc', 'configs', f'{sample_rate}.json')
        else:
            config_path = os.path.join(current_dir, 'rvc', 'configs', f'{sample_rate}.json')
            
        # Fallback config paths if the above doesn't exist
        if not os.path.exists(config_path):
            config_path = os.path.join(current_dir, 'configs', f'{sample_rate}.json')
        if not os.path.exists(config_path):
            config_path = os.path.join(current_dir, 'rvc', 'configs', '40000.json')  # Default fallback
        
        config_save_path = os.path.join(model_path, 'config.json')
        
        if not pathlib.Path(config_save_path).exists():
            if os.path.exists(config_path):
                with open(config_save_path, "w", encoding="utf-8") as f:
                    with open(config_path, "r") as config_file:
                        config_data = json.load(config_file)
                        json.dump(
                            config_data,
                            f,
                            ensure_ascii=False,
                            indent=4,
                            sort_keys=True,
                        )
                    f.write("\n")
                print(f"Config copied from {config_path}")
            else:
                print(f"Config file not found at {config_path}, using default")
                # Create a basic config if none exists
                default_config = {
                    "train": {
                        "log_interval": 200,
                        "eval_interval": 800,
                        "seed": 1234,
                        "epochs": 20000,
                        "learning_rate": 0.0002,
                        "betas": [0.8, 0.99],
                        "eps": 1e-9,
                        "batch_size": 4,
                        "fp16_run": False,
                        "lr_decay": 0.999875,
                        "segment_size": 17920,
                        "init_lr_ratio": 1,
                        "warmup_epochs": 0,
                        "c_mel": 45,
                        "c_kl": 1.0
                    },
                    "data": {
                        "max_wav_value": 32768.0,
                        "sampling_rate": int(sample_rate),
                        "filter_length": 2048,
                        "hop_length": 400,
                        "win_length": 2048,
                        "n_mel_channels": 125,
                        "mel_fmin": 0.0,
                        "mel_fmax": None
                    },
                    "model": {
                        "inter_channels": 192,
                        "hidden_channels": 192,
                        "filter_channels": 768,
                        "n_heads": 2,
                        "n_layers": 6,
                        "kernel_size": 3,
                        "p_dropout": 0.1,
                        "resblock": "1",
                        "resblock_kernel_sizes": [3, 7, 11],
                        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                        "upsample_rates": [10, 10, 2, 2],
                        "upsample_initial_channel": 512,
                        "upsample_kernel_sizes": [16, 16, 4, 4],
                        "spk_embed_dim": 109,
                        "gin_channels": 256,
                        "sr": int(sample_rate)
                    }
                }
                
                with open(config_save_path, "w", encoding="utf-8") as f:
                    json.dump(default_config, f, ensure_ascii=False, indent=4, sort_keys=True)
                    f.write("\n")
                print("Created default config.json")
        
    except Exception as e:
        print(f"Error setting up config: {str(e)}")

def run_colab_style_training(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, 
                            batch_size12, if_save_latest13, pretrained_G14, pretrained_D15,
                            gpus16, if_cache_gpu17, if_save_every_weights18, version19):
    """
    Run training using the exact Colab command-line approach
    """
    try:
        import subprocess
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Build the exact command from Colab
        cmd = [
            'python', 'core.py', 'train',
            '--model_name', exp_dir1,
            '--save_every_epoch', str(save_epoch10),
            '--save_only_latest', str(if_save_latest13).lower(),
            '--save_every_weights', str(if_save_every_weights18).lower(), 
            '--total_epoch', str(total_epoch11),
            '--sample_rate', str(sr2),
            '--batch_size', str(batch_size12),
            '--gpu', str(gpus16),
            '--pretrained', 'true',  # Always use pretrained
            '--custom_pretrained', 'true' if (pretrained_G14 and pretrained_D15) else 'false',
            '--overtraining_detector', 'false',  # Disable for stability
            '--overtraining_threshold', '50',
            '--cleanup', 'false',
            '--cache_data_in_gpu', str(if_cache_gpu17).lower(),
            '--vocoder', 'HiFi-GAN',
            '--checkpointing', 'false',
            '--index_algorithm', 'Auto'  # Add the missing index algorithm parameter
        ]
        
        # Add custom pretrained paths if provided
        if pretrained_G14:
            cmd.extend(['--g_pretrained_path', pretrained_G14])
        if pretrained_D15:
            cmd.extend(['--d_pretrained_path', pretrained_D15])
        
        print(f"Running training command: {' '.join(cmd)}")
        
        # Execute the training command
        process = subprocess.Popen(
            cmd, 
            cwd=current_dir, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            bufsize=1, 
            universal_newlines=True
        )
        
        # Print output as it runs
        output_lines = []
        for line in process.stdout:
            line = line.strip()
            print(line)
            output_lines.append(line)
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            return "Training completed successfully"
        else:
            error_output = '\n'.join(output_lines[-20:])  # Last 20 lines for error context
            return f"Training failed with return code {return_code}. Last output: {error_output}"
            
    except Exception as e:
        print(f"Training execution error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Training failed: {str(e)}"

import traceback

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
DATASET_FOLDER = 'datasets'
LOGS_FOLDER = 'logs'  # This should match core.py logs_path
MODELS_FOLDER = 'model'

# Create necessary directories
for folder in [UPLOAD_FOLDER, DATASET_FOLDER, LOGS_FOLDER, MODELS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Debug: Print the logs folder path to ensure it matches core.py
current_script_directory = os.path.dirname(os.path.realpath(__file__))
core_logs_path = os.path.join(current_script_directory, "logs")
print(f"App LOGS_FOLDER: {os.path.abspath(LOGS_FOLDER)}")
print(f"Core logs_path: {os.path.abspath(core_logs_path)}")
if os.path.abspath(LOGS_FOLDER) != os.path.abspath(core_logs_path):
    print("WARNING: Logs folder mismatch!")
    LOGS_FOLDER = core_logs_path  # Use the same path as core.py


# Route to serve files from the uploads directory
from flask import send_from_directory

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# --- ROUTES ---

@app.route('/')
def home():
    """Serves the main HTML page for the frontend."""
    return render_template('index.html')

@app.route('/run_inference', methods=['POST'])
def run_inference():
    """
    Handles the inference request from the frontend.
    Receives form data, saves files, calls the core script, and returns the result.
    """
    try:
        # Check if files were uploaded and if 'uploads' directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        print("--- Incoming /run_inference request ---")
        print("Files received:", list(request.files.keys()))
        for k, v in request.files.items():
            print(f"  {k}: filename={v.filename}, size={len(v.read())} bytes")
            v.seek(0)  # Reset file pointer after reading
        print("Form data received:", dict(request.form))

        audio_file_provided = False
        audio_path = None

        # Case 1: Handle a regular file upload from the 'File Upload' tab
        if 'audio_file' in request.files and request.files['audio_file'].filename != '':
            audio_file = request.files['audio_file']
            print(f"audio_file filename: {audio_file.filename}")
            audio_path = os.path.join('uploads', secure_filename(audio_file.filename))
            audio_file.save(audio_path)
            print(f"Saved audio_file to {audio_path}, size: {os.path.getsize(audio_path)} bytes")
            audio_file_provided = True
        
        # Case 2: Handle a base64 audio string from the 'Text to Voice' tab
        elif 'tts_audio_data' in request.form:
            tts_audio_data = request.form['tts_audio_data']
            tts_audio_filename = request.form.get('tts_audio_filename', 'gemini_tts.wav')
            
            # The base64 string includes a prefix "data:audio/wav;base64,"
            # We need to remove this prefix before decoding.
            if 'base64,' in tts_audio_data:
                header, encoded = tts_audio_data.split(',', 1)
                audio_bytes = base64.b64decode(encoded)

                audio_path = os.path.join('uploads', secure_filename(tts_audio_filename))
                
                # Write the binary data to a .wav file
                with open(audio_path, 'wb') as f:
                    f.write(audio_bytes)
                audio_file_provided = True

        if not audio_file_provided:
            return jsonify({
                'success': False,
                'message': "No audio file found. Please upload a file or synthesize from text.",
                'error_details': "No audio_file or tts_audio_data received."
            }), 400

        # Get and save the model and index files
        model_file = request.files.get('model_file')
        index_file = request.files.get('index_file')

        model_path = os.path.join('uploads', secure_filename(model_file.filename))
        index_path = os.path.join('uploads', secure_filename(index_file.filename))
        
        model_file.save(model_path)
        index_file.save(index_path)



        # Get inference parameters from the form
        pitch = request.form.get('pitch', request.form.get('pitch_shift', '0'))
        index_rate = request.form.get('index_rate', '0.3')
        volume_envelope = request.form.get('volume_envelope', '1.0')
        protect = request.form.get('protect', '0.33')
        f0_method = request.form.get('f0_method', 'rmvpe')

        # Debug: Print all received inference parameters
        print("[DEBUG] Inference parameters received from form:")
        print(f"  pitch_shift: {pitch}")
        print(f"  index_rate: {index_rate}")
        print(f"  volume_envelope: {volume_envelope}")
        print(f"  protect: {protect}")
        print(f"  f0_method: {f0_method}")

        # Define the output path for the converted audio
        output_filename = 'converted_' + os.path.basename(audio_path)
        output_path = os.path.join('uploads', output_filename)

        from core import run_infer_script

        print(f"Starting inference with parameters:")
        print(f"  Model: {model_path}")
        print(f"  Index: {index_path}")
        print(f"  Audio: {audio_path}")
        print(f"  Pitch Shift: {pitch}")
        print(f"  Index Rate: {index_rate}")
        print(f"  Volume Envelope: {volume_envelope}")
        print(f"  Protect: {protect}")
        print(f"  F0 Method: {f0_method}")
        print("-" * 20)

        result_message, converted_file_path = run_infer_script(
            pitch=int(pitch),
            index_rate=float(index_rate),
            volume_envelope=float(volume_envelope),
            protect=float(protect),
            f0_method=f0_method,
            input_path=audio_path,
            output_path=output_path,
            pth_path=model_path,
            index_path=index_path,
            split_audio=False,
            f0_autotune=False,
            f0_autotune_strength=1.0,
            proposed_pitch=False,
            proposed_pitch_threshold=155.0,
            clean_audio=False,
            clean_strength=0.7,
            export_format='WAV',
            embedder_model='contentvec',
        )
        print(result_message)

        # Return the converted audio file
        return jsonify({
            'success': True,
            'message': result_message,
            'audio_url': '/' + converted_file_path
        })

    except Exception as e:
        print("An error occurred during inference:")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': "Failed to process audio. Please check your inputs and try again.",
            'error_details': str(e)
        }), 500

@app.route('/run_preprocess', methods=['POST'])
def run_preprocess():
    """
    Handles the preprocessing request from the frontend.
    Receives form data, saves audio files, and runs preprocessing.
    """
    try:
        print("--- Incoming /run_preprocess request ---")
        
        # Get form data
        model_name = request.form.get('model_name')
        sample_rate = int(request.form.get('sample_rate', 40000))
        cpu_cores = int(request.form.get('cpu_cores', 4))
        f0_method = request.form.get('f0_method', 'rmvpe')
        noise_reduction = 'noise_reduction' in request.form
        process_effects = 'process_effects' in request.form
        
        if not model_name:
            return jsonify({
                'success': False,
                'message': 'Model name is required'
            }), 400

        # Check if audio files were uploaded
        if 'training_audio' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No audio files uploaded'
            }), 400

        audio_files = request.files.getlist('training_audio')
        if not audio_files or not any(f.filename for f in audio_files):
            return jsonify({
                'success': False,
                'message': 'No audio files selected'
            }), 400

        # Create dataset directory for this model
        dataset_path = os.path.join(DATASET_FOLDER, model_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        # Save uploaded audio files
        for audio_file in audio_files:
            if audio_file.filename:
                filename = secure_filename(audio_file.filename)
                file_path = os.path.join(dataset_path, filename)
                audio_file.save(file_path)
                print(f"Saved audio file: {file_path}")

        # Import and run preprocessing
        from core import run_preprocess_script, run_extract_script
        
        print(f"Starting preprocessing for model: {model_name}")
        print(f"Dataset path: {dataset_path}")
        print(f"Expected output path: {os.path.join(LOGS_FOLDER, model_name)}")
        
        # Check the input audio file
        audio_files_in_dataset = [f for f in os.listdir(dataset_path) if f.endswith(('.wav', '.mp3', '.flac'))]
        print(f"Audio files in dataset: {audio_files_in_dataset}")
        
        for audio_file in audio_files_in_dataset:
            file_path = os.path.join(dataset_path, audio_file)
            file_size = os.path.getsize(file_path)
            print(f"  {audio_file}: {file_size} bytes")

        # Check total duration and adjust preprocessing strategy
        total_duration = 0
        for audio_file in audio_files_in_dataset:
            file_path = os.path.join(dataset_path, audio_file)
            file_size = os.path.getsize(file_path)
            print(f"  {audio_file}: {file_size} bytes")
            
            # Estimate duration (rough estimate: 1MB ≈ 1 minute for compressed audio)
            estimated_duration = file_size / (1024 * 1024)  # MB
            total_duration += estimated_duration
        
        print(f"Estimated total duration: {total_duration:.1f} minutes")
        
        # Choose preprocessing strategy based on audio length
        if total_duration > 5:  # If more than 5 minutes, use proper slicing
            print("Using Simple cutting mode for longer audio...")
            cut_preprocess = "Simple"  # Use "Simple" instead of "Cut_Audio"
            chunk_len = 15.0  # 15 second chunks - good for training
            overlap_len = 1.0  # 1 second overlap
            process_effects = True  # Enable effects for better quality
            noise_reduction = True  # Enable noise reduction
            clean_strength = 0.7  # Higher clean strength for longer audio
        else:
            print("Using Skip mode for shorter audio...")
            cut_preprocess = "Skip"
            chunk_len = 4.0
            overlap_len = 0.5
            process_effects = False
            noise_reduction = False
            clean_strength = 0.1

        # Run preprocessing with adaptive settings
        print(f"Preprocessing with mode: {cut_preprocess}, chunk_len: {chunk_len}")
        
        preprocess_result = run_preprocess_script(
            model_name=model_name,
            dataset_path=dataset_path,
            sample_rate=sample_rate,
            cpu_cores=cpu_cores,
            cut_preprocess=cut_preprocess,
            process_effects=process_effects,
            noise_reduction=noise_reduction,
            clean_strength=clean_strength,
            chunk_len=chunk_len,
            overlap_len=overlap_len,
            normalization_mode="none"
        )
        
        print(f"Preprocessing completed: {preprocess_result}")
        
        # Check if preprocessing actually created files
        model_logs_path = os.path.join(LOGS_FOLDER, model_name)
        sliced_audios_path = os.path.join(model_logs_path, 'sliced_audios')
        
        if not os.path.exists(sliced_audios_path):
            print(f"ERROR: sliced_audios folder not created at {sliced_audios_path}")
            return jsonify({
                'success': False,
                'message': f'Preprocessing failed - no sliced_audios folder created at {sliced_audios_path}'
            }), 500
        
        audio_files = [f for f in os.listdir(sliced_audios_path) if f.endswith('.wav')]
        print(f"Created {len(audio_files)} sliced audio files")
        
        if len(audio_files) == 0:
            print(f"ERROR: No audio files created in {sliced_audios_path}")
            return jsonify({
                'success': False,
                'message': f'Preprocessing failed - no audio files created'
            }), 500
        
        # Run feature extraction
        extract_result = run_extract_script(
            model_name=model_name,
            f0_method=f0_method,
            cpu_cores=cpu_cores,
            gpu=0,
            sample_rate=sample_rate,
            embedder_model='contentvec'
        )
        
        print(f"Feature extraction completed: {extract_result}")

        return jsonify({
            'success': True,
            'message': f'Model {model_name} preprocessed and extracted successfully',
            'preprocess_result': preprocess_result,
            'extract_result': extract_result
        })

    except Exception as e:
        print("An error occurred during preprocessing:")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': "Failed to preprocess data. Please check your inputs and try again.",
            'error_details': str(e)
        }), 500

@app.route('/run_training', methods=['POST'])
def run_training():
    """
    Handles the training request from the frontend - Updated with Colab-style approach.
    """
    try:
        print("--- Incoming /run_training request ---")
        
        # Get form data
        model_name = request.form.get('preprocessed_model')
        total_epochs = int(request.form.get('total_epochs', 300))
        save_every_epoch = int(request.form.get('save_every_epoch', 50))
        batch_size = int(request.form.get('batch_size', 8))
        gpu_id = int(request.form.get('gpu_id', 0))
        overtraining_detector = 'overtraining_detector' in request.form
        save_only_latest = 'save_only_latest' in request.form
        
        if not model_name:
            return jsonify({
                'success': False,
                'message': 'Please select a preprocessed model'
            }), 400

        # Check if model exists in logs folder
        model_path = os.path.join(LOGS_FOLDER, model_name)
        if not os.path.exists(model_path):
            return jsonify({
                'success': False,
                'message': f'Preprocessed model {model_name} not found'
            }), 400

        print(f"=== TRAINING MODEL: {model_name} ===")
        
        # Setup training parameters following Colab approach
        sample_rate = '40000'  # Use 40k sample rate
        version = 'v2'  # Use v2 version
        
        # Get pretrained models
        current_script_directory = os.path.dirname(os.path.abspath(__file__))
        pretrained_G = os.path.join(current_script_directory, 'assets', 'pretrained_v2', f'f0G{sample_rate}.pth')
        pretrained_D = os.path.join(current_script_directory, 'assets', 'pretrained_v2', f'f0D{sample_rate}.pth')
        
        # If pretrained files don't exist, use empty strings (training script will handle defaults)
        if not os.path.exists(pretrained_G):
            pretrained_G = ""
        if not os.path.exists(pretrained_D):
            pretrained_D = ""
            
        print(f"Pretrained G: {pretrained_G if pretrained_G else 'Default'}")
        print(f"Pretrained D: {pretrained_D if pretrained_D else 'Default'}")

        # Generate proper filelist following Colab approach
        success = generate_training_filelist(model_name, model_path, sample_rate, version)
        if not success:
            return jsonify({
                'success': False,
                'message': 'Failed to generate training filelist. Check that preprocessing completed successfully.'
            }), 500

        # Setup config file
        setup_training_config(model_path, sample_rate, version)

        # Execute training using Colab-style approach
        print("=== Starting Training Process ===")
        training_result = run_colab_style_training(
            exp_dir1=model_name,
            sr2=int(sample_rate),
            if_f0_3=True,  # Always use f0 (pitch) for voice cloning
            spk_id5=0,  # Single speaker
            save_epoch10=save_every_epoch,
            total_epoch11=total_epochs,
            batch_size12=batch_size,
            if_save_latest13=save_only_latest,
            pretrained_G14=pretrained_G,
            pretrained_D15=pretrained_D,
            gpus16=str(gpu_id),
            if_cache_gpu17=False,  # Keep false to avoid memory issues
            if_save_every_weights18=False,
            version19=version
        )
        print(f"Training result: {training_result}")

        # Create zip file with trained model
        zip_filename = f"{model_name}_trained_model.zip"
        zip_path = os.path.join(MODELS_FOLDER, zip_filename)
        
        # Debug: List all files in the model directory
        print(f"Contents of {model_path}:")
        for root, dirs, files in os.walk(model_path):
            for file in files:
                print(f"  {os.path.join(root, file)}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add model files from logs folder - look for both .pth and .index files
            files_added = []
            
            # Look for .pth files (trained models) - these are in the main directory
            for file in os.listdir(model_path):
                if file.endswith('.pth'):
                    file_path = os.path.join(model_path, file)
                    zipf.write(file_path, file)
                    files_added.append(file)
                    print(f"Added .pth file: {file}")
            
            # If no .pth files found in logs, check uploads folder for pre-existing models
            if not any(f.endswith('.pth') for f in files_added):
                print(f"No .pth files found in logs/{model_name}, checking uploads folder...")
                uploads_path = os.path.join(current_script_directory, 'uploads')
                if os.path.exists(uploads_path):
                    for file in os.listdir(uploads_path):
                        # Look for .pth files that might match this model name or contain it
                        if file.endswith('.pth') and (model_name.lower() in file.lower() or file.startswith(model_name)):
                            file_path = os.path.join(uploads_path, file)
                            zipf.write(file_path, file)
                            files_added.append(file)
                            print(f"Added .pth file from uploads: {file}")
            
            # Look for .index files - these can be in main directory or subdirectories
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith('.index'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, model_path)
                        zipf.write(file_path, arcname)
                        files_added.append(arcname)
                        print(f"Added .index file: {arcname}")
            
            # Also add config.json and model_info.json if they exist
            for config_file in ['config.json', 'model_info.json', 'filelist.txt']:
                config_path = os.path.join(model_path, config_file)
                if os.path.exists(config_path):
                    zipf.write(config_path, config_file)
                    files_added.append(config_file)
                    print(f"Added config file: {config_file}")
            
            print(f"Total files added to zip: {files_added}")
            
            if not any(f.endswith('.pth') for f in files_added):
                # Create a helpful message about training status
                message = f"No .pth model files found for {model_name}. "
                message += "This indicates that training hasn't been completed yet. "
                message += "The download contains preprocessing files (index, config) but no trained model weights. "
                message += "Please complete the training process to generate .pth files."
                raise Exception(message)

        return jsonify({
            'success': True,
            'message': f'Model {model_name} trained successfully',
            'training_result': training_result,
            'download_url': f'/download_model/{zip_filename}',
            'filename': zip_filename
        })

    except Exception as e:
        print("An error occurred during training:")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': "Failed to train model. Please check your inputs and try again.",
            'error_details': str(e)
        }), 500

@app.route('/get_preprocessed_models', methods=['GET'])
def get_preprocessed_models():
    """
    Returns a list of available preprocessed models with details.
    """
    try:
        models = []
        if os.path.exists(LOGS_FOLDER):
            for item in os.listdir(LOGS_FOLDER):
                item_path = os.path.join(LOGS_FOLDER, item)
                if os.path.isdir(item_path):
                    # Check if it has the required preprocessing files
                    sliced_audios_path = os.path.join(item_path, 'sliced_audios')
                    if os.path.exists(sliced_audios_path):
                        # Count files in sliced_audios to give info about dataset size
                        audio_count = len([f for f in os.listdir(sliced_audios_path) if f.endswith('.wav')])
                        
                        model_info = {
                            'name': item,
                            'audio_count': audio_count,
                            'path': item_path,
                            'has_f0': os.path.exists(os.path.join(item_path, 'f0')),
                            'has_features': os.path.exists(os.path.join(item_path, 'extracted'))
                        }
                        models.append(model_info)
        
        return jsonify({
            'success': True,
            'models': [model['name'] for model in models],  # For dropdown
            'models_detail': models  # For detailed view
        })

    except Exception as e:
        print("An error occurred getting preprocessed models:")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': "Failed to get preprocessed models",
            'error_details': str(e)
        }), 500

@app.route('/download_model/<filename>')
def download_model(filename):
    """
    Serves trained model zip files for download.
    """
    try:
        file_path = os.path.join(MODELS_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        print(f"Error downloading model: {e}")
        return jsonify({'error': 'Download failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
