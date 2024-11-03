
## Quick Start

### Requirements
The project requires Python 3.11. See `requirements.txt`.

    pip install -r requirements.txt

### Inference

python inference.py --num_layers=48 --latent_dim=12 \
                        --output_folder=your_output_folder \
                        --model_path=your_model_weight_path \
                        --audio_path=your_audio_path \
                        --midi_path=your_midi_path \
                        --chord_path=your_chord_path \
                        --prompt_path=your_prompt_path \
                        --onset=your_onset  
See `demo` folder for the input data format. `Onset` should be a number indicating the starting second of the input audio.

