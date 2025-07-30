import os
import numpy as np
from disvoice.prosody import Prosody                # type: ignore
from disvoice.phonation import Phonation            # type: ignore
from disvoice.articulation import Articulation      # type: ignore

def process_interview(args):
    interview, dataset_path = args

    if not os.path.exists(audio_path := f'{dataset_path}{interview}_P/{interview}_AUDIO.wav'):
        print(f"Audio file {audio_path} does not exist. Skipping...")
        return

    if not os.path.exists(features_path := f'{dataset_path}{interview}_P/features'):
        os.makedirs(features_path)
    else:
        expected_files = [
            os.path.join(features_path, 'articulation_features.npy'),
            os.path.join(features_path, 'phonation_features.npy'),
            os.path.join(features_path, 'prosody_features.npy')
        ]
        if all(os.path.exists(f) for f in expected_files):
            #print(f"Audio features in {features_path} already exist. Skipping...")
            return
    
    articulation = Articulation()
    phonation = Phonation()
    prosody = Prosody()

    articulation_f = articulation.extract_features_file(audio_path, static=True, plots=False, fmt="npy")
    phonation_f = phonation.extract_features_file(audio_path, static=True, plots=False, fmt="npy")
    prosody_f = prosody.extract_features_file(audio_path, static=True, plots=False, fmt="npy")

    np.save(os.path.join(features_path, 'articulation_features.npy'), articulation_f)
    np.save(os.path.join(features_path, 'phonation_features.npy'), phonation_f)
    np.save(os.path.join(features_path, 'prosody_features.npy'), prosody_f)