import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

class E1_DAIC():
    """
    A class for processing and managing the DAIC-WOZ, E-DAIC-WOZ and E1-DAIC-WOZ datasets, including dataset creation, audio preprocessing, and dataset splitting.

    Attributes:
        daic_path (str): Path to the original DAIC-WOZ dataset directory.
        e_daic_path (str): Path to the original E-DAIC-WOZ dataset directory.
        e1_daic_path (str): Path to the processed E1-DAIC-WOZ dataset directory.
        e_daic_fold (list): List of folder paths for each participant in the E-DAIC-WOZ dataset.
        e1_daic_fold (list or None): List of folder paths for each participant in the E1-DAIC-WOZ dataset, set during dataset creation.
    
    Methods:
    --------
    __init__(daic_path: str, e_daic_path: str, e1_daic_path: str): 
        Initializes the E1_DAIC class with dataset paths.
    __make_dataset() -> pd.DataFrame:
        Creates and processes the dataset by loading, merging, and cleaning data from CSV files.
    __create_splits(df: pd.DataFrame) -> pd.DataFrame:
        Splits the dataframe into training, testing, and development sets based on original DAIC-WOZ splits.
    __preprocess() -> pd.DataFrame:
        Preprocesses the E1-DAIC-WOZ dataset by processing audio and transcription files, removing overlapping segments, and saving the results.
    _load_dataset() -> pd.DataFrame:
        Loads the E1-DAIC-WOZ dataset from a CSV file, or preprocesses and generates it if not found.
    get_dataset_splits() -> tuple:
        Loads or creates the E1-DAIC-WOZ dataset and returns the training, testing, and development splits as DataFrames.
    """
    def __init__(self, daic_path : str = 'datasets/DAIC-WOZ/', e_daic_path : str = 'datasets/E-DAIC-WOZ/', e1_daic_path : str = 'datasets/E1-DAIC-WOZ/'):
        self.daic_path = daic_path
        self.e_daic_path = e_daic_path
        self.e1_daic_path = e1_daic_path
        self.e_daic_fold = [f"{self.e_daic_path}/{i}_P" for i in range(600, 719)]
        self.e1_daic_fold = None

    # Making dataset CSV
    def __make_dataset(self) -> pd.DataFrame:
        """
        Creates and processes the dataset for the E1_DAIC class by loading, merging, and cleaning data from CSV files.
        This method performs the following steps:
        1. Loads `dev`, `test`, and `train` splits from CSV files.
        2. Concatenates and sorts the data by `Participant_ID`.
        3. Converts `PHQ_Score` >= 10 to binary label 1 in `PHQ_Binary`.
        4. Removes samples with `PHQ_Binary` == 0 and `Participant_ID` >= 600.
        5. Updates the `e1_daic_fold` attribute with participant folder paths.
        
        :returns pd.DataFrame: A DataFrame containing participant IDs, PHQ binary labels, PHQ scores, and data split information.
        """
        files = {'dev': self.e_daic_path + 'dev_split.csv', 
                 'test': self.e_daic_path + 'test_split.csv',
                 'train': self.e_daic_path + 'train_split.csv'}

        file = pd.concat([pd.read_csv(f, usecols=["Participant_ID", "PHQ_Binary", "PHQ_Score"]) 
                          for _, f in files.items()], ignore_index=True)
        file = file.sort_values(by="Participant_ID")

        # Fixing the PHQ_Score to binary classification
        file.loc[file["PHQ_Score"] >= 10, "PHQ_Binary"] = 1
        
        # Removing class 0 samples with Participant_ID >= 600
        file = file.drop(file[(file['PHQ_Binary'] == 0) & (file['Participant_ID'] >= 600)].index)

        self.e1_daic_fold = [f"{self.e_daic_path}{i}_P" for i in file['Participant_ID'].unique()]

        return file
    
    # Splitting the dataset into train, test, and dev sets
    def __create_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Splits the dataframe into training, testing, and development sets.

        :param pd.DataFrame df: The input dataframe to split.
        :returns pd.DataFrame: A DataFrame with an added 'Split' column indicating the split for each row.
        """

        train_ids = pd.read_csv(self.daic_path + 'train_split_Depression_AVEC2017.csv')['Participant_ID'].values
        test_ids = pd.read_csv(self.daic_path + 'test_split_Depression_AVEC2017.csv')['Participant_ID'].values
        dev_ids = pd.read_csv(self.daic_path + 'dev_split_Depression_AVEC2017.csv')['Participant_ID'].values
        e_daic_ids = df[df['Participant_ID'] >= 600]['Participant_ID'].values

        df.loc[df['Participant_ID'].isin(train_ids), 'Split'] = 'train'
        df.loc[df['Participant_ID'].isin(test_ids), 'Split'] = 'test'
        df.loc[df['Participant_ID'].isin(dev_ids), 'Split'] = 'dev'
        df.loc[df['Participant_ID'].isin(e_daic_ids), 'Split'] = 'train'

        return df

    # Preprocessing data based on E1-DAIC dataset CSV   
    def __preprocess(self) -> pd.DataFrame:
        """
        Preprocesses the E1-DAIC-WOZ dataset by performing the following steps:
        1. Generates the initial dataset using `__make_dataset`.
        2. Iterates through each interview folder in `self.e1_daic_fold`:
            - Loads audio and transcription.
            - For each segment in the transcription:
                - Extracts the audio segment.
                - Removes silence from the segment.
                - Stores the processed segment and its new duration.
            - Concatenates all processed segments.
            - Creates a new transcription with updated timestamps.
            - Saves the processed audio and new transcription.
        3. Saves the final dataset as a CSV file in `self.e1_daic_path`.
        
        :returns pd.DataFrame: The processed dataset as a pandas DataFrame, or None if an error occurs during processing.
        """
        
        try:
            df = self.__make_dataset()
        except Exception as e:
            print(f"Error making dataset: {e}")
            return

        if not os.path.exists(self.e1_daic_path):
            os.makedirs(self.e1_daic_path)

        for interview in tqdm(self.e1_daic_fold, desc="Processing E1-DAIC-WOZ interviews"):
            if not os.path.exists(interview):
                print(f"Interview folder {interview} does not exist. Skipping...")
                continue

            try:
                interview_id_str = interview.split('/')[-1].split("_")[0]
                interview_id = int(interview_id_str)
                audio_path = f'{interview}/{interview_id}_AUDIO.wav'
                
                audio, sr = librosa.load(audio_path, sr=16000)
                audio = audio / np.max(np.abs(audio))

                if interview_id < 600:
                    # DAIC-WOZ: Use original transcripts to filter for 'Participant'
                    transcription_path = f'{self.daic_path}/{interview_id_str}_P/{interview_id_str}_TRANSCRIPT.csv'
                    transcription_df = pd.read_csv(transcription_path, sep='\t')
                    transcription_df = transcription_df[transcription_df['speaker'] == 'Participant'].copy()
                    # Rename columns to match E-DAIC-WOZ format
                    transcription_df = transcription_df.rename(columns={'start_time': 'Start_Time', 'stop_time': 'End_Time'})
                else:
                    # E-DAIC-WOZ: Use provided transcripts
                    transcription_path = f'{interview}/{interview_id_str}_Transcript.csv'
                    transcription_df = pd.read_csv(transcription_path)

                # Remove speaker column if it exists
                if 'speaker' in transcription_df.columns:
                    transcription_df = transcription_df.drop(columns=['speaker'])

                # Remove Confidence column if it exists
                if 'Confidence' in transcription_df.columns:
                    transcription_df = transcription_df.drop(columns=['Confidence'])

                # Rename Text to value if it exists
                if 'Text' in transcription_df.columns:
                    transcription_df = transcription_df.rename(columns={'Text': 'value'})
                
                # Remove utterances that contains <synch>, <sync> and scrubbed_entry
                if 'value' in transcription_df.columns:
                    transcription_df = transcription_df[~transcription_df['value'].astype(str).str.contains(r'<synch>|<sync>|scrubbed_entry', case=False, na=False)]

                # Clean up overlapping segments
                transcription_df['Prev_End_Time'] = transcription_df['End_Time'].shift(1)
                transcription_df = transcription_df[transcription_df['Start_Time'] >= transcription_df['Prev_End_Time'].fillna(0)].copy()
                transcription_df = transcription_df.drop(columns=['Prev_End_Time'])
                transcription_df = transcription_df[transcription_df['Start_Time'] < transcription_df['End_Time']].copy()

                segments = []
                new_transcription_rows = []
                current_time = 0.0

                for _, row in transcription_df.iterrows():
                    start_sample = int(row['Start_Time'] * sr)
                    end_sample   = int(row['End_Time'] * sr)
                    segment      = audio[start_sample:end_sample]

                    segments.append(segment)
                    
                    # Calculate new duration and update timestamps
                    new_duration = len(segment) / sr
                    new_start_time = current_time
                    new_end_time = current_time + new_duration
                    
                    new_row = row.copy()
                    new_row['Start_Time'] = new_start_time
                    new_row['End_Time'] = new_end_time
                    new_transcription_rows.append(new_row)

                    current_time = new_end_time
                
                # Concatenate all processed audio segments
                final_audio = np.concatenate(segments)
                new_transcription_df = pd.DataFrame(new_transcription_rows)

                # Save the processed audio and transcription file
                if not os.path.exists(interview_fold := f'{self.e1_daic_path}{interview_id}_P'):
                    os.makedirs(interview_fold)

                output_prefix = f'{interview_fold}/{interview_id}_'
                sf.write(f'{output_prefix}AUDIO.wav', final_audio, 16000)
                new_transcription_df.to_csv(f'{output_prefix}Transcript.csv', index=False, float_format='%.3f')

            except Exception as e:
                print(f"Error processing interview {interview}: {e}")
                return

        # Create and save splits
        df = self.__create_splits(df)
        df.to_csv(self.e1_daic_path + 'e1_daic_dataset.csv', index=False)

        return df

    # Loading the dataset
    def _load_dataset(self) -> pd.DataFrame:
        """
        Loads the E1_DAIC dataset from a CSV file into a pandas DataFrame.
        If the dataset file is not found at the specified path, it returns None.
        
        :returns pd.DataFrame or None: The loaded dataset, or None if the file is not found.
        """

        try:
            return pd.read_csv(self.e1_daic_path + 'e1_daic_dataset.csv')
        except FileNotFoundError:
            print(f"Dataset file not found at {self.e1_daic_path + 'e1_daic_dataset.csv'}. Preprocessing the dataset...")
            return self.__preprocess()

    def get_dataset_splits(self) -> tuple:
        """
        Loads or creates the E1-DAIC-WOZ dataset and returns the train, test, and development splits.
        If the dataset file exists, it loads it. Otherwise, it preprocesses the data to create it.

        :returns tuple: A tuple containing three pandas DataFrames (`train_df`, `test_df`, `dev_df`).
                        Returns (`None`, `None`, `None`) if the dataset cannot be created.
        """
        
        print(f'{"*"*60}\nE1-DAIC-WOZ Dataset Loading/Preprocessing\n{"*"*60}')
        
        df = self._load_dataset()

        if df is None:
            print("Dataset file not found. Preprocessing the dataset...")
            print(f'{"-"*60}')
            try:
                df = self.__preprocess()
                print(f'{"-"*60}')
                print("Dataset preprocessed and saved successfully.")
            except Exception as e:
                print(f"Error preprocessing the dataset: {e}")
                return None, None, None
        else:
            print(f'E1-DAIC-WOZ dataset loaded from file.')

        if df is None:
            return None, None, None

        print(f'{"*"*60}\n')
        
        train_df = df[df['Split'] == 'train']
        test_df = df[df['Split'] == 'test']
        dev_df = df[df['Split'] == 'dev']

        return train_df, test_df, dev_df
    
    def print_audio_duration_stats(self, split):
        daic_durations = {0: 0.0, 1: 0.0}
        edaic_durations = {0: 0.0, 1: 0.0}
        daic_utterance_durations = []
        edaic_utterance_durations = []
        daic_audio_durations = {0: [], 1: []}
        edaic_audio_durations = {0: [], 1: []}

        for _, row in split.iterrows():
            participant_id = row['Participant_ID']
            phq_bin = row['PHQ_Binary']
            audio_path = f"{self.e1_daic_path}{participant_id}_P/{participant_id}_AUDIO.wav"
            transcript_path = f"{self.e1_daic_path}{participant_id}_P/{participant_id}_Transcript.csv"

            info = sf.info(audio_path)
            duration = info.frames / info.samplerate / 60  
            df_trans = pd.read_csv(transcript_path)
            utt_durs = df_trans['End_Time'] - df_trans['Start_Time']
            if participant_id >= 600:
                edaic_durations[phq_bin] += duration
                edaic_utterance_durations.extend(utt_durs.tolist())
                edaic_audio_durations[phq_bin].append(duration)
            else:
                daic_durations[phq_bin] += duration
                daic_utterance_durations.extend(utt_durs.tolist())
                daic_audio_durations[phq_bin].append(duration)

        print("\nAudio duration statistics (in minutes):")
        print(f"  Class 0: {daic_durations[0]:.2f} min")
        print(f"  Class 1: {daic_durations[1]:.2f} min (E-DAIC: {edaic_durations[1]:.2f} min)")
        print(f"  Total:   {daic_durations[0] + daic_durations[1]:.2f} min " + \
              f"(with E-DAIC: {daic_durations[0] + daic_durations[1] + edaic_durations[1]:.2f} min)")
        
        for cls in [0, 1]:
            print(f"  Average DAIC audio duration class {cls}: {np.mean(daic_audio_durations[cls]):.2f} min")
            if edaic_audio_durations[cls]:
                print(f"  Average E-DAIC audio duration class {cls}: {np.mean(edaic_audio_durations[cls]):.2f} min")
        all_daic = daic_audio_durations[0] + daic_audio_durations[1]
        all_edaic = edaic_audio_durations[0] + edaic_audio_durations[1]
        print(f"  Average total DAIC audio duration: {np.mean(all_daic):.2f} min")
        print(f"  Longest DAIC audio: {np.max(all_daic):.2f} min")
        if all_edaic:
            print(f"  Average total E-DAIC audio duration: {np.mean(all_edaic):.2f} min")
            print(f"  Longest E-DAIC audio: {np.max(all_edaic):.2f} min")
        
        if daic_utterance_durations:
            avg_utt = np.mean(daic_utterance_durations)
            min_utt = np.min(daic_utterance_durations)
            max_utt = np.max(daic_utterance_durations)
            over_10s = np.sum(np.array(daic_utterance_durations) > 10)
            print("\nUtterance statistics DAIC (seconds):")
            print(f"  Average duration: {avg_utt:.2f} s")
            print(f"  Shortest utterance: {min_utt:.2f} s")
            print(f"  Longest utterance: {max_utt:.2f} s")
            print(f"  Utterances > 10s: {over_10s} ({over_10s/len(daic_utterance_durations)*100:.2f}%)")

        if edaic_utterance_durations:
            avg_utt = np.mean(edaic_utterance_durations)
            min_utt = np.min(edaic_utterance_durations)
            max_utt = np.max(edaic_utterance_durations)
            over_10s = np.sum(np.array(edaic_utterance_durations) > 10)
            print("\nUtterance statistics E-DAIC (seconds):")
            print(f"  Average duration: {avg_utt:.2f} s")
            print(f"  Shortest utterance: {min_utt:.2f} s")
            print(f"  Longest utterance: {max_utt:.2f} s")
            print(f"  Utterances > 10s: {over_10s} ({over_10s/len(edaic_utterance_durations)*100:.2f}%)")