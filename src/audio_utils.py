import librosa
import torch
import numpy as np

def segment_audio_by_transcript(
    audio, 
    transcript_df, 
    sample_rate, 
    max_utt_seconds, 
    min_utt_seconds,
    overlap_seconds
):
    """
    Segments audio based on a transcript, grouping utterances and intelligently
    handling very long utterances.

    Args:
        audio (np.array): The complete audio array.
        transcript_df (pd.DataFrame): DataFrame with 'Start_Time' and 'End_Time' columns.
        sample_rate (int): The audio's sample rate.
        max_utt_seconds (float): The maximum desired duration for a segment in seconds.
        min_utt_seconds (float): The minimum duration for the final segment to be kept.
        overlap_seconds (float): The duration of the overlap in seconds when a long
                                 utterance is split into multiple chunks.
    """
    max_samples = int(max_utt_seconds * sample_rate)
    min_samples = int(min_utt_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)
    step = max_samples - overlap_samples
    fullness_threshold = int(0.8 * max_samples)

    segments = []
    current_utterances_audio = []
    current_duration_samples = 0

    for _, row in transcript_df.iterrows():
        start_sample = int(row['Start_Time'] * sample_rate)
        end_sample = int(row['End_Time'] * sample_rate)
        utterance_audio = audio[start_sample:end_sample]
        utterance_samples = len(utterance_audio)
        
        # If the utterance fits perfectly in the remaining space of the buffer
        if current_duration_samples + utterance_samples <= max_samples:
            current_utterances_audio.append(utterance_audio)
            current_duration_samples += utterance_samples
        else:
            # The utterance is too long for the remaining space.
            audio_to_chunk = None 
            if current_duration_samples >= fullness_threshold:
                # The current buffer is "full enough".
                full_segment = np.concatenate(current_utterances_audio)
                segments.append(full_segment)
                
                # The entire new utterance will be chunked.
                audio_to_chunk = utterance_audio
            else:
                # Fill the remaining space in the buffer.
                space_left = max_samples - current_duration_samples
                part_to_fill = utterance_audio[:space_left]
                current_utterances_audio.append(part_to_fill)

                # Finalize and save the now-full segment.
                full_segment = np.concatenate(current_utterances_audio)
                segments.append(full_segment)
                
                # The remainder of the new utterance will be chunked.
                audio_to_chunk = utterance_audio[space_left:]

            current_utterances_audio = []
            current_duration_samples = 0
            
            if audio_to_chunk is not None and len(audio_to_chunk) > 0:
                idx = 0
                while idx < len(audio_to_chunk):
                    chunk = audio_to_chunk[idx : idx + max_samples]
                    
                    if len(chunk) == max_samples:
                        segments.append(chunk)
                        idx += step 
                    else:
                        # This is the last, shorter chunk. It becomes the new buffer.
                        current_utterances_audio = [chunk]
                        current_duration_samples = len(chunk)
                        # Break the while loop as we've processed all of audio_to_chunk
                        break

    if current_utterances_audio:
        last_segment = np.concatenate(current_utterances_audio)
        # Only add the last segment if it meets the minimum duration requirement.
        if len(last_segment) >= min_samples:
            segments.append(last_segment)
            
    return segments

def segment_audio_sliding_window(
    audio, 
    sample_rate, 
    max_utt_seconds, 
    min_utt_seconds,
    overlap_seconds
):
    """
    Segmenta l'audio in blocchi di lunghezza fissa usando una finestra scorrevole (con sovrapposizione).

    Args:
        audio (np.array): L'array audio da segmentare.
        sample_rate (int): La frequenza di campionamento dell'audio.
        max_utt_seconds (float): La durata di ogni segmento (la "lunghezza della finestra").
        overlap_seconds (float): La durata della sovrapposizione tra segmenti consecutivi.
        min_utt_seconds (float): La durata minima dell'ultimo segmento per essere incluso.
    
    Returns:
        list[np.array]: Una lista di segmenti audio.
    """
    segment_samples = int(max_utt_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)
    min_samples = int(min_utt_seconds * sample_rate)
    step = segment_samples - overlap_samples

    segments = []
    
    for i in range(0, len(audio), step):
        segment = audio[i:i + segment_samples]
        segments.append(segment)

    if segments and len(segments[-1]) < min_samples:
        segments.pop()
        
    return segments

def load_audio(audio_path, sample_rate=16_000, offset_samples=0, duration_samples=None):
    offset = offset_samples / sample_rate
    if duration_samples is not None:
        duration = duration_samples / sample_rate
    else:
        duration = None

    audio, _ = librosa.load(audio_path, sr=sample_rate, offset=offset, duration=duration)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=0)
    audio = audio.squeeze()
    audio = torch.tensor(audio, dtype=torch.float32)
    return audio