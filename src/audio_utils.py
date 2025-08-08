import librosa
import torch
import numpy as np

def segment_audio_by_transcript(
    audio, 
    transcript_df, 
    sample_rate, 
    max_utt_seconds, 
    min_utt_seconds,
    overlap_seconds,
    fullness_threshold=0.8,
    return_indices=False,
):
    """
    Segments audio based on a transcript.
    Can return either audio segments or their start/end sample indices.
    """
    max_samples = int(max_utt_seconds * sample_rate)
    min_samples = int(min_utt_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)
    step = max_samples - overlap_samples
    fullness_threshold = int(fullness_threshold * max_samples)

    segments = []
    current_utterances_audio = []
    current_duration_samples = 0
    current_segment_start_sample = None

    for i, row in transcript_df.iterrows():
        start_sample = int(row['Start_Time'] * sample_rate)
        end_sample = int(row['End_Time'] * sample_rate)
        utterance_audio = audio[start_sample:end_sample]
        utterance_samples = len(utterance_audio)

        if not current_utterances_audio:
            current_segment_start_sample = start_sample
        
        # If the utterance fits perfectly in the remaining space of the buffer
        if current_duration_samples + utterance_samples <= max_samples:
            current_utterances_audio.append(utterance_audio)
            current_duration_samples += utterance_samples
        else:
            # The utterance is too long for the remaining space.
            audio_to_chunk = None 
            if current_duration_samples >= fullness_threshold:
                # The current buffer is "full enough".
                if return_indices:
                    segment_end_sample = current_segment_start_sample + current_duration_samples
                    segments.append((current_segment_start_sample, segment_end_sample))
                else:
                    full_segment = np.concatenate(current_utterances_audio)
                    segments.append(full_segment)
                
                # The entire new utterance will be chunked.
                audio_to_chunk = utterance_audio
                current_segment_start_sample = start_sample
            else:
                # Fill the remaining space in the buffer.
                space_left = max_samples - current_duration_samples
                part_to_fill = utterance_audio[:space_left]
                current_utterances_audio.append(part_to_fill)

                # Finalize and save the now-full segment.
                if return_indices:
                    segment_end_sample = current_segment_start_sample + max_samples
                    segments.append((current_segment_start_sample, segment_end_sample))
                else:
                    full_segment = np.concatenate(current_utterances_audio)
                    segments.append(full_segment)
                
                # The remainder of the new utterance will be chunked.
                audio_to_chunk = utterance_audio[space_left:]
                current_segment_start_sample = start_sample + space_left

            current_utterances_audio = []
            current_duration_samples = 0
            
            if audio_to_chunk is not None and len(audio_to_chunk) > 0:
                idx = 0
                while idx < len(audio_to_chunk):
                    chunk = audio_to_chunk[idx : idx + max_samples]
                    
                    if len(chunk) == max_samples:
                        if return_indices:
                            chunk_start = current_segment_start_sample + idx
                            chunk_end = chunk_start + len(chunk)
                            segments.append((chunk_start, chunk_end))
                        else:
                            segments.append(chunk)
                        idx += step 
                    else:
                        # This is the last, shorter chunk. It becomes the new buffer.
                        current_utterances_audio = [chunk]
                        current_duration_samples = len(chunk)
                        # Break the while loop as we've processed all of audio_to_chunk
                        break

    if current_utterances_audio:
        # Only add the last segment if it meets the minimum duration requirement.
        if current_duration_samples >= min_samples:
            if return_indices:
                segment_end_sample = current_segment_start_sample + current_duration_samples
                segments.append((current_segment_start_sample, segment_end_sample))
            else:
                last_segment = np.concatenate(current_utterances_audio)
                segments.append(last_segment)
            
    return segments

def segment_audio_sliding_window(
    audio, 
    sample_rate, 
    max_utt_seconds, 
    min_utt_seconds,
    overlap_seconds,
    return_indices=False
):
    """
    Segments audio using a sliding window approach.
    Can return either audio segments or their start/end sample indices.
    """
    segment_samples = int(max_utt_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)
    min_samples = int(min_utt_seconds * sample_rate)
    step = segment_samples - overlap_samples

    segments = []
    
    for i in range(0, len(audio), step):
        start_sample = i
        end_sample = start_sample + segment_samples
        segment = audio[start_sample:end_sample]
        if return_indices:
            segments.append((start_sample, start_sample + len(segment)))
        else:
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