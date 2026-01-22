#%%
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

## naloga 1
def load_video_frames(video_path, max_frames=None):
    """
    Prebere video datoteko in vrne seznam slik v RGB barvnem prostoru.
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return np.array([])
        
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break
            
    cap.release()
    return np.array(frames, dtype=np.uint8)

## naloga 2
def sharpness_score(frame):
    # Pretvori v sivinsko sliko
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Izračunaj Laplaceov operator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Izračunaj varianco
    score = np.var(laplacian)
    return score

## naloga 3
def select_sharpest_frames(frames, scores, keep_fraction=0.2):
    """
    Izbere najbolj ostre slike glede na podane ocene.
    """
    # Število slik za ohranitev
    num_keep = int(len(frames) * keep_fraction)
    if num_keep == 0:
        num_keep = 1
        
    # Pridobi indekse sortiranih ocen (padajoče)
    sorted_indices = np.argsort(scores)[::-1]
    
    # Izberi najboljše indekse
    best_indices = sorted_indices[:num_keep]
    
    # Izberi slike
    best_frames = frames[best_indices]
    
    return best_frames, best_indices

## naloga 4
def lucky_imaging(frames, scores):
    """
    Izvede združevanje slik z uteženim povprečjem glede na oceno ostrine.
    """
    # Pretvori v float32 za natančen izračun
    frames_f = frames.astype(np.float32)
    scores_f = np.array(scores, dtype=np.float32)
    
    # Preoblikuj scores, da se bodo dimenzije ujemale (N, 1, 1, 1) za broadcasting
    # scores ima dolžino N, frames so (N, H, W, 3)
    weights = scores_f[:, np.newaxis, np.newaxis, np.newaxis]
    
    # Izračunaj uteženo vsoto
    weighted_sum = np.sum(frames_f * weights, axis=0)
    
    # Izračunaj vsoto uteži
    sum_weights = np.sum(weights)
    
    # Izračunaj končno sliko
    final_image_f = weighted_sum / sum_weights
    
    # Pretvori nazaj v uint8 (pazi na clipping)
    final_image = np.clip(final_image_f, 0, 255).astype(np.uint8)
    
    return final_image

if __name__ == "__main__":
    video_path = r"zagovor\data\EXAM_video.mp4"
    # Če datoteke ne najde na originalni poti, poskusi še lokalno (če se skripta poganja direktno iz mape zagovor)
    
    if not os.path.exists(video_path):
        video_path = r"data\EXAM_video.mp4"

    ## naloge 1: prikaz nekaj framov iz videa
    print("Naloga 1: Naložimo in prikažemo nekaj framov iz videa")   

    # Naložimo prvih 5 slik za prikaz
    frames = load_video_frames(video_path, max_frames=5)
    print(f"Naloženih frames: {len(frames)}")
    

    if len(frames) > 0:
        plt.figure(figsize=(20, 4))
        for i in range(min(5, len(frames))):
            plt.subplot(1, 5, i+1)
            plt.imshow(frames[i])
            plt.title(f'Frame {i}')
            plt.axis('off')
        plt.show()


    ## Naloga 2: Izračun ostrine za vse naložene slike
    frames = load_video_frames(video_path)
    print("\nNaloga 2: Izračun ostrine za vse naložene slike")

    if len(frames) > 0:
        scores = [sharpness_score(f) for f in frames]
        
        max_idx = np.argmax(scores)
        min_idx = np.argmin(scores)
        
        print(f"\nNajbolj ostra slika: indeks {max_idx}, ocena {scores[max_idx]:.2f}")
        print(f"Najmanj ostra slika: indeks {min_idx}, ocena {scores[min_idx]:.2f}")
        
        # Prikaz najostrejše in najmanj ostre slike
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(frames[max_idx])
        plt.title(f"Najbolj ostra (Frame {max_idx})\nScore: {scores[max_idx]:.2f}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(frames[min_idx])
        plt.title(f"Najmanj ostra (Frame {min_idx})\nScore: {scores[min_idx]:.2f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    ## Naloga 3: Izbira najboljših slik
    print("\nNaloga 3: Izbira najboljših slik (Lucky Imaging)")
    frames = load_video_frames(video_path)
    print(f"Naloženih frames: {len(frames)}")

    if len(frames) > 0:
        # Ponovimo izračun scores
        scores = [sharpness_score(f) for f in frames]

        keep_fraction = 0.2
        best_frames, best_indices = select_sharpest_frames(frames, scores, keep_fraction=keep_fraction)
        
        print(f"Ohranili smo {len(best_frames)} slik od {len(frames)} (delež {keep_fraction})")
        print(f"Indeksi izbranih slik: {best_indices}")
        
        # Prikaz prvih 5 najboljših slik
        if len(best_frames) > 0:
            plt.figure(figsize=(15, 3))
            count = min(5, len(best_frames))
            for i in range(count):
                plt.subplot(1, count, i+1)
                plt.imshow(best_frames[i])
                original_idx = best_indices[i]
                plt.title(f"Best #{i+1}\n(Orig: {original_idx}, Score: {scores[original_idx]:.2f})")
                plt.axis('off')
            plt.tight_layout()
            plt.show() 

    ## naloga 4: Lucky Imaging - združevanje in ostrenje
    frames = load_video_frames(video_path)
    scores = [sharpness_score(f) for f in frames]
    print("\nNaloga 4: Lucky Imaging (združevanje in ostrenje)")
    
    if len(frames) > 0:
        # Izberemo delež slik za ohranitev
        # Keep fraction adjustment: 
        # A smaller fraction (e.g. 0.1) selects only the very best frames, reducing blur.
        # However, we need enough frames to average out noise.
        # 0.1 (10%) is a good starting point for lucky imaging.
        keep_fraction_final = 0.1 
        
        best_frames_final, best_indices_final = select_sharpest_frames(frames, scores, keep_fraction=keep_fraction_final)
        
        # Pridobi ocene za najboljše slike
        best_scores_final = np.array(scores)[best_indices_final]
        
        print(f"Uporabljamo {keep_fraction_final*100}% najboljših slik ({len(best_frames_final)}) za končno sliko.")
        
        # Izvedemo združevanje
        result_lucky = lucky_imaging(best_frames_final, best_scores_final)
        
        # Najbolj ostra posamezna slika
        sharpest_idx = np.argmax(scores)
        result_sharpest = frames[sharpest_idx]
        
        # Shranimo rezultate
        # Save as png
        cv2.imwrite("result_sharpest.png", cv2.cvtColor(result_sharpest, cv2.COLOR_RGB2BGR))
        cv2.imwrite("result_lucky.png", cv2.cvtColor(result_lucky, cv2.COLOR_RGB2BGR))
        print("Slike shranjene: result_sharpest.png in result_lucky.png")

        # Prikaz rezultatov
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(result_sharpest)
        plt.title(f"Sharpest Frame (Score: {scores[sharpest_idx]:.2f})")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result_lucky)
        plt.title(f"Lucky Imaging Result (Top {keep_fraction_final*100}%)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()




