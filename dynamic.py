import time
import os
import mediapipe as mp
import numpy as np

import cv2
import torch
import torch.nn as nn

from typing import Optional, Dict
from collections import deque
from collections import Counter

#configuration 

BUFFER_SIZE = 30
NUM_FRAMES = 20
D_MODEL=32
VELOCITY_THRESHOLD= 0.005
GESTURE_CLASSES ={
    0:"none",
    1:"swipe down",
    2:"swipe_left",
    3:"swipe_right",
    4:"swipe_up"
}

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DYNAMIC_GESTURE_MODEL_PATH = os.path.join(_BASE_DIR, "models", "dynamic_gesture_model.pth")

#model architecture

class NanoTransformer(nn.Module):

    def __init__(self,num_frames=15, num_classes=5, input_dim= 63 ,d_model = 32, num_heads=4 ):
        super().__init__(self)
        self.conv=nn.Conv1d(in_channels=63, out_channels=32, kernel_size=3,padding=1)
        self.posn_embed=nn.Parameter(torch.randn(1,num_frames, d_model))
        encoder_layer=nn.TransformerEncoderLayer(32,nhead=4,dim_feedforward=64,batch_first=True)
        self.transformer=nn.Transformer(encoder_layer,num_encoder_layers=2)
        self.classifier=nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16,5)
    )
    
    def forward(self,x):
        x=x.permute(0,2,1)
        x=self.conv(x)
        x=x.permute(0,2,1)
        x=x+self.posn_embed(x)
        x=self.transformer(x)
        x=x.mean()
        return 

#inference code
_model = None
_device = None
def model_load()->bool:
    global _model,_device
    try:
        _model=NanoTransformer()
        if os.path.exists(DYNAMIC_GESTURE_MODEL_PATH):
            _model.load_state_dict(torch.load(DYNAMIC_GESTURE_MODEL_PATH,map_location=_device,weights_only=True))
            print(f"loaded from {DYNAMIC_GESTURE_MODEL_PATH}")
        else:
            print(f"model warning not found at {DYNAMIC_GESTURE_MODEL_PATH}")
        _model.eval()
        return True
    except Exception as e:
        print(f"model error {e}")
        return False

def normalize_window(window: np.ndarray) -> np.ndarray:
    
    N = window.shape[0]
    landmarks = window.reshape(N, 21, 3)
    
   
    reference_wrist = landmarks[0, 0]
    
    
    frame0_centered = landmarks[0] - reference_wrist
    reference_scale = np.abs(frame0_centered).max() + 1e-6 # basically for edge case 0 its the problem 
    
    
    normalized = (landmarks - reference_wrist) / reference_scale
 
    return normalized.reshape(N,63)


def predict(landmark_sequence: np.ndarray) -> Optional[Dict[str, float]]:
    if _model is None:
        return None
    try:
        if landmark_sequence.ndim == 3:
            landmark_sequence = landmark_sequence.reshape(landmark_sequence.shape[0], -1)
        
        seq_len = landmark_sequence.shape[0]
        
        # Elastic Sampling
        if seq_len >= NUM_FRAMES:
            indices = np.linspace(0, seq_len-1, NUM_FRAMES).astype(int)
            sampled = landmark_sequence[indices]
        else:
            padding = np.tile(landmark_sequence[-1], (NUM_FRAMES - seq_len, 1))
            sampled = np.vstack((landmark_sequence, padding))
        
        # Normalize
        normalized = normalize_window(sampled)
        
        x = torch.from_numpy(normalized.reshape(1, NUM_FRAMES, -1).astype("float32")).to(_device)
        with torch.no_grad():
            logits = _model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return {GESTURE_CLASSES[i]: float(p) for i, p in enumerate(probs)}
    except Exception as e:
        print(f"[Predict] Error: {e}")
        return None

#temporal buffers and gating
_buffer=deque(maxlen=BUFFER_SIZE)

def push_landmark(landmarks: np.ndarray):
    _buffer.append(landmarks)

def get_sequence() ->Optional[np.ndarray]:
    if len(_buffer)<5:
        return None
    return np.stack(list(_buffer), axis=0)

def calc_velocity(current_landmarks, prev_landmarks):
    if current_landmarks is None or prev_landmarks is None:
        return 0.0
    indices=[0,8,4]

    max_v=0.0
    for i in indices:
        c= current_landmarks[i]
        p= prev_landmarks[i]
        d= np.sqrt(np.sum(c-p)**2)
        if d>max_v:
            max_v=d

#handtracking 

# UI

def draw_overlay():

#main loop
def main():
    if not model_load() or not init_tracking():
        return
    
    print("\nRunning... Press 'q' to quit\n")
    
    # Configuration
    COOLDOWN = 1.5
    CONFIDENCE_THRESHOLD = 0.85
    VOTE_CONSENSUS = 12
    from collections import Counter
    
    # State
    vote_buffer = deque(maxlen=15)
    last_detection_time = 0
    prev_landmarks = None
    
    try:
        while True:
            frame, landmarks = get_frame_and_landmarks()
            if frame is None: continue
            
            # Default Frame State
            velocity = 0.0
            gate_open = False
            gesture_name = None
            confidence = 0.0
            
            # 1. Hand Processing
            if landmarks is not None:
                # Velocity & Gate
                if prev_landmarks is not None:
                    velocity = calc_velocity(landmarks, prev_landmarks)
                
                gate_open = velocity >= VELOCITY_THRESHOLD
                push_landmark(landmarks) # Always update buffer context
                
                # Inference Logic (Gate + Cooldown)
                if gate_open and (time.time() - last_detection_time > COOLDOWN):
                    seq = get_sequence()
                    if seq is not None:
                        preds = predict(seq)
                        if preds:
                            g_name, conf = max(preds.items(), key=lambda x: x[1])
                            vote = g_name if (conf >= CONFIDENCE_THRESHOLD and g_name != "none") else "none"
                            vote_buffer.append(vote)

                prev_landmarks = landmarks
            else:
                prev_landmarks = None
                vote_buffer.clear()
            
            # 2. Consensus Check
            if len(vote_buffer) >= 10:
                counts = Counter(vote_buffer)
                if counts:
                    top_gesture, top_count = counts.most_common(1)[0]
                    if top_count >= VOTE_CONSENSUS and top_gesture != "none":
                        print(f"*** CONFIRMED: {top_gesture} (votes: {top_count}/15) ***")
                        gesture_name = top_gesture
                        confidence = 1.0 # For display
                        last_detection_time = time.time()
                        vote_buffer.clear()
            
            # 3. Visualization
            frame = cv2.flip(frame, 1)
            draw_overlay(frame, landmarks, gesture_name, confidence, velocity, gate_open)
            cv2.imshow("VisionNodes Run3 - vFINAL", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        close_tracking()
        print("Stopped.")

if __name__ == "__main__":
    main()


#-----------------  NEED TO BE INTEGRATED WITH STATIC.py