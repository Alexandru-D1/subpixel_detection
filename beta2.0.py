import time                                      
from pathlib import Path                         
import cv2                                       
import numpy as np  
import math                             
from picamera2 import Picamera2                  
from libcamera import controls 
from scipy import ndimage

FPS = 5
STATIC_N = 15  # frames (~3 s at 5 FPS) a change must persist in place to count as clutter
dt = 1/FPS
EXPOSURE = 2000
trcaks=[]
next_track_id=0
GAIN = 1.0
WIDTH = 2028
HEIGHT = 1520
RUN_SEC = 10
WARMUP_SEC = 5
WARMUP2_SEC = 5
WARMUP3_SEC = 5 #FRAMES FOR EWMA_FAST AND EWMA_SLOW TO STABILIZE
K_THRESHOLD = 4.5
SEC = 3 ########################################## CONTEXT FOR EWMA_FAST ###############################################
SEC2 = 60 ########################################## CONTEXT FOR EWMA_SLOW ###############################################
num_samples = 50000
total_pixels = WIDTH * HEIGHT
SAVE_FRAMES = False
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(exist_ok=True)
frame_period = int(1_000_000 / FPS)
if EXPOSURE > frame_period:
    raise ValueError(f"Exposure time {EXPOSURE} us is longer than frame period {frame_period} us for FPS {FPS}")

camera = Picamera2() #connect to camera
config = camera.create_video_configuration(      # describe how we want the camera to run
    main={"size": (WIDTH, HEIGHT), "format": "RGB888"},                 # don't compress, use 8 bit for each pixel
    sensor={"output_size": (WIDTH, HEIGHT), "bit_depth": 10},           # lock the sensor mode so it never changes and uses 0 to 1023 values instead of 0-255(10 bit))
    controls={                                                          
        "AeEnable": False,                                              # turn OFF automatic exposure microsec.
        "ExposureTime": EXPOSURE,                                    # our fixed exposure time
        "AnalogueGain": GAIN,                                           # our fixed (lowest) gain - gain apmlifies the signal but it can create noise
        "AwbEnable": False,                                             # turn OFF automatic white balance
        "ColourGains": (2.0, 2.0),                                      # fixed red/blue gains (exact values don't matter, they just must not change) blue was 1.5
        "FrameDurationLimits": (frame_period, frame_period),      # force a constant frame rate = our FPS
        "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Off,  # no built-in denoising: it would blur tiny objects away
        "Sharpness": 0.0,                                               # no sharpening: it adds fake edges and boosts noise
    },
)

camera.configure(config) 
camera.start()                                 
time.sleep(1)

BOOT_OFFSET = time.time() - time.monotonic()

def grab_frame():                                # helper: returns (grey image, timestamp in seconds) for the next frame
    request = camera.capture_request()           # wait for the next frame from the camera
    frame = request.make_array("main")           # turn that frame into a NumPy array (blue, green, red order)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grey: one number (0-255) per pixel
    timestamp = request.get_metadata()["SensorTimestamp"] / 1e9 + BOOT_OFFSET  # moment the sensor took the picture, in seconds
    request.release()                            # give the frame buffer back to the camera
    return gray, timestamp                       # hand both values back to the caller

def process_detections(
    event_mask, d_residuals, frame_timestamp, min_area=1, max_area=20
):
  
  # 1. Connected Components (8-connectivity)
  structure = np.ones((3, 3), dtype=int) # 3x3 matrix of ones for 8-connectivity
  labeled_array, num_features = ndimage.label(event_mask, structure=structure) # label connected components and count them

  detections = [] 

  if num_features == 0: # no detections, return empty list
    return detections

  # Extract properties for each connected component
  slices = ndimage.find_objects(labeled_array) # mini matrix for each component, bounding box coordinates

  for i, slc in enumerate(slices): #each bounding box, i is the index of the component, slc is the slice object for that component
    component_id = i + 1
    mask_patch = labeled_array[slc] == component_id # create a binary mask for the current component BOOLEAN MATRIX
    area = np.sum(mask_patch) #size of the component in pixels

    # --- CLUTTER REJECTION (Step B4) ---
    # Drop too small (cosmic rays/noise) or too large (clouds/shadows)
    if area < min_area or area > max_area:
      continue

    # Drop border events (where vignetting and flat-field error are worst)
    y_min, y_max = slc[0].start, slc[0].stop
    x_min, x_max = slc[1].start, slc[1].stop
    if y_min <= 5 or y_max >= (event_mask.shape[0] - 5):
      continue
    if x_min <= 5 or x_max >= (event_mask.shape[1] - 5):
      continue

    # --- SUBPIXEL LOCALIZATION (Step B5a - Weighted Centroid) ---
    # Extract the residual patch for this component (clip negative values at 0)

    signed = d_residuals[slc] * mask_patch # only keep the values that are not noise
    sign = -1 if signed.sum() < 0 else 1   # make a matrix with the sign of the residuals
    weights = np.clip(signed * sign, 0, None)  # if the value is negative (-) * (-) = (+) 

    total_weight = np.sum(weights)
    if total_weight <= 0:
      continue

    # Create coordinate grids relative to the patch
    y_indices, x_indices = np.indices(mask_patch.shape) # .shape gives the dimensions of the patch, np.indices creates a grid of coordinates for each pixel in the patch
                                                        # it is from 0 
    # Compute weighted centroid (subpixel precision)
    local_x = np.sum(x_indices * weights) / total_weight
    local_y = np.sum(y_indices * weights) / total_weight

    # Convert back to absolute image coordinates
    abs_x = x_min + local_x
    abs_y = y_min + local_y

    detections.append({
        "timestamp": frame_timestamp,
        "flux": total_weight,
        "sign": sign,
        "x": abs_x,
        "y": abs_y,
        
    })

  return detections


def update_tracks(
    detections,
    tracks,
    next_track_id,
    dt,
    gate_chi2=9.21,
    m_hits=3,
    n_frames=5,
    max_misses=5,
):
  """Step B6: Single-function implementation for Kalman tracking and m-of-n logic."""
  # Predefined matrices for Constant Velocity model
  F = np.array(
      [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]  # State transition
  )
  SIGMA_A = 2.0   # px/s² — how hard a bird may turn; raise if tracks break on maneuvers
  Q = SIGMA_A**2 * np.array([
    [dt**4/4, 0,       dt**3/2, 0      ],
    [0,       dt**4/4, 0,       dt**3/2],
    [dt**3/2, 0,       dt**2,   0      ],
    [0,       dt**3/2, 0,       dt**2  ]])
  H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Measurement matrix
  R = np.eye(2) * 0.5  # Measurement noise

  # --- 1. PREDICT STEP: Move all existing tracks forward ---
  for track in tracks:
    state = np.array([track["x"], track["y"], track["vx"], track["vy"]])
    state = F @ state
    track["P"] = F @ track["P"] @ F.T + Q 
    track["x"], track["y"], track["vx"], track["vy"] = state

  # --- 2. ASSOCIATION & GATING (Nearest Neighbor via Mahalanobis Distance) ---
  unassigned_detections = list(range(len(detections))) # list from 0 to the number of elements-1
  unassigned_tracks = list(range(len(tracks)))
  assignments = []

  if tracks and detections:
    cost_matrix = np.zeros((len(tracks), len(detections)))

    for t_idx, track in enumerate(tracks):
      predicted_pos = np.array([track["x"], track["y"]])
      S = H @ track["P"] @ H.T + R
      S_inv = np.linalg.inv(S)

      for d_idx, det in enumerate(detections):
        meas = np.array([det["x"], det["y"]])
        innovation = meas - predicted_pos
        cost_matrix[t_idx, d_idx] = innovation.T @ S_inv @ innovation

    # Greedy assignment within gating threshold
    while unassigned_tracks and unassigned_detections:
      min_cost = np.inf
      best_t, best_d = -1, -1

      for t in unassigned_tracks:
        for d in unassigned_detections:
          if cost_matrix[t, d] < min_cost:
            min_cost = cost_matrix[t, d]
            best_t, best_d = t, d

      if min_cost > gate_chi2:
        break

      assignments.append((best_t, best_d))
      unassigned_tracks.remove(best_t)
      unassigned_detections.remove(best_d)

  # --- 3. UPDATE ASSIGNED TRACKS ---
  assigned_track_indices = set()
  for t_idx, d_idx in assignments:
    track = tracks[t_idx]
    det = detections[d_idx]
    assigned_track_indices.add(t_idx)

    state = np.array([track["x"], track["y"], track["vx"], track["vy"]])
    meas = np.array([det["x"], det["y"]])

    # Kalman Gain & State Update
    y = meas - (H @ state)
    S = H @ track["P"] @ H.T + R
    K = track["P"] @ H.T @ np.linalg.inv(S)

    state = state + (K @ y)
    track["P"] = (np.eye(4) - (K @ H)) @ track["P"]
    track["x"], track["y"], track["vx"], track["vy"] = state

    track["hits"] += 1
    track["total_frames"] = min(track["total_frames"] + 1, n_frames)
    track["consecutive_misses"] = 0

    track["history"].append({
        "timestamp": det["timestamp"],
        "x": track["x"],
        "y": track["y"],
        "flux": det["flux"],
    })

    if not track["confirmed"] and track["hits"] >= m_hits:
      track["confirmed"] = True

  # --- 4. HANDLE UNASSIGNED TRACKS (Misses) ---
  for t_idx in range(len(tracks)):
    if t_idx not in assigned_track_indices:
      track = tracks[t_idx]
      track["consecutive_misses"] += 1
      track["total_frames"] = min(track["total_frames"] + 1, n_frames)
      track["hits"] = max(0, track["hits"] - 1)

  # --- 5. SPAWN NEW TRACKS FROM UNASSIGNED DETECTIONS ---
  for d_idx in unassigned_detections:
    det = detections[d_idx]
    new_track = {
        "id": next_track_id,
        "x": det["x"],
        "y": det["y"],
        "vx": 0.0,
        "vy": 0.0,
        "P": np.eye(4) * 10.0,
        "hits": 1,
        "total_frames": 1,
        "consecutive_misses": 0,
        "confirmed": False,
        "history": [{
            "timestamp": det["timestamp"],
            "x": det["x"],
            "y": det["y"],
            "flux": det["flux"],
        }],
    }
    tracks.append(new_track)
    next_track_id += 1

  # --- 6. PRUNE DEAD TRACKS ---
  tracks = [
      t
      for t in tracks
      if t["consecutive_misses"] <= max_misses
      and (t["confirmed"] or t["total_frames"] < n_frames or t["hits"] >= 2)
  ]

  # Return updated tracks list, next ID tracker, and only confirmed ones for output
  confirmed_tracks = [t for t in tracks if t["confirmed"]]
  return tracks, next_track_id, confirmed_tracks
#---------------------------------------------------------CALIBRATION-------------------------------------------------  # sum of all
#-----------------------------------------------------CALIBRATION LENS CAP OFF------------------------------------------------------
############################################ MAP #########################################
total = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
total_sq = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

for i in range(FPS * WARMUP_SEC):
    gray, timestamp = grab_frame()
    g = gray.astype(np.float32)
    total += g
    total_sq += g * g

background = total / (FPS * WARMUP_SEC)  # average value for each pixel
variance = total_sq / (FPS * WARMUP_SEC) - background * background  # variance for each pixel
noise = np.sqrt(np.maximum(variance, 0))  # standard deviation for each pixel

######################################################### MASK ############################


sorted_noise = np.sort(noise.flatten())
noise_median = sorted_noise[int(len(sorted_noise)) // 2] #it is not 0-255
MAD_scale = 1.4826 ######################## WHY ########################
noise_MAD = MAD_scale * np.median(np.abs(sorted_noise - noise_median))  
mask = noise > 5*noise_MAD + noise_median  # mask out pixels below the mean noise level


#------------------------------------------------------ blank wall / empty sky---------------------------------------------------------
total = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
for i in range(FPS * WARMUP2_SEC):
    gray, timestamp = grab_frame()
    l = gray.astype(np.float32)
    total += l

flat = total / (FPS * WARMUP2_SEC)  # average value for each pixel
mean_value = np.mean(flat)  # mean value of the flat image
flat = flat / mean_value  # normalize the flat image to have a mean of 1    
# Guard against dead pixels (division by zero later on)
eps = 1e-5
flat = np.clip(flat, eps, None)



####################################### BACKGROUND MODEL ############################################
############################# EWMA (Exponential Weighted Moving Average) ############################ for now I won't use them

l_fast = 1 - math.exp(-1/FPS / SEC)  # fast EWMA factor for foreground detection
l_slow = 1 - math.exp(-1/FPS / SEC2)  # slow EWMA factor for background model

########################################### MEDIAN ##################################################
b_slow = None
b_fast = None
static_count = np.zeros((HEIGHT, WIDTH), dtype=np.float32)   # persistence counter
frame_count = 0

for i in range(FPS * (RUN_SEC + WARMUP3_SEC)):
    gray, timestamp = grab_frame()
    frame = gray.astype(np.float32)

    median_value = np.median(frame[::8, ::8])  # downsample to speed up median calculation

    frame -= median_value    # reaches the same intensity as the rest        # subtract the median value from the image

    frame = frame / flat # flat is 1, it doesn't change the image, if flat is <1, it increases the pixel value, if flat >1, it decreases the pixel value
                         # valorific pixeli care nu prea au noise

    # Isolate pixel changes against the fast background model
    if b_fast is None: b_fast = frame.copy(); b_slow = frame.copy(); continue
        
    d = frame - b_fast   # subtract the average of the last n frames so i obtain only new changes
    d_slow = frame - b_slow
    static_now = (np.abs(d_slow) > K_THRESHOLD * noise) & ~mask

    event_mask = (np.abs(d) > K_THRESHOLD * noise) & ~mask # 1 and 0 if it passes the treshold

    static_count[static_now] += 1.0                 # ← CHANGE 3: grows where change sits still
    static_count[~static_now] = 0.0                 #            resets where the scene is normal

    clutter = static_count > STATIC_N               # ← CHANGE 3: persistent-clutter map
    event_clean = event_mask & ~clutter             # ← CHANGE 3: detection mask minus clutter

    b_fast = np.where(event_mask, b_fast, (1-l_fast)*b_fast + l_fast*frame)    
    b_slow = (1.0 - l_slow) * b_slow + l_slow * frame

    drift = np.median(np.abs(b_fast - b_slow))  # median of the absolute difference between the fast and slow background models
    if drift > 5 * np.median(noise):
        b_fast = frame.copy()  
        b_slow = frame.copy()
        continue  # skip detection for this frame

    if frame_count < WARMUP3_SEC * FPS:
        frame_count += 1
        continue  # skip detection during warmup period

    detections = process_detections(event_clean, d, timestamp) # it returns a list of dictionaries with the detected events

    tracks, next_track_id, confirmed_tracks = update_tracks(detections, tracks, next_track_id, dt)    