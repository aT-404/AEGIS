import cv2
import numpy as np
from pathlib import Path


def extract_person_sprites():
    """Extracts realistic person crops from Ultralytics assets."""
    assets_dir = Path(r"C:\Users\alber\AppData\Local\Programs\Python\Python313\Lib\site-packages\ultralytics\assets")
    zidane = cv2.imread(str(assets_dir / "zidane.jpg"))
    bus = cv2.imread(str(assets_dir / "bus.jpg"))

    sprites = []
    # Crop from zidane: right person
    sprites.append(zidane[50:680, 750:1120])
    # Crop from bus: pedestrian 1
    sprites.append(bus[230:720, 50:240])
    # Crop from bus: pedestrian 2
    sprites.append(bus[240:700, 220:340])
    # Crop from bus: pedestrian 3
    sprites.append(bus[380:720, 670:800])

    return sprites


def overlay_sprite(background, sprite, x, y, scale=1.0):
    """Overlays a sprite onto background with boundary clipping."""
    h_bg, w_bg = background.shape[:2]
    sh, sw = sprite.shape[:2]
    target_w = max(20, int(sw * scale))
    target_h = max(40, int(sh * scale))
    resized = cv2.resize(sprite, (target_w, target_h))

    x1, y1 = int(x), int(y)
    x2, y2 = x1 + target_w, y1 + target_h

    # Clip coordinates to background
    bx1 = max(0, x1)
    by1 = max(0, y1)
    bx2 = min(w_bg, x2)
    by2 = min(h_bg, y2)

    if bx2 <= bx1 or by2 <= by1:
        return

    sx1 = bx1 - x1
    sy1 = by1 - y1
    sx2 = sx1 + (bx2 - bx1)
    sy2 = sy1 + (by2 - by1)

    # Simple alpha blend or direct overwrite
    background[by1:by2, bx1:bx2] = resized[sy1:sy2, sx1:sx2]


def create_factory_floor(output_path: str, duration_sec: int = 12, fps: int = 30):
    """1280x720 @ 30 FPS: Factory floor with R-01 (Intrusion) and R-02 (Loitering)."""
    w, h = 1280, 720
    sprites = extract_person_sprites()
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for f in range(total_frames):
        # Industrial concrete floor background
        frame = np.full((h, w, 3), (110, 115, 120), dtype=np.uint8)
        # Floor safety markings and grid
        for gy in range(0, h, 80):
            cv2.line(frame, (0, gy), (w, gy), (95, 100, 105), 1)
        for gx in range(0, w, 100):
            cv2.line(frame, (gx, 0), (gx, h), (95, 100, 105), 1)

        # Worker 1: walks across into R-01 (Machinery Exclusion Zone: [120, 250] to [480, 600])
        # Trajectory enters R-01 around frame 30
        w1_x = 50 + (f * 3.5)
        w1_y = 350
        overlay_sprite(frame, sprites[0], w1_x, w1_y, scale=0.6)

        # Worker 2: stands motionless inside R-02 ([310, 200] to [640, 580]) to trigger Loitering
        w2_x = 450 + int(np.sin(f * 0.05) * 4)  # slight breathing jitter
        w2_y = 300
        overlay_sprite(frame, sprites[1], w2_x, w2_y, scale=0.7)

        out.write(frame)

    out.release()
    print(f"Created {output_path} ({total_frames} frames, {fps} fps)")


def create_entrance_gate(output_path: str, duration_sec: int = 8, fps: int = 25):
    """1920x1080 @ 25 FPS: Entrance gate with crowd convergence into GATE-01."""
    w, h = 1920, 1080
    sprites = extract_person_sprites()
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for f in range(total_frames):
        frame = np.full((h, w, 3), (140, 145, 150), dtype=np.uint8)
        # Gate barriers
        cv2.rectangle(frame, (500, 250), (900, 750), (120, 125, 130), -1)

        # 6 pedestrians walking through gate zone (GATE-01: [550, 300] to [850, 720])
        for idx in range(6):
            sp = sprites[idx % len(sprites)]
            base_x = 560 + (idx * 45)
            y = 320 + ((f * (3 + idx)) % 380)
            overlay_sprite(frame, sp, base_x, y, scale=0.8)

        out.write(frame)

    out.release()
    print(f"Created {output_path} ({total_frames} frames, {fps} fps)")


def create_warehouse(output_path: str, duration_sec: int = 6, fps: int = 15):
    """640x480 @ 15 FPS: Low-res warehouse surveillance."""
    w, h = 640, 480
    sprites = extract_person_sprites()
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for f in range(total_frames):
        frame = np.full((h, w, 3), (85, 90, 95), dtype=np.uint8)
        # Shelves / racks
        cv2.rectangle(frame, (20, 20), (160, 460), (50, 55, 60), -1)
        cv2.rectangle(frame, (480, 20), (620, 460), (50, 55, 60), -1)

        # Worker walking down center aisle
        w_x = 260
        w_y = 50 + int((f / total_frames) * 320)
        overlay_sprite(frame, sprites[2 % len(sprites)], w_x, w_y, scale=0.45)

        out.write(frame)

    out.release()
    print(f"Created {output_path} ({total_frames} frames, {fps} fps)")


def create_outdoor_perimeter(output_path: str, duration_sec: int = 8, fps: int = 30):
    """1280x720 @ 30 FPS: Outdoor perimeter fence line."""
    w, h = 1280, 720
    sprites = extract_person_sprites()
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for f in range(total_frames):
        # Ground and sky
        frame = np.full((h, w, 3), (70, 100, 80), dtype=np.uint8)
        frame[:260, :] = (180, 150, 120)  # sky

        # Perimeter fence line
        cv2.line(frame, (0, 400), (w, 400), (160, 160, 160), 3)

        # Intruder walking along fence
        p_x = 100 + (f * 4)
        p_y = 320
        overlay_sprite(frame, sprites[3 % len(sprites)], p_x, p_y, scale=0.65)

        out.write(frame)

    out.release()
    print(f"Created {output_path} ({total_frames} frames, {fps} fps)")


if __name__ == "__main__":
    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    create_factory_floor("data/samples/factory_floor_sample.mp4")
    create_entrance_gate("data/samples/entrance_gate_sample.mp4")
    create_warehouse("data/samples/warehouse_sample.mp4")
    create_outdoor_perimeter("data/samples/outdoor_perimeter_sample.mp4")
    print("All sample videos generated successfully.")
