import subprocess
import atexit
from pathlib import Path
from datetime import datetime

def start_rtsp_recording(rtsp_url: str, log_file_path: str = "ffmpeg_recording.log") -> subprocess.Popen:
    """
    Directly spawns an FFmpeg background process to record an RTSP stream.
    Replaces the external .sh script entirely.
    """
    # 1. Replicate bash logic: Ensure output ends in .mp4
    output_path = datetime.now().strftime("%d_%m_%Y__%H_%M")
    out_path = Path(output_path)
    if out_path.suffix.lower() != '.mp4':
        out_path = out_path.with_suffix('.mp4')
    
    # Ensure parent directories exist
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Starting recording...")
    print(f"Source: {rtsp_url}")
    print(f"Destination: {out_path}")

    # 2. Replicate bash logic: Construct the exact FFmpeg command
    # -rtsp_transport tcp: Forces TCP instead of UDP to prevent packet loss
    # -c:v copy: Copies the video stream without re-encoding (ultra-low CPU usage)
    # -an: Drops audio (matching your bash script's '-an' flag)
    command = [
        "ffmpeg",
        "-y",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-map", "0:v:0",
        "-c", "copy",
        str(out_path)
    ]

    # 3. Open a log file for FFmpeg output so it doesn't block memory or flood the console
    log_file = open(log_file_path, "w")
    
    # 4. Spawn FFmpeg in the background
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=log_file,
        text=True
    )
    
    print(f"🎥 Recording started in background via FFmpeg (PID: {process.pid})")
    print(f"📋 Logs are being written to: {log_file_path}")
    
    # --- Cleanup Functions ---
    def cleanup():
        print(f"\n🛑 Python exiting. Safely terminating FFmpeg process (PID: {process.pid})...")
        # FFmpeg expects a standard termination signal to cleanly finalize the MP4 container
        process.terminate() 
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill() # Force kill if it hangs
        finally:
            log_file.close() # Ensure file handle is closed

    # Register the cleanup to run automatically on exit
    atexit.register(cleanup)
    
    return process

