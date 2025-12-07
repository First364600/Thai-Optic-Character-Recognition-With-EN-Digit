import os
import subprocess
import whisper
import torch
import psutil
import time
from threading import Thread
import gc

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

def monitor_gpu_usage():
    """Monitor GPU usage in real-time"""
    while True:
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"\rGPU Memory - Allocated: {memory_allocated:.2f}GB | Reserved: {memory_reserved:.2f}GB | Total: {memory_total:.2f}GB", end="")
        time.sleep(2)    ลดความถี่การ update

def convert_mp4_to_mp3(mp4_file_path, mp3_file_path=None):
    """
    แปลงไฟล์ MP4 เป็น MP3 โดยใช้ ffmpeg (optimized)
    """
    if mp3_file_path is None:
        mp3_file_path = mp4_file_path.replace('.mp4', '.mp3')
    
      ใช้ ffmpeg แปลงไฟล์ด้วยการตั้งค่าที่เหมาะสม
    command = [
        'ffmpeg',
        '-i', mp4_file_path,
        '-vn',                  ไม่ต้องการ video
        '-acodec', 'libmp3lame',    ใช้ MP3 codec
        '-ab', '320k',          เพิ่ม bitrate เป็น 320k สำหรับคุณภาพที่ดีขึ้น
        '-ar', '44100',         เพิ่ม sample rate เป็น 44.1kHz
        '-ac', '2',             stereo channel สำหรับคุณภาพเสียงที่ดีขึ้น
        '-y',                   overwrite
        mp3_file_path
    ]
    
    try:
        print("กำลังแปลง MP4 เป็น MP3...")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"แปลงสำเร็จ: {mp3_file_path}")
        return mp3_file_path
    except subprocess.CalledProcessError as e:
        print(f"เกิดข้อผิดพลาดในการแปลงไฟล์: {e}")
        print(f"Error output: {e.stderr}")
        raise
    except FileNotFoundError:
        print("ไม่พบ ffmpeg โปรดติดตั้ง ffmpeg ก่อนใช้งาน")
        print("ดาวน์โหลดได้จาก: https://ffmpeg.org/download.html")
        raise

def transcribe_audio_with_whisper_optimized(audio_file_path, model_name="large-v2"):
    """
    ใช้ Whisper แปลงเสียงเป็นข้อความ (optimized for maximum GPU usage)
    """
      เลือก device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nใช้ device: {device}")
    
      ล้าง cache ก่อนโหลดโมเดล
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
      โหลดโมเดลขนาดใหญ่สุดที่ VRAM รองรับได้
    print(f"กำลังโหลดโมเดล {model_name}...")
    model = whisper.load_model(model_name, device=device)
    
      แสดงการใช้ memory หลังโหลดโมเดล
    if torch.cuda.is_available():
        print(f"\nVRAM ใช้ไปหลังโหลดโมเดล: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
      ตั้งค่าการแปลงที่เหมาะสม สำหรับใช้ VRAM เต็มที่
    options = {
        "language": None,    ให้โมเดลตรวจจับภาษาเอง (ใช้ computation มากขึ้น)
        "task": "transcribe",
        "fp16": False,    ใช้ fp32 เพื่อใช้ VRAM มากขึ้น (แต่แม่นยำกว่า)
        "verbose": False,    ปิด verbose เพื่อลด output ที่ไม่จำเป็น
        "beam_size": 10,    เพิ่มเป็น 10 (ใช้ memory มากขึ้น)
        "best_of": 10,      เพิ่มเป็น 10 (ใช้ memory มากขึ้น)
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),    หลายอุณหภูมิ (ใช้ computation มากขึ้น)
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,    ใช้ context จากประโยคก่อนหน้า
    }
    
      แปลงเสียงเป็นข้อความ
    print("กำลังแปลงเสียงด้วยการตั้งค่าที่ใช้ VRAM เต็มที่...")
    print("นี่อาจใช้เวลานานกว่าปกติเพื่อความแม่นยำสูงสุด...")
    
    result = model.transcribe(audio_file_path, **options)
    
      แสดงการใช้ memory หลังการแปลง
    if torch.cuda.is_available():
        print(f"\nVRAM สูงสุดที่ใช้: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    return result

def chunk_audio_processing(audio_file_path, model_name="large-v2", chunk_duration=300):
    """
    แบ่งไฟล์เสียงเป็นชิ้นเล็กๆ แล้วประมวลผลทีละชิ้น เพื่อใช้ VRAM เต็มที่
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
      โหลดโมเดลครั้งเดียว
    model = whisper.load_model(model_name, device=device)
    
      ใช้ ffprobe เพื่อหาความยาวของไฟล์
    command = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
               '-of', 'csv=p=0', audio_file_path]
    result = subprocess.run(command, capture_output=True, text=True)
    total_duration = float(result.stdout.strip())
    
    print(f"ความยาวไฟล์เสียง: {total_duration/60:.2f} นาที")
    print(f"จะแบ่งเป็น chunks ละ {chunk_duration/60:.1f} นาที")
    
      แบ่งไฟล์และประมวลผลทีละชิ้น
    all_segments = []
    chunk_count = int(total_duration / chunk_duration) + 1
    
    for i in range(chunk_count):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, total_duration)
        
        if start_time >= total_duration:
            break
            
        print(f"\nประมวลผล chunk {i+1}/{chunk_count} ({start_time/60:.1f}-{end_time/60:.1f} นาที)")
        
          สร้าง chunk ไฟล์ชั่วคราว
        chunk_file = f"temp_chunk_{i}.mp3"
        
          ตัดไฟล์เสียง
        command = [
            'ffmpeg', '-y', '-i', audio_file_path,
            '-ss', str(start_time), '-t', str(end_time - start_time),
            '-c', 'copy', chunk_file
        ]
        subprocess.run(command, capture_output=True)
        
          ตั้งค่าการแปลงที่ใช้ VRAM เต็มที่
        options = {
            "language": None,
            "task": "transcribe", 
            "fp16": False,    ใช้ fp32
            "beam_size": 15,    เพิ่มขึ้นอีก
            "best_of": 15,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        }
        
          แปลง chunk นี้
        chunk_result = model.transcribe(chunk_file, **options)
        
          ปรับ timestamp ให้ตรงกับเวลาจริง
        for segment in chunk_result['segments']:
            segment['start'] += start_time
            segment['end'] += start_time
            all_segments.append(segment)
        
          ลบไฟล์ชั่วคราว
        os.remove(chunk_file)
        
          แสดงการใช้ VRAM
        if torch.cuda.is_available():
            print(f"VRAM ใช้: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
      รวมผลลัพธ์
    full_text = " ".join([segment['text'] for segment in all_segments])
    
    return {
        'text': full_text,
        'segments': all_segments
    }

def mp4_to_text_max_performance(mp4_file_path, model_name="large-v2", keep_mp3=False, use_chunking=True):
    """
    แปลง MP4 เป็นข้อความ (maximum performance version)
    """
      เริ่ม monitor GPU usage ใน background
    monitor_thread = Thread(target=monitor_gpu_usage, daemon=True)
    monitor_thread.start()
    
      ล้าง GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    start_time = time.time()
    
    print(f"กำลังแปลง {mp4_file_path} เป็น MP3 คุณภาพสูง...")
    mp3_file_path = convert_mp4_to_mp3(mp4_file_path)
    
    if use_chunking:
        print(f"ใช้วิธี chunking เพื่อใช้ VRAM เต็มที่...")
        result = chunk_audio_processing(mp3_file_path, model_name, chunk_duration=180)    3 นาทีต่อ chunk
    else:
        print(f"ใช้วิธีปกติแต่เต็มประสิทธิภาพ...")
        result = transcribe_audio_with_whisper_optimized(mp3_file_path, model_name)
    
      ลบไฟล์ MP3 หากไม่ต้องการเก็บ
    if not keep_mp3:
        os.remove(mp3_file_path)
        print(f"ลบไฟล์ {mp3_file_path} แล้ว")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nใช้เวลาทั้งหมด: {processing_time:.2f} วินาที ({processing_time/60:.2f} นาที)")
    
      แสดงสถิติการใช้ VRAM
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"VRAM สูงสุดที่ใช้: {max_memory:.2f} GB")
        print(f"การใช้ VRAM: {(max_memory/6)*100:.1f}%")
        torch.cuda.empty_cache()
        print(f"VRAM หลังล้าง cache: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return result

def save_transcript_to_file(result, output_file="transcript.txt"):
    """
    บันทึกผลลัพธ์เป็นไฟล์ text
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Full Transcript ===\n")
        f.write(result['text'])
        f.write(f"\n\n=== Detailed Segments ===\n")
        
        for i, segment in enumerate(result['segments']):
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            f.write(f"[{start_time:.2f}s - {end_time:.2f}s]: {text}\n")
    
    print(f"บันทึกผลลัพธ์ลงไฟล์: {output_file}")

  ตัวอย่างการใช้งาน
if __name__ == "__main__":
      กำหนด path ของไฟล์ MP4
    mp4_file = "C:\\Users\\ChxraWish\\Downloads\\Lec_LRU_LFU_MFU_EAT-Fri_3OCT2025.mp4"
    
      ตรวจสอบว่าไฟล์มีอยู่จริง
    if os.path.exists(mp4_file):
          เลือกโมเดลตามขนาด VRAM แบบเต็มประสิทธิภาพ
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        
          เนื่องจากมี VRAM 6GB ให้ใช้โมเดล large เต็มที่
        model_choice = "base"    ใช้ large-v2 แทน large-v3 เพื่อประสิทธิภาพ
        print(f"ใช้โมเดล {model_choice} เพื่อใช้ VRAM 6GB เต็มที่")
        
          แปลงและรับผลลัพธ์ด้วยการตั้งค่าเต็มประสิทธิภาพ
        result = mp4_to_text_max_performance(
            mp4_file, 
            model_name=model_choice, 
            keep_mp3=True,
            use_chunking=True    ใช้ chunking เพื่อใช้ VRAM เต็มที่
        )
        
          บันทึกผลลัพธ์ลงไฟล์
        output_filename = mp4_file.replace('.mp4', '_transcript_optimized.txt')
        save_transcript_to_file(result, output_filename)
        
        # แสดงผลลัพธ์สั้นๆ
        print("\n" + "="*50)
        print("--- ผลลัพธ์การแปลงเสียง (Optimized) ---")
        print(f"ข้อความทั้งหมด ({len(result['text'])} ตัวอักษร):")
        print(f"{result['text'][:1000]}..." if len(result['text']) > 1000 else result['text'])
        
        print(f"\nจำนวน segments: {len(result['segments'])}")
        print(f"ไฟล์ผลลัพธ์: {output_filename}")
        
    else:
        print(f"ไม่พบไฟล์ {mp4_file}")