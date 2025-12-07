import os
from PIL import Image

def save_augmented_images_fixed_zoom(past_dir, present_dir, out_past_dir, out_present_dir, zoom_levels=4, zoom_step=0.15):
    os.makedirs(out_past_dir, exist_ok=True)
    os.makedirs(out_present_dir, exist_ok=True)

    files = [f for f in os.listdir(past_dir) if f in os.listdir(present_dir) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"พบ {len(files)} คู่ภาพ")

    for fname in files:
        past_img = Image.open(os.path.join(past_dir, fname)).convert("RGB")
        present_img = Image.open(os.path.join(present_dir, fname)).convert("RGB")

          บันทึกภาพต้นฉบับ
        past_img.save(os.path.join(out_past_dir, fname))
        present_img.save(os.path.join(out_present_dir, fname))

          สร้างภาพ augmented ด้วยระดับการซูมที่คงที่
        for i in range(zoom_levels):
            zoom_scale = 1 + (i + 1) * zoom_step    เริ่มจาก 1.15, 1.30, 1.45, ...
            width, height = past_img.size
            new_width, new_height = int(width * zoom_scale), int(height * zoom_scale)

            aug_past = past_img.resize((new_width, new_height), Image.Resampling.LANCZOS).crop((0, 0, width, height))
            aug_present = present_img.resize((new_width, new_height), Image.Resampling.LANCZOS).crop((0, 0, width, height))

            aug_past.save(os.path.join(out_past_dir, f"{os.path.splitext(fname)[0]}_zoom{i+1}.png"))
            aug_present.save(os.path.join(out_present_dir, f"{os.path.splitext(fname)[0]}_zoom{i+1}.png"))
    print("บันทึกข้อมูล augmented เสร็จแล้ว")

  เรียกใช้ฟังก์ชัน
save_augmented_images_fixed_zoom("Past", "Present", "Past_Augmented", "Present_Augmented", zoom_levels=4, zoom_step=0.15)