import os
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Папки
input_folder = "img"  # Папка зображень
output_folder = "txt"  # Папка для текстових файлів
debug_folder = "debug"  # Папка для збереження проміжних результатів

# Налаштування для розпізнавання тексту
custom_config = r'--oem 3 --psm 4 -l ukr+rus+eng'

# Переконайтеся, що вихідні папки існують
for folder in [output_folder, debug_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def preprocess_image(image):
    """Більш м'яка обробка зображення: збереження природності."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Перетворення в градації сірого

    # Невелике розмиття для зменшення шуму
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Підвищення видимості тексту через покращення контрасту
    alpha = 1.0  # Контраст
    beta = 30    # Яскравість
    enhanced = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)

    return enhanced

def deskew_image(image):
    """Вирівнює нахилене зображення."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated

def save_debug_image(image, filename):
    """Зберігає проміжний результат для діагностики."""
    debug_path = os.path.join(debug_folder, filename)
    cv2.imwrite(debug_path, image)

def replace_words(text):
    """Замінює слова 'меп' на 'мсп' та 'месет' на 'несет'"""
    replacements = {
        "меп": "мсп",
        "вісл": "в/сл",
        "омебр": "омcбр",
        "мед": "мсд"
    }
    for old_word, new_word in replacements.items():
        text = text.replace(old_word, new_word)
    return text

# Прохід по всіх файлах у папці input_folder
for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)

    # Перевірка, чи файл є зображенням
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        try:
            # Завантаження зображення
            image = cv2.imread(file_path)

            # Вирівнювання зображення
            deskewed_image = deskew_image(image)

            # Попередня обробка
            preprocessed_image = preprocess_image(deskewed_image)

            # Збереження проміжного результату
            save_debug_image(preprocessed_image, f"processed_{file_name}")

            # Перетворення в формат PIL для Tesseract
            pil_image = Image.fromarray(preprocessed_image)

            # Отримання тексту з зображення
            text = pytesseract.image_to_string(pil_image, config=custom_config)

            # Заміна слів у тексті
            text = replace_words(text)

            # Формування шляху до текстового файлу
            text_file_name = os.path.splitext(file_name)[0] + ".txt"
            text_file_path = os.path.join(output_folder, text_file_name)

            # Збереження тексту у файл
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(text)

            print(f"Оброблено: {file_name} -> {text_file_name}")

        except Exception as e:
            print(f"Помилка при обробці файлу {file_name}: {e}")

print("Обробка завершена.")





