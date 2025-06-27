import re

def predict_number_plate(img, ocr):
    try:
        result = ocr.ocr(img)
        print("[DEBUG OCR]", result)

        if not result or not isinstance(result[0], list) or len(result[0]) == 0:
            return None, None

        texts = []
        scores = []

        for line in result[0]:
            if len(line) >= 2:
                text, score = line[1]
                texts.append(text)
                scores.append(score)

        if not texts or not scores:
            return None, None

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_text = texts[best_idx]
        confidence = scores[best_idx]

        cleaned_text = re.sub(r'[^A-Z0-9]', '', best_text.upper())

        if confidence * 100 >= 60 and 6 <= len(cleaned_text) <= 12:
            return cleaned_text, confidence
        else:
            return None, None

    except Exception as e:
        print("[ERROR] OCR failed:", e)
        return None, None
