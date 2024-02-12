import cv2
import numpy as np
import utlis
from static import *
import json


# Initialize an empty list to store answer keys
all_answer_keys = []

def load_existing_answer_keys():
    json_filename = 'C:/Users/Mehmet/Desktop/local_server/exam_results/all_answer_key.json'

    try:
        with open(json_filename, 'r') as json_file:
            existing_answer_keys = json.load(json_file)
            return existing_answer_keys
    except FileNotFoundError:
        return []

# Load existing answer keys at the beginning
all_answer_keys = load_existing_answer_keys()

def saveAnswerKey(time, sinavKodu):
    try:
        ########################################################################
        path = f'C:/Users/Mehmet/Desktop/local_server/answer_key/{time}.jpg'
        img = cv2.imread(path)
        # Görüntü dosyasını belirtilen yoldan oku
        # path = "14.png"
        # path = f'C:/Users/Mehmet/Desktop/local_server/photos/{time}.jpg'
        img = cv2.resize(img, (widthImg, heightImg))  # Görüntüyü belirtilen genişlik ve yüksekliğe yeniden boyutlandır

        # İşleme geçici olarak kullanılmak üzere görüntüyü kopyala
        imgContours = img.copy()
        imgTresh = img.copy()

        ################################################################
        # Görüntüyü işlenmeye uygun hale getir
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Görüntüyü gri tonlamaya dönüştür
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Görüntüye Gaussian bulanıklığı uygula
        imgCanny = cv2.Canny(imgBlur, 10, 70)  # Canny kenar algılama uygula

        # Görüntüdeki konturları bul
        contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Konturları bul
        cv2.drawContours(imgContours, contours, -1, yesil, 3)  # Resmin konturlarını çiz
        ########################################################################
        # cv2.imshow("Split Test 35", imgContours)
        ## konturlardan öğrenci no ve şıkların olduğu kutuları buluyoruz.
        sutunlar_ve_kare = utlis.sutunlar_ve_kare(contours)

        # öğrenci no ve şıkların olduğu kutuların köşe noktalarını buluyoruz.
        noktalar_id = utlis.getCornerPoints(sutunlar_ve_kare[0])
        noktalar_1 = utlis.getCornerPoints(sutunlar_ve_kare[1])
        noktalar_2 = utlis.getCornerPoints(sutunlar_ve_kare[2])
        noktalar_3 = utlis.getCornerPoints(sutunlar_ve_kare[3])

        # ## her bir sütunun çekim açısı kaynaklı yamukluğunu düzeltip, siyah-beyaz yapıyoruz
        tresh_id, _ = utlis.warpTreshImages(imgTresh, noktalar_id, kare_hw, kare_hw)
        tresh_1, _ = utlis.warpTreshImages(imgTresh, noktalar_1, sutun_h, sutun_w)
        tresh_2, _ = utlis.warpTreshImages(imgTresh, noktalar_2, sutun_h, sutun_w)
        tresh_3, _ = utlis.warpTreshImages(imgTresh, noktalar_3, sutun_h, sutun_w)


        ##siyah beyaz resimleri, her bir şıkka bölüyoruz.
        # öğrenci no 10x10, soru şıkları questions*choices kadar kutu oluşturuyor.
        boxes_id = utlis.splitBoxes(tresh_id, 10, 10)  # GET INDIVIDUAL BOXES
        boxes_1 = utlis.splitBoxes(tresh_1, questions, choices)  #
        boxes_2 = utlis.splitBoxes(tresh_2, questions, choices)  #
        boxes_3 = utlis.splitBoxes(tresh_3, questions, choices)  #

        ###########################################

        # öğrenci no da işaretli sayıları tespit ediyoruz
        myPixelVal_id = utlis.pixelVal(10, 10, boxes_id)
        myPixelVal_id = utlis.id_reorder(myPixelVal_id)
        student_id = utlis.id_answers(10, myPixelVal_id)
        student_number = ''
        for i in student_id:
            student_number += str(i)

    
        # Birinci sütunda işaretli sayıları tespit ediyoruz
        myPixelVal_1 = utlis.pixelVal(questions, choices, boxes_1)
        answerKey_1 = utlis.answerKey(questions, myPixelVal_1)
        
        
        # İkinci sütunda işaretli sayıları tespit ediyoruz
        myPixelVal_2 = utlis.pixelVal(questions, choices, boxes_2)
        answerKey_2 = utlis.answerKey(questions, myPixelVal_2)
 
        # Üçüncü sütunda işaretli sayıları tespit ediyoruz
        myPixelVal_3 = utlis.pixelVal(questions, choices, boxes_3)
        answerKey_3 = utlis.answerKey(questions, myPixelVal_3)

        answerKey = answerKey_1 + answerKey_2 + answerKey_3
        
    ################################################################  

                
        answerKey = [int(value) for value in answerKey]

        json_filename = 'C:/Users/Mehmet/Desktop/local_server/exam_results/all_answer_key.json'

        # Check if sinavKodu already exists in the list or in any other list in the JSON file
        existing_codes = [item["exam_code"] for item in all_answer_keys]
        if sinavKodu in existing_codes:
            return f"Sınav kodu {sinavKodu} zaten mevcut. Lütfen başka bir kod kullanın."

   
        # Append the current answer key to the list
        exam_info = {
            "exam_code": sinavKodu,
            "answer_key": answerKey
        }
        all_answer_keys.append(exam_info)

        # Save the combined list back to the JSON file
        with open(json_filename, 'w') as json_file:
            json.dump(all_answer_keys, json_file, default=str)

        return f"Sınav bilgisi {json_filename} dosyasına kaydedildi."

    except Exception as e:
        print(f"Hata: {e}")
        return "Bir hata oluştu"
    
        
        

    