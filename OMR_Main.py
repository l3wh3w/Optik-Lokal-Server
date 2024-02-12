import cv2
import numpy as np
import utlis
from static import *
import json


def load_existing_answer_keys():
    json_filename = 'C:/Users/Mehmet/Desktop/local_server/exam_results/all_answer_key.json'

    try:
        with open(json_filename, 'r') as json_file:
            existing_answer_keys = json.load(json_file)
            return existing_answer_keys
    except FileNotFoundError:
        return []

def get_answer_key(exam_code, all_answer_keys):
    for entry in all_answer_keys:
        if entry["exam_code"] == exam_code:
            return entry["answer_key"]
    return None  # Exam code not found
 
def process(time, sinavKodu):
    
# Load existing answer keys at the beginning
    all_answer_keys = load_existing_answer_keys()
 
    ########################################################################
    exam_code = sinavKodu
    answerKey = get_answer_key(exam_code, all_answer_keys)
    ########################################################################
    ########################################################################
    print(exam_code, answerKey)
    # Görüntü dosyasını belirtilen yoldan oku
    path = f'C:/Users/Mehmet/Desktop/local_server/photos/{time}.jpg'
    img = cv2.imread(path)
    img = cv2.resize(img, (widthImg, heightImg))  # Görüntüyü belirtilen genişlik ve yüksekliğe yeniden boyutlandır

    # İşleme geçici olarak kullanılmak üzere görüntüyü kopyala
    imgFinal = img.copy()
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
    tresh_id, warpColored_id = utlis.warpTreshImages(imgTresh, noktalar_id, kare_hw, kare_hw)
    tresh_1, warpColored_1 = utlis.warpTreshImages(imgTresh, noktalar_1, sutun_h, sutun_w)
    tresh_2, warpColored_2 = utlis.warpTreshImages(imgTresh, noktalar_2, sutun_h, sutun_w)
    tresh_3, warpColored_3 = utlis.warpTreshImages(imgTresh, noktalar_3, sutun_h, sutun_w)


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
    myIndex_1 = utlis.user_answers(questions, myPixelVal_1)
    correct_ans1, wrong_ans1, empty1, score1 = utlis.grading(questions, answerKey, myIndex_1)

    # İkinci sütunda işaretli sayıları tespit ediyoruz
    myPixelVal_2 = utlis.pixelVal(questions, choices, boxes_2)
    myIndex_2 = utlis.user_answers(questions, myPixelVal_2)
    correct_ans2, wrong_ans2, empty2, score2 = utlis.grading(questions, answerKey, myIndex_2)

    # Üçüncü sütunda işaretli sayıları tespit ediyoruz
    myPixelVal_3 = utlis.pixelVal(questions, choices, boxes_3)
    myIndex_3 = utlis.user_answers(questions, myPixelVal_3)
    correct_ans3, wrong_ans3, empty3, score3 = utlis.grading(questions, answerKey, myIndex_3)

    total_score = (score1 + score2 + score3) / 3

    ## 2. ve 3. sütunların soru indexi 20 artıyor tüm soruları tek listeye alınca indexleri eklensin diye
    empty2 = [sayi + 20 for sayi in empty2]; empty3 = [sayi + 40 for sayi in empty3]
    wrong_ans2 = [sayi + 20 for sayi in wrong_ans2]; wrong_ans3 = [sayi + 40 for sayi in wrong_ans3]
  

    # 1. sütunun işaretli şıkları ve doğru cevapları resimde gösteriyoruz
    utlis.showAnswers(warpColored_1, myIndex_1, correct_ans1, answerKey, questions, choices)  # işaretli şıkları göster
    utlis.drawGrid(warpColored_1, questions, choices)  # ızgara çiz
    RawDrawings_1 = np.zeros_like(warpColored_1)  # warp image boyutunda boş bir resim oluştur
    utlis.showAnswers(RawDrawings_1, myIndex_1, correct_ans1, answerKey, questions, choices)  # cevapları bu resme ekle
    imgInvWarp_1 = utlis.invWarpImage(RawDrawings_1, noktalar_1, sutun_h, sutun_w, heightImg,
                                      widthImg)  # resmi tekrar ters çevir
    utlis.drawGrid(RawDrawings_1, questions, choices)  # ızgara çiz
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp_1, 1, 0)

    # 2. sütunun  işaretli şıkları ve doğru cevapları resimde gösteriyoruz
    utlis.showAnswers(warpColored_2, myIndex_2, correct_ans2, answerKey, questions, choices)  # işaretli şıkları göster
    utlis.drawGrid(warpColored_2, questions, choices)  # ızgara çiz
    RawDrawings_2 = np.zeros_like(warpColored_2)  # warp image boyutunda boş bir resim oluştur
    utlis.showAnswers(RawDrawings_2, myIndex_2, correct_ans2, answerKey, questions, choices)  # # cevapları bu resme ekle
    imgInvWarp_2 = utlis.invWarpImage(RawDrawings_2, noktalar_2, sutun_h, sutun_w, heightImg,
                                      widthImg)  # resmi tekrar ters çevir
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp_2, 1, 0)

    # 3. sütunun işaretli şıkları ve doğru cevapları resimde gösteriyoruz
    utlis.showAnswers(warpColored_3, myIndex_3, correct_ans3, answerKey, questions, choices)  # işaretli şıkları göster
    utlis.drawGrid(tresh_3, questions, choices)  # ızgara çiz
    RawDrawings_3 = np.zeros_like(warpColored_3)  # warp image boyutunda boş bir resim oluştur
    utlis.showAnswers(RawDrawings_3, myIndex_3, correct_ans3, answerKey, questions, choices)  # # cevapları bu resme ekle
    imgInvWarp_3 = utlis.invWarpImage(RawDrawings_3, noktalar_3, sutun_h, sutun_w, heightImg,
                                      widthImg)  # resmi tekrar ters çevir

    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp_3, 1, 0)
    cv2.imwrite(f'C:/Users/Mehmet/Desktop/local_server/final_photos/{student_number}.jpg',
                imgFinal)

  ################################################################  
    
    all_answers = correct_ans1 + correct_ans2 + correct_ans3 ##doğruların olduğu listeleri birleştirip tek bir liste yapıyoruz
    all_correct = [i for i, value in enumerate(all_answers) if value == 1] ##all answers listesindeki 1 lerin indexi doğru sayıların listesi oluyor.
    all_correct = [sayi + 1 for sayi in all_correct]  ## soru sayısı index sayısının 1 fazlası olduğu için
    all_empty = empty1 + empty2 + empty3
    all_wrong = wrong_ans1 + wrong_ans2 + wrong_ans3
   
  ################################################################
            
    response = {"student_id": student_number}
    response["all_correct"] = all_correct
    response["all_wrong"] = all_wrong
    response["all_empty"] = all_empty
    response["total_score"] = total_score
    response["raw_ans_file"] = f'static/opticsPhotos/rawFiles/{time}.jpg'
    response["final_ans_file"] = f'static/opticsPhotos/finalFiles/{time}.jpg'

    response = json.dumps(response)
    return response


