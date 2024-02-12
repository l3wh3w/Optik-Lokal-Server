import cv2
import numpy as np

yesil = (0, 255, 0);
kirmizi = (0, 0, 255)


## konturlardan öğrenci no ve şıkların olduğu kutuları buluyoruz.
def sutunlar_ve_kare(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 150 :
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)

    # Dikdörtgenleri sırala, x koordinatına göre
    rectCon = sorted(rectCon, key=lambda x: cv2.boundingRect(x)[0])

    # Kareyi bul ve sıralanan listeye ekle
    square_index = find_closest_square(rectCon)
    if square_index is not None:
        rectCon.insert(0, rectCon.pop(square_index))

    return rectCon


# Öğrenci numarası kutusunun tespiti
def find_closest_square(rectangles):
    min_difference = float('inf')
    closest_square_index = None

    for i, contour in enumerate(rectangles):
        rect = cv2.boundingRect(contour)
        # Kareye en yakın dikdörtgeni bul
        difference = abs(rect[3] - rect[2])
        if difference < min_difference:
            min_difference = difference
            closest_square_index = i

    return closest_square_index


# köşe noktalarının yaklaşık tespiti
def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)  # Konturun uzunluğunu hesapla
    koseler = cv2.approxPolyDP(cont, 0.02 * peri, True)  # Poligonu köşe noktalarını elde etmek için yaklaşık olarak çöz

    return koseler


def warpTreshImages(img, noktalar, sutun_h, sutun_w):
    noktalar = reorder_padding(noktalar)  # köşe noktalarını yeniden düzenle
    cv2.drawContours(img, noktalar, -1, (0, 255, 0), 5)  # konturu çiz

    # Köşe noktalarını belirtmek için iki farklı nokta kümesi oluşturun
    pts1 = np.float32(noktalar)  # Konturun köşe noktalarını içerir
    pts2 = np.float32(
        [[0, 0], [sutun_w, 0], [0, sutun_h], [sutun_w, sutun_h]])  # Dönüştürülmüş görüntünün dört köşesini belirtir

    # Perspektif dönüşüm matrisini oluşturuyoruz
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Orijinal görüntüyü perspektif dönüşüm matrisi ile dönüştürüyoruz
    imgWarpColored = cv2.warpPerspective(img, matrix, (sutun_w, sutun_h))
    # kenardaki siyah çerçeveyi siliyoruz
    # imgWarpColored = imgWarpColored[5:-5, 5:-5]
    # Renkli dönüştürülmüş görüntüyü gri tonlamaya dönüştürüyoruz
    imgWarGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    # Gri tonlamalı görüntüyü eşikle ve tersine çevirme işlemi uyguluyoruz

    imgTresh = cv2.adaptiveThreshold(imgWarGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)

    return imgTresh, imgWarpColored


def invWarpImage(imgRawDrawings, noktalar, sutun_h, sutun_w, heightImg, widthImg):
    noktalar = reorder_padding(noktalar)
    pts1 = np.float32(noktalar)
    pts2 = np.float32([[0, 0], [sutun_w, 0], [0, sutun_h], [sutun_w, sutun_h]])
    invMatrix = cv2.getPerspectiveTransform(pts2, pts1)  # Tersine Perspektif dönüşüm matrisini oluşturun
    imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix,
                                     (widthImg, heightImg))  # perspektif matrisini ters çevir

    return imgInvWarp


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))  # Ekstra parantezi kaldır
    # print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32)  # düzenlenmiş değerlerle yeni matrix
    add = myPoints.sum(1)
    # print(add)
    # print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  # [0,0]
    myPointsNew[3] = myPoints[np.argmax(add)]  # [w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # [w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)]  # [h,0]

    return myPointsNew


def reorder_padding(myPoints, padding=5):
    myPoints = myPoints.reshape((4, 2))

    # Toplam ve fark hesaplamalarını güncelle
    add = myPoints.sum(1)
    diff = np.diff(myPoints, axis=1)

    # En küçük toplam ve en büyük farkı içeren indeksleri bul
    topLeftIndex = np.argmin(add)
    bottomRightIndex = np.argmax(add)
    bottomLeftIndex = np.argmin(diff)
    topRightIndex = np.argmax(diff)

    # Köşelerin içe kaydırılmış versiyonlarını oluştur
    topLeft = myPoints[topLeftIndex] + [padding, padding]
    bottomRight = myPoints[bottomRightIndex] - [padding, padding]
    bottomLeft = myPoints[bottomLeftIndex] - [padding, -padding]
    topRight = myPoints[topRightIndex] + [padding, -padding]

    # Yeni düzenlenmiş köşe noktalarını oluştur
    myPointsNew = np.array([topLeft, bottomLeft, topRight, bottomRight], dtype=np.int32).reshape((4, 1, 2))

    return myPointsNew


# öğrenci numrasının tespiti
def id_reorder(myPixelVal):
    duz_liste = []
    for sutun in range(len(myPixelVal[0])):
        for satir in range(len(myPixelVal)):
            duz_liste.append(myPixelVal[satir][sutun])
    yeni_liste = []
    satir = []
    for eleman in duz_liste:
        satir.append(eleman)
        if len(satir) == len(myPixelVal):
            yeni_liste.append(satir)
            satir = []
    return yeni_liste


# resmi satır ve sütunlara bölere satır*sütun sayısı kadar tekli şıkkı listeye ekler
def splitBoxes(img, questions, choice):
    rows = np.vsplit(img, questions)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, choice)
        for box in cols:
            boxes.append(box)
    return boxes


## İşaretli şıkkı bulmak için her kutudaki piksel sayısını buluyoruz
def pixelVal(questions, choices, box):
    countR = 0  # rows
    countC = 0  # column
    myPixelVal = np.zeros((questions, choices))
    for image in box:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if (countC == choices): countC = 0;countR += 1

        # print(myPixelVal)
    return myPixelVal

    # puan hesaplama alani
    # soru sayisi dogru cevaplari ve ogrenci cevaplarini aliyor
    # bunlari karsilastirip yeni bir listeye 1/0 seklinde kodluyor
    # 1ler toplanip puan hesaplanmis oluyor


def grading(questions, answers, myAnswers):
    correct_ans = []
    wrong_ans = []
    empty = []

    for x in range(0, questions):
        if answers[x] == myAnswers[x]:
            correct_ans.append(1)
        # birden fazla şık işaretleme indexe -1 olarak ekleniyor user_answer fonksiyonundan
        elif myAnswers[x] == -1:
            correct_ans.append(0)
            wrong_ans.append(x + 1)
        # empty answers değeri indexe 5 olarak ekleniyor  user_answer fonksiyonundan
        elif myAnswers[x] == 5:
            correct_ans.append(0)
            empty.append(x + 1)
            # yanlış şık işaretlenmişse
        else:
            correct_ans.append(0)
            wrong_ans.append(x + 1)

    score = ((sum(correct_ans) - (len(wrong_ans) / 4)) / questions) * 100
    return correct_ans, wrong_ans, empty, score


# ogrenci numarasi kisminin piksel degerine gore hangisinin iseretli oldugunun tespiti
def id_answers(vertical_num, myPixelVal):
    myIndex = []
    for x in range(0, vertical_num):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])
    return myIndex

    # piksel degerlerinde kullanici cevaplarini okuyupu index seklinde listeye kaydediyor


def user_answers(num_questions, myPixelVal):
    myIndex = []
    for x in range(0, num_questions):
        arr = myPixelVal[x]
        t = 300
        # empty answers
        if arr[0] < t and arr[1] < t and arr[2] < t and arr[3] < t and arr[4] < t:
            myIndex.append(5)

        # # 2 or more answers
        elif (arr[1] > t and arr[2] > t) or (arr[1] > t and arr[3] > t) or (arr[1] > t and arr[4] > t) or (
            arr[1] > t and arr[0] > t) or (arr[2] > t and arr[3] > t) or (arr[2] > t and arr[4] > t) or (
            arr[2] > t and arr[0] > t) or (arr[3] > t and arr[4] > t) or (arr[3] > t and arr[0] > t) or (
            arr[4] > t and arr[0] > t):
            myIndex.append(-1)

        else:
            myIndexVal = np.where(arr == np.amax(arr))
            myIndex.append(myIndexVal[0][0])

    return myIndex


# ızgara çiz
def drawGrid(img, questions, choices):
    secW = int(img.shape[1] / choices)
    secH = int(img.shape[0] / questions)
    for i in range(0, questions):
        pt1 = (0, secH * i)
        pt2 = (img.shape[1], secH * i)
        pt3 = (secW * i, 0)
        pt4 = (secW * i, img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        cv2.line(img, pt3, pt4, (255, 255, 0), 2)

    return img


# işaretli cevap doğru ise yeşil yanlış ise kırmızı şekil çiz
def showAnswers(img, myIndex, grading, ans, questions, choices):
    secW = int(img.shape[1] / choices)
    secH = int(img.shape[0] / questions)

    for x in range(0, questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2
        if grading[x] == 1:
            # cv2.rectangle(img,(myAns*secW,x*secH),((myAns*secW)+secW,(x*secH)+secH),myColor,cv2.FILLED)
            cv2.circle(img, (cX, cY), 10, yesil, cv2.FILLED)
        else:
            # cv2.rectangle(img, (myAns * secW, x * secH), ((myAns * secW) + secW, (x * secH) + secH), myColor, cv2.FILLED)
            cv2.circle(img, (cX, cY), 10, kirmizi, cv2.FILLED)
            # işaretli cevap yanlışsa doğru cevap biraz küçük ve yeşil
            correctAns = ans[x]
            cv2.circle(img, ((correctAns * secW) + secW // 2, (x * secH) + secH // 2),
                       6, yesil, cv2.FILLED)


# cevap anahtarı kisminin piksel degerine gore hangisinin iseretli oldugunun tespiti


import numpy as np

def answerKey(question, myPixelVal):
    myIndex = []

    for x in range(question):
        arr = myPixelVal[x]
        print(x,"-->",arr)
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])
   
    return myIndex
