import cv2
import numpy as np

#RESİM OKUNDU
Resim = cv2.imread("canli.jpg")

#RESİM GRİ RENGE ÇEVRİLDİ
Resim_Gri = cv2.cvtColor(Resim,cv2.COLOR_BGR2GRAY)
kernel =np.ones((3,3),np.uint8)



#AŞAĞIDA RESİM BELİRLENEN ARALIKTAYSA BEYAZ DEĞİLSE SİYAH YAPILDI
ret,Resim_Threshold=cv2.threshold(Resim_Gri,150,255,cv2.THRESH_BINARY)

#SİYAH BEYAZ  RESME EREZYON UYGULANDI
erosion=cv2.erode(Resim_Threshold,kernel,iterations=1)

#BURADA İSTENİLMEYEN GÜRÜLTÜLERİ ATMAK İÇİN MASKELEME KULLANILDI
Resim_Mask = cv2.bitwise_and(Resim,Resim,mask=erosion)

#RESİMİN HSV GÖRÜNTÜSÜ ÇIKARILDI
Resim_HSV = cv2.cvtColor(Resim_Mask,cv2.COLOR_BGR2HSV)

#BURADA HSV DEĞERLERİNE KARAR VERİLDİ
Gri_Alt_Sinir = np.array([0,0,15])
Gri_Ust_Sinir = np.array([255,255,255])


#BELİRLENEN ARALIKLARI BEYAZ YAPIYORUZ
Gri_Renk_Filtre_Sonucu=cv2.inRange(Resim_HSV,Gri_Alt_Sinir,Gri_Ust_Sinir)


#SONUCU GÖRMEK İÇİN RESMİ KOPYALADIK
Sonuc=Resim.copy()

#CIKARILAN İSKELET ÜZERİNE YEŞİL NOKTALAR KONULDU
_, cnts, _ =cv2.findContours(Gri_Renk_Filtre_Sonucu,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(Sonuc,cnts, -1 ,(0,255,0), 35)






#GOSTERİLEN SAYFALAR


cv2.imshow("normal",Resim)
cv2.imshow("Gri",Resim_Gri)
cv2.imshow("Threshold",Resim_Threshold)
cv2.imshow("erezyon",erosion)
cv2.imshow("Mask",Resim_Mask)
cv2.imshow("HSV",Resim_HSV)
cv2.imshow("Gri Filtre",Gri_Renk_Filtre_Sonucu)
cv2.imshow("Sonuc",Sonuc)

cv2.waitKey(0)