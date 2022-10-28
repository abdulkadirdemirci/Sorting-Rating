import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
import scipy.stats as st

############################################
# Rating Product & Sorting Reviews in Amazon
############################################
"""
is problemi:
E-ticaretteki en onemli problemlerden bir tanesi urunlere satis
sonrasi verilen puanlarin dogru sekilde hesaplanmasidir. Bu
problemin cozumu e-ticaret sitesi icin daha fazla musteri
memnuniyeti saglamak, saticilar icin urunun one cikmasi ve satin
alanlar icin sorunsuz bir alisveris deneyimi demektir. Bir diger
problem ise urunleree verilen yorumlarin dogru bir sekilde
siralanmasi olarak karsimiza cikmaktadir. Yaniltici yorumlarin one
cikmasi urunun satisini dogrudan etkileyeceginden dolayi hem
maddi kayip hem de musteri kaybina neden olacaktir. Bu 2 temel
problemin cozumunde e-ticaret sitesi ve saticilar satislarini
arttirirken musteriler ise satin alma yolculugunu sorunsuz olarak
tamamlayacaktir.
"""
"""
veri seti hikayesi:
Amazon urun verilerini iceren bu veri seti urun kategorileri ile 
cesitli metadatalari icermektedir. Elektronik kategorisindeki en
fazla yorum alan urunun kullanici puanlari ve yorumlari vardir.

reviewerID          Kullanici ID’si
asin                Urun ID’si
reviewerName        Kullanici Adi
helpful             Faydali degerlendirme derecesi
reviewText          Degerlendirme
overall             Urun rating’i
summary             Degerlendirme ozeti
unixReviewTime      Degerlendirme zamani
reviewTime          Degerlendirme zamani Raw
day_diff            Degerlendirmeden itibaren gecen gun sayisi
helpful_yes         Degerlendirmenin faydali bulunma sayisi
total_vote          Degerlendirmeye verilen oy sayisi
"""
###############################
# GOREV1: Average Rating’i guncel yorumlara gore hesaplayiniz
# ve var olan average rating ile kiyaslayiniz.
# Paylasilan veri setinde kullanicilar bir urune puanlar vermis ve yorumlar yapmistir.
# Bu gorevde amacimiz verilen puanlari tarihe gore agirliklandirarak degerlendirmek.
# Ilk ortalama puan ile elde edilecek tarihe gore agirlikli puanin karsilastirilmasi gerekmektedir
###############################
# GOREV1 ADIM1: Urunun ortalama puanini hesaplayiniz.
###############################
pd.set_option("display.expand_frame_repr", False)

df_ = pd.read_csv("measurement_problems/miuul_hafta5_imdb/amazon_review.csv")
df = df_.copy()
df.head(5)
df.sample(5)
df["helpful_yes"].value_counts().sort_index().head(7)
df.shape
df.columns

urun_ort_rating = df["overall"].mean()
###############################
# GOREV1 ADIM2: Tarihe gore agirlikli puan ortalamasini hesaplayiniz.
# • reviewTime degiskenini tarih degiskeni olarak tanitmaniz
# • reviewTime'in max degerini current_date olarak kabul etmeniz
# • her bir puan-yorum tarihi ile current_date'in farkini gun cinsinden
# ifade ederek yeni degisken olusturmaniz ve gun cinsinden ifade edilen
# degiskeni quantile fonksiyonu ile 4'e bolup (3 ceyrek verilirse 4 parca cikar)
# ceyrekliklerden gelen degerlere gore agirliklandirma yapmaniz
# gerekir. Ornegin q1 = 12 ise agirliklandirirken 12 gunden az sure once yapilan
# yorumlarin ortalamasini alip bunlara yuksek agirlik vermek gibi.
###############################
df.head()
df.info()
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = df.reviewTime.max()
df["fark_gun"] = (current_date - df["reviewTime"]).dt.days
df["fark_gun"].quantile([.1, .25, .5, .75, .8, .9, .95])
"""
ilk 300 gun içinde yorum yapanların agirligi:   w1
2. 300 gun içinde yorum yapanlarin agirligi:    w2
600 gunden daha önce yorum yapanlarin agirligi: w3
"""


def time_based_weighted(df, w1, w2, w3, lim1=300, lim2=600):
    """
    verilen df icinde limitler ile parcalara ayrilmis araliklara
     w1,w2,w3 agirliklarini kullanarak ortalama hesaplar

    :param df: dataframe
    :param w1: 0 < w1 < 1, float
    en yakin donemde yorum yapanlarin agirligi
    :param w2: 0 < w2 < 1, float
    orta donemde yorum yapanlarin agirligi
    :param w3: 0 < w3 < 1, float
    uzak donemde yorum yapanlarin agirligi
    :param lim1: zaman dilimi 1
    :param lim2: zaman dilimi 2
    :return:
    iki kesitle 3 parcaya ayrilmis veri setine w1, w2, w3 agirliklari
    uygulanarak zaman_tabanli_agirlikli_rating degerini verir.
    """
    return df.loc[df["fark_gun"] < lim1, "overall"].mean() * w1 + \
           df.loc[(df["fark_gun"] > lim1) & (df["fark_gun"] <= lim2), "overall"].mean() * w2 + \
           df.loc[df["fark_gun"] > lim2, "overall"].mean()*w3


time_based_weighted(df, w1=.45, w2=.33, w3=.22)

###############################
# GOREV1 ADIM3: Agirliklandirilmis puanlamada her bir zaman diliminin
# ortalamasini karsilastirip yorumlayiniz
###############################
df.loc[df["fark_gun"] < 300, "overall"].mean()
df.loc[(df["fark_gun"] > 300) & (df["fark_gun"] <= 600), "overall"].mean()
df.loc[df["fark_gun"] > 600, "overall"].mean()
"""
ortalama ilk 1 yil, 2 yil ve 2 yildan daha uzun sure once yapilan yorumlar 
olarak verimizi ayirdigimizda, goruyoruz ki zaman icindeki trendi daha iyi 
yakalama imkanina sahibiz. ilgilenile uruna her gecen yil ilgi daha da artarak 
devam etmekte. artis azimsanmayacak bir oranda ilgi gormeye devam etmis(~0.11)
eger zamani goz ardi etmis olsaydik urunun suanki ortalama puani gecmisteki ratingler
altinda eziliyor olacaktir
"""
###############################
# GOREV2: Urun icin urun detay sayfasinda goruntulenecek 20 review’i belirleyiniz.
###############################
###############################
# GOREV2 ADIM1: helpful_no değişkenini üretiniz.
# • total_vote bir yoruma verilen toplam up-down sayısıdır.
# • up, helpful demektir.
# • Veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.
# • Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes)
# çıkarılarak yararlı bulunmayan oy sayılarını (helpful_no) bulunuz.
###############################
df["helpful_no"] = df["total_vote"]-df["helpful_yes"]
###############################
# GOREV2 ADIM2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarini hesaplayip veriye ekleyiniz.
# • score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarini hesaplayabilmek icin score_pos_neg_diff,
# score_average_rating ve wilson_lower_bound fonksiyonlarini tanimlayiniz.
# • score_pos_neg_diff'a gore skorlar olusturunuz. Ardindan; df icerisinde score_pos_neg_diff ismiyle kaydediniz.
# • score_average_rating'a gore skorlar olusturunuz. Ardindan; df icerisinde score_average_rating ismiyle kaydediniz.
# • wilson_lower_bound'a gore skorlar olusturunuz. Ardindan; df icerisinde wilson_lower_bound ismiyle kaydediniz.
###############################
# score_pos_neg_diff
df["score_pos_neg_diff"] = df["helpful_yes"]-df["helpful_no"]
# score_average_rating
df["score_average_rating"] = [df["helpful_yes"][x]/df["total_vote"][x] if df["helpful_yes"][x] >0 else 0 for x in df.index ]
# wilson_lower_bound
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)
df.head()
df["wilson_lower_bound_score"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],x["helpful_no"]),axis=1)


###############################
# GOREV2 ADIM3 : 20 Yorumu belirleyiniz ve sonuclari Yorumlayiniz.
# • wilson_lower_bound'a gore ilk 20 yorumu belirleyip siralayaniz.
# • Sonuclari yorumlayiniz.
###############################
df.sort_values(by="score_pos_neg_diff",ascending=False).head(10)
df.sort_values(by="score_average_rating",ascending=False).head(10)

targets=["helpful","total_vote","overall","helpful_yes","helpful_no","wilson_lower_bound_score"]
df.sort_values(by="wilson_lower_bound_score",ascending=False)[targets].head(20)
"""
bir satis platformunda ilk 20 yorum yorumlar sekmesinin ilk 
sekmesi demektir. satin alicilarinin cok az bir kismi 2. ve 3. sekmelere
bakar. bu yuzden bu ilk 20 de urun hakkında gercekten anlamlı ve  diger
kullanicilar tarfindan faydali bulunmus yorumlar bulunmali.
ciktimizin kiymeti diger reviews siralama methondali ile karsilastirilmadan 
anlasilamayabilir. lakin goruldugu uzere hem kullanicilardan 5 yildiz verenlerin
yorumlari hem de 1 yildiz verenlerin yorumlari gorunmektedir. lakin 
wilson_lower_bound yontemi de digerleri gibi tek basina kullanidiginda 
yakaladigi iyi yonler oldugu gibi kacirdigi kisimlar da vardir, mesela daha önce 
incelemis ve degerlendirmis oldugumuz zaman_tabanli hesaplamalar bize zaman
icindeki urun hakkinda gelismekte olan fikir ve dusuncelerin trendini 
yakalama imkani veriyordu. eger WLB kullanilmak isteniyorsa tek basina degil
bu iki yaklasimin agirlikli ortalamasi icinde belirlenen bir agirliga sahip olacak
sekilde degerlendirilmeli. ayrıca verimizde bulunuyordu fakat kimin ne zman oy verdigi
kadar kimin oy verdigi yorum yaptigi da önemli burada devreye kullanici skorlari
giriyor. her kullanici platform icinde bulundugu etkilesime gore bir degere sahiptir
ve bu deger miktarince kendisinden gelen yorumlar geri donusler o derece oneme sahip olur.


"""




































