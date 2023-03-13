Harika makine öğrenimi çerçevelerinin, kitaplıklarının ve yazılımlarının (dile göre) derlenmiş bir listesi. `awesome-php`den esinlenilmiştir.

_Bu listeye katkıda bulunmak istiyorsanız (lütfen yapın), bana bir çekme isteği gönderin veya benimle iletişime geçin [@josephmisiti](https://twitter.com/josephmisiti)._
Ayrıca, listelenen bir depo şu durumlarda kullanımdan kaldırılmalıdır:

* Deponun sahibi açıkça "bu kitaplığın bakımı yapılmadığını" söylüyor.
* Uzun süredir (2~3 yıl) taahhüt edilmemiştir.

Diğer kaynaklar:

* İndirilebilecek ücretsiz makine öğrenimi kitaplarının listesi için [buraya](https://github.com/josephmisiti/awesome-machine-learning/blob/master/books.md) gidin.

* Profesyonel makine öğrenimi etkinliklerinin listesi için [buraya](https://github.com/josephmisiti/awesome-machine-learning/blob/master/events.md) gidin.

* Çevrimiçi olarak sunulan (çoğunlukla) ücretsiz makine öğrenimi kurslarının bir listesi için [buraya](https://github.com/josephmisiti/awesome-machine-learning/blob/master/courses.md) gidin.

* Veri bilimi ve makine öğrenimi ile ilgili blogların ve haber bültenlerinin listesi için [buraya](https://github.com/josephmisiti/awesome-machine-learning/blob/master/blogs.md) gidin.

* Ücretsiz katılım sağlanan buluşmaların ve yerel etkinliklerin listesi için [buraya](https://github.com/josephmisiti/awesome-machine-learning/blob/master/meetups.md) gidin.

## İçindekiler

### Çerçeveler ve Kitaplıklar
<!-- MarkdownTOC derinliği=4 -->

- [Harika Makine Öğrenimi ![Harika](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](#awesome-machine-learning-)
   - [İçindekiler](#içindekiler tablosu)
     - [Çerçeveler ve Kitaplıklar](#çerçeveler ve kitaplıklar)
     - [Araçlar](#araçlar)
   - [APL](#apl)
       - [Genel Amaçlı Makine Öğrenimi](#apl-genel amaçlı makine öğrenimi)
   - [C](#c)
       - [Genel Amaçlı Makine Öğrenimi](#c-genel amaçlı makine öğrenimi)
       - [Bilgisayarlı Görüntü](#c-bilgisayarlı görüş)
   - [C++](#cpp)
       - [Bilgisayar Görüntüsü](#cpp-bilgisayar görüşü)
       - [Genel Amaçlı Makine Öğrenimi](#cpp-genel amaçlı makine öğrenimi)
       - [Doğal Dil İşleme](#cpp-natural-language-processing)
       - [Konuşma Tanıma](#cpp-konuşma tanıma)
       - [Dizi Analizi](#cpp-dizi-analizi)
       - [Hareket Algılama](#cpp-jest algılama)
   - [Ortak Lisp](#ortak-lisp)
       - [Genel Amaçlı Makine Öğrenimi](#ortak-lisp-genel-amaçlı-makine öğrenimi)
   - [Clojure](#clojure)
       - [Doğal Dil İşleme](#clojure-natural-language-processing)
       - [Genel Amaçlı Makine Öğrenimi](#clojure-genel amaçlı makine öğrenimi)
       - [Derin Öğrenme](#clojure-deep-learning)
       - [Veri Analizi](#clojure-data-analysis--data-visualization)
       - [Veri Görselleştirme](#clojure-data-visualization)
       - [Birlikte çalışma](#clojure-birlikte çalışma)
       - [Çeşitli](#clojure-misc)
       - [Ekstra](#clojure-ekstra)
   - [Kristal](#kristal)
       - [Genel Amaçlı Makine Öğrenimi](#kristal genel amaçlı makine öğrenimi)
   - [İksir](#iksir)
       - [Genel Amaçlı Makine Öğrenimi](#iksir-genel amaçlı makine öğrenimi)
       - [Doğal Dil İşleme](#iksir-doğal-dil-işleme)
   - [Erlang](#erlang)
       - [Genel Amaçlı Makine Öğrenimi](#erlang-genel amaçlı makine öğrenimi)
   - [Fortran](#fortran)
       - [Genel Amaçlı Makine Öğrenimi](#fortran-genel amaçlı makine öğrenimi)
       - [Veri Analizi / Veri Görselleştirme](#fortran-data-analysis--data-visualization)
   - [Git git)
       - [Doğal Dil İşleme](#go-natural-language-processing)
       - [Genel Amaçlı Makine Öğrenimi](#go-genel amaçlı makine öğrenimi)
       - [Uzamsal analiz ve geometri](#go-uzaysal-analiz-ve-geometri)
       - [Veri Analizi / Veri Görselleştirme](#go-data-analysis--data-visualization)
       - [Bilgisayarla görme](#go-computer-vision)
       - [Güçlendirmeli öğrenme](#go-reinforcement-learning)
   - [Haskell](#haskell)
       - [Genel Amaçlı Makine Öğrenimi](#haskell-genel amaçlı makine öğrenimi)
   - [Java](#java)
       - [Doğal Dil İşleme](#java-natural-language-processing)
       - [Genel Amaçlı Makine Öğrenimi](#java-genel amaçlı makine öğrenimi)
       - [Konuşma Tanıma](#java-konuşma tanıma)
       - [Veri Analizi / Veri Görselleştirme](#java-data-analysis--data-visualization)
       - [Derin Öğrenme](#java-deep-learning)
   - [Javascript](#javascript)
       - [Doğal Dil İşleme](#javascript-natural-language-processing)
       - [Veri Analizi / Veri Görselleştirme](#javascript-data-analysis--data-visualization)
       - [Genel Amaçlı Makine Öğrenimi](#javascript-genel amaçlı makine öğrenimi)
       - [Çeşitli](#javascript-çeşitli)
       - [Demolar ve Komut Dosyaları](#javascri
