# Laporan Proyek Machine Learning - Widi Afandi

## Project Overview

Buku merupakan sumber ilmu yang memberikan banyak pengetahuan kepada pembacanya. Di tahun sebelumnya, buku hanya dapat diakses dengan ke perpustakaan ataupun membelinya di toko buku. Di zaman yang canggih ini, buku dapat dibeli dengan melalui e-commerce atau bahkan sekarang tersedia e-book yang mana singkatan dari electronic book (buku elektronik) yang merupakan buku digital dengan format pdf dan sebagainya yang dapat dibuka melalui smartphone atau pun laptop. Namun, di tengah kemudahan yang diberikan untuk mengakses buku ternyata minat membaca orang-orang masih rendah. Menurut data dari UNESCO, minat baca masyarakat Indonesia sangat rendah yaitu hanya sebesar 0,001%. Artinya, dari 1,000 orang Indonesia, cuma 1 orang yang rajin membaca. Beberapa alasan orang malas membaca dikutip dari [IDNTimes](https://www.idntimes.com/life/education/andri-andreas-1/alasan-mengapa-kamu-malas-membaca-buku-c1c2?page=all) adalah sebagai berikut:
1. **Image dari buku sudah sangat tidak baik, mereka akan menganggap semua buku membosankan**
2. Sibuk dengan kegiatan lain
3. Ada media sosial dan game online yang mengisi waktu luang
4. Dari awal memang tidak memiliki ketertarikan untuk membaca buku
5. Harga buku yang cukup mahal

Dengan demikian yang menjadi alasan mayoritas orang malas membaca buku adalah anggapan buku itu membosankan. Padahal film populer seperti "Harry Potter" diangkat dari sebuah buku/novel. Dengan banyaknya akses untuk mendapatkan buku, yang menjadikan banyaknya informasi buku yang tersebar luas, penerapan sistem rekomendasi dapat dimanfaatkan untuk memudahkan pembaca buku dalam menemukan buku yang sesuai dengan kriteria yang diinginkan. Sehingga pengguna dapat dengan mudah mendapatkan referensi baru mengenai buku yang bisa dibaca sesuai dengan kategori buku yang diminati oleh pembaca sebelumnya dan harapannya dapat meningkatkan minat masyarakat untuk membaca.

## Business Understanding

### Problem Statements
- Bagaimana cara untuk menemukan buku yang populer dengan rating yang tinggi?
- Bagaimana cara menerapkan sistem rekomendasi dengan pendekatan yang sesuai pada data yang dipakai untuk membuat sistem rekomendasi buku? 

### Goals
- Melakukan Exploratory Data Analysis untuk menemukan buku populer dan penulis populer dengan rating yang tinggi
- Melakukan analisis pada data yang digunakan untuk menentukan apakah data lebih cocok menggunakan content-based atau collaborative-based dalam membangun sistem rekomendasi buku

## Data Understanding
![image](https://github.com/technisekai/book-recommendation/assets/54144923/0416848c-22e6-4c77-9637-5b332838ebfa)

Data yang digunakan diunduh dari penyedia data terbuka [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) dimana terdapat 3 file csv disin yaitu:
- Books : Buku yang tersedia pada sistem
- Users : Daftar user yang terdaftar pada sistem
- Ratings : User yang memberikan rating pada judul buku tertentu

Berikut hasil analisis terhadap data yang dipakai:
| Jumlah Penulis Buku | Jumlah Buku di Sistem | Jumlah User yang Memberikan Rating |
|---------------------|-----------------------|------------------------------------|
| 102.024             | 278.858               | 105.283                            |

Terdapat informasi yang menarik disini bahwa dari keseluruhan user yang terdaftar pada sistem hanya sekitar 30% saja yang memberikan rating. Jika diasumsikan yang memberikan rating ini adalah orang yang membeli dan dibaca bukunya, maka hanya ada 30% orang yang membaca buku.

Kemudia Rating yang diberikan oleh user adalah dalam rentang 0-10 yang mana 0 adalah yang terburuk dan 10 adalah yang terbaik. Berikut dilampirkan hasil visualisasi penulis dengan buku terbanyak, judul buku dan penulis yang mendapatkan rating rata-rata 10:

![5 Penulis dengan Jumlah Buku Terbanyak](https://github.com/technisekai/book-recommendation/assets/54144923/cd5f1e8f-4559-46e0-9925-fa6d19c7d8e6)
|                                                                                                                             Book-Title | Rating   | 
|---------------------------------------------------------------------------------------------------------------------------------------:|---------:|
|                                                 Film Is: The International Free Cinema                                                 |     10.0 |
| More Secrets of Happy Children: Embrace Your Power as a Parent--and Help Your Children be Confident, Positive, Well-Adjusted and Happy |     10.0 |
|                                                 Jo's Boys : From the Original Publisher                                                |     10.0 |
|                                             The Vanished Priestess : An Annie Szabo Mystery                                            |     10.0 | 
|                                                            Game and Hunting                                                            |     10.0 |
|                                             The Vanished Priestess : An Annie Szabo Mystery                                            |     10.0 |
|                                                            Game and Hunting                                                            |     10.0 |

![Penulis dengan rating buku tertinggi](https://github.com/technisekai/book-recommendation/assets/54144923/15e20f30-6d66-4faf-86fe-18284feeecd5)

Di tahap ini, data digabungkan berdasarkan file rating yaitu pada kolom User-ID untuk menggabungkan file rating dan user dan ISBN untuk menggabungkan kolom rating dan buku.

## Data Preparation
![Flow Preparation Data](https://github.com/technisekai/book-recommendation/assets/54144923/d3531dd8-13de-45f7-93e8-20b34b6f37d7)

Pada tahap ini ditemukan bahwa fitur buku lebih lengkap yaitu penulis, tahun terbit, dan publisher. Sehingga berdasarkan kelengkapan pada fitur item dimana item di studi kasus ini adalah buku maka pendekatan modelling yang sesuai adalah *conten-based* dimana digunakan 2 kolom sebagai data yaitu kolom Book-Title/Judul Buku dan Book-Author/Penulis. Dipilihnya Book-Author dikarenakan para pembaca biasanya lebih tertarik dengan jalan cerita yang dibuat oleh penulis tertentu karena masing-masing penulis memiliki style yang berbeda dalam menulis buku. Kemudian di tahap ini juga row data akan dikurangi dari yang sebelumnya 1.013.965 data menjadi 30.000 data karena keterbatasan komputasi yang dimiliki. Adapun masing masing tahapan data preparation dilakukan pada kolom Book-Author saja. Berikut penjelasan step-by-stepnya:
1. lowercase : nama penulis terkadang ditulis dengan huruf kapital semua dan huruf kecil, jika tidak disamakan ini akan menimbulkan bias karena sistem pada saat proses tf-idf akan mendeteksi bahwa kedua penulis tersebut berbeda padahal sama hanya karena beda penulisan huruf kapital dan kecil.
2. remove non-alphanumeric : tanda baca pada nama penulis juga tidak dibutuhkan sehingga dibuang agar lebih terstandarisasi
3. remove missing value : terdapat beberapa missing value pada data, penulis memutuskan untuk menghilangkan data tersebut karena penulis masih mempunyai cukup data untuk dianalisis
4. TF-IDF : melakukan vektorisasi yaitu mengubah kata menjadi numerik pada kolom Book-Author karena algoritma tidak bisa membaca teks melainkan angka.

## Modeling
Tahapan ini dipilih pendekatan **Content-Based** dikarenakan data darii items yaitu buku lebih lengkap dibandingkan data yang diberikan user. Pada kasus ini, data yang diberikan user hanya jumlah rating saja sehingga kurang representatif jika digunakan pendekatan **Collaborative-Filtering**. Pada tahap ini digunakan cosine-similiarity untuk menghitung jarak kedekatan/kemiripan antar item. Adapun formula dari cosine-similiarity adalah sebagai berikut:

![image](https://github.com/technisekai/book-recommendation/assets/54144923/12f8d40d-da0a-4498-b626-f18d19c83a8f)

Dimana:
- 0 adalah derajat kedekatan items
- A . B adalah hasil dot product dari A dan B
- ||A|| L2 norm atau magnitude dari vector

Adapun hasil cosine-similiaritynya adalah sebagai berikut:
|                            Book-Title | Palm Sunday | Shroud of Shadow | Singing in the Comeback Choir | Count Your Blessings : 63 Things to Be Grateful for in Everyday Life . . . and How to Appreciate Them | For My Daughter (Harlequin Promo) |
|--------------------------------------:|------------:|-----------------:|------------------------------:|------------------------------------------------------------------------------------------------------:|----------------------------------:|
|                            Book-Title |             |                  |                               |                                                                                                       |                                   |
|              Captive Star             |         0.0 |              0.0 |                           0.0 |                                                                                                   0.0 |                               0.0 |
|              The Witness              |         0.0 |              0.0 |                           0.0 |                                                                                                   0.0 |                               0.0 |
|    The Canadian Rockies Trail Guide   |         0.0 |              0.0 |                           0.0 |                                                                                                   0.0 |                               0.0 |
|              On the Beach             |         0.0 |              0.0 |                           0.0 |                                                                                                   0.0 |                               0.0 |
|    The Ultimate Hitchhiker's Guide    |         0.0 |              0.0 |                           0.0 |                                                                                                   0.0 |                               0.0 |
|                Bookends               |         0.0 |              0.0 |                           0.0 |                                                                                                   0.0 |                               0.0 |
|           Die Faust Gottes.           |         0.0 |              0.0 |                           0.0 |                                                                                                   0.0 |                               0.0 |
|             Her Own Rules             |         0.0 |              0.0 |                           0.0 |                                                                                                   0.0 |                               0.0 |
|                Bygones                |         0.0 |              0.0 |                           0.0 |                                                                                                   0.0 |                               0.0 |
| The Information Please Girls' Almanac |         0.0 |              0.0 |                           0.0 |                                                                                                   0.0 |                               0.0 |

kemudian untuk hasil rekomendasinya dicoba melakukan rekomendasi top 5 dari buku dengan judul Palm Sunday dan didapatkan hasil berikut:

| No |                                        Book-Title | Book-Author   |
|---:|--------------------------------------------------:|---------------|
|  0 | Slaughterhouse Five or the Children's Crusade:... | kurt vonnegut |
|  1 |                               The Sirens of Titan | kurt vonnegut |
|  2 |                                         Bluebeard | kurt vonnegut |
|  3 |                                         Bluebeard | kurt vonnegut |
|  4 |                                      Mother Night | kurt vonnegut |

## Evaluation

Palm Sunday merupakan karya dari penulis Kurt Vonnegut. Buku yang direkomendasikan pun benar adanya bahwa karya Kurt Vonnegut. Untuk menghitung metrik precision pada sistem rekomendasi digunakan formula berikut:

`precision = Jumlah Prediksi Benar / k * 100`

sedemikian sehingga:

`precision = 5 /5 * 100 = 100%`

hasilnya precision model cosine similiarity ini adalah 100%

## Referensi
[1]	Mobius, “Book Recommendation Dataset | Kaggle.” https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset (accessed May 20, 2023).

[2]	M. Fajriansyah, P. P. Adikara, and A. W. Widodo, “Sistem Rekomendasi Film Menggunakan Metode Content Based Filtering,” J. Pengemb. Teknol. Inf. dan Ilmu Komput., vol. 5, no. 6, pp. 2188–2199, 2021, Accessed: May 20, 2023. [Online]. Available: http://e-journal.uajy.ac.id/20600/

[3]	R. W. Online, “Minat Baca Orang Indonesia Serendah Ini? Benar Gak Sih?!,” 18 Juni 2021, 2021. https://www.wartaekonomi.co.id/read346432/minat-baca-orang-indonesia-serendah-ini-benar-gak-sih (accessed May 20, 2023).

[4]	A. Yanto, “5 Alasan Mengapa Kamu Malas Membaca Buku.” https://www.idntimes.com/life/education/andri-andreas-1/alasan-mengapa-kamu-malas-membaca-buku-c1c2?page=all (accessed May 20, 2023).

[5]	H. S. Kusuma and A. Musdholifah, “Recommendation System for Thesis Topics Using Content-based Filtering,” IJCCS (Indonesian J. Comput. Cybern. Syst., vol. 15, no. 1, p. 65, Jan. 2021, doi: 10.22146/ijccs.62716.

[6]	S. H. Nallamala, U. R. Bajjuri, S. Anandarao, D. D. D. Prasad, and D. P. Mishra, “A Brief Analysis of Collaborative and Content Based Filtering Algorithms used in Recommender Systems,” IOP Conf. Ser. Mater. Sci. Eng., vol. 981, no. 2, p. 022008, Dec. 2020, doi: 10.1088/1757-899X/981/2/022008.


