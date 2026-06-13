Loyiha Xulosasi: Bank Tranzaksiyalarida Firibgarlikni Aniqlash (Fraud Detection)
Muammo (Problem Statement):
Bank tizimlarida firibgarlik holatlari umumiy tranzaksiyalarning juda kichik qismini (0.17%) tashkil qiladi. Ushbu keskin "Imbalanced Data" (muvozanatsiz ma'lumotlar) muammosi an'anaviy Machine Learning modellarini chalg'itib, ularning barcha tranzaksiyalarni "toza" deb bashorat qilishiga olib keladi. Loyihaning maqsadi — halol mijozlarni asossiz bloklash (False Positives) holatlarini minimal darajada saqlagan holda, firibgarlik tranzaksiyalarini (False Negatives) maksimal darajada aniqlovchi tizim yaratish.

Yechim va Metodologiya (Methodology):

Data Engineering (SQL & Pandas): Ma'lumotlar PostgreSQL bazasidan xavfsiz tarzda ajratib olindi va tahlil uchun Pandas DataFrame'ga o'tkazildi.

Exploratory Data Analysis (EDA): Tranzaksiya summasi (Amount) va vaqti (Time) tahlil qilindi. Firibgarlar odatda insonlarning tabiiy uyqu ritmiga bo'ysunmasligi va o'rtacha xariddan kattaroq, lekin bank cheklovlariga tushmaydigan summalarni yechishga harakat qilishi vizual isbotlandi.

Model Selection & Tuning: * Baseline: Dastlabki mantiqni tekshirish uchun Logistic Regression ishlatildi.

Optimization: Random Forest algoritmi class_weight='balanced' va Probability Thresholding (0.3) orqali biznes talablariga moslashtirildi.

Final Model: Sanoat standarti bo'lgan XGBoost algoritmi qullanildi va scale_pos_weight orqali muvozanatsizlik muammosi hal qilindi.

Model Explainability: SHAP kutubxonasi yordamida modelning qaror qabul qilish mantiqi ochib berildi (eng muhim omillar aniqlandi).

Biznes Qiymat (Business Impact):
Yakuniy XGBoost modeli test ma'lumotlarida ajoyib muvozanatni ko'rsatdi. U 85,300 ta toza tranzaksiya ichidan atigi 10 tasini adashib blokladi (False Positive yuki deyarli 0 ga teng), shu bilan birga haqiqiy firibgarlarning mutlaq ko'p qismini muvaffaqiyatli ushlab qoldi. Bu bank xavfsizligini ta'minlash va mijozlar tajribasini (Customer Experience) saqlab qolish o'rtasidagi eng ideal yechimdir.