"""
Casharka: Day 15 - CNN + LSTM Dhismihii Maskaxda
Ujeedada: In la fahmo sida loo dhiso maskax garata Video-ga. "Indho" (CNN) iyo "Xasuus" (LSTM
CNN (MobileNetV2) = Feature Extractor (Indhaha)
LSTM = Sequence Learner (Xasuusta)
"""
# Sawir: Waa hal xabbo oo taagan. CNN-ta caadiga ah hal sawir ayay mar qura eegtaa. Video: Waa sawirro badan (frames) oo is xiga.
# TimeDistributed:
# Layer-kani wuxuu u dhaqmaa sidii "Loop" ama "Gadiid" CNN-ta dusha ka saaran. Wuxuu u oggolaanayaa Model-ka inuu CNN-ta ku isticmaalo frame kasta oo video-ga ka mid ah isagoo mid mid u dhex maraya.isla weightki 1 miisan
# CNN = Waxay soo saartaa waxa ku dhex jira hal frame (tusaale: qof baa taagan, lug baa kor u kacday).
# TimeDistributed = Waxay CNN-ta ka dhigtaa mid frame kasta mid-mid u eegi karta.
# LSTM = Waxay isku xirtaa macnihii ay CNN-ta soo saartay si ay u ogaato in lugta kacday iyo dhaqdhaqaaqa kale ay la macno yihiin "Orod".
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Flatten, TimeDistributed, Dropout
from tensorflow.keras.applications import MobileNetV2

# 1. Qeexidda Cabirka Video-ga (Hyperparameters)
# (Samples, Frames, Height, Width, Channels)
NUM_FRAMES = 20  # Video kasta waxaan ka qaadaynaa 20 sawir (frames) Model-ku wuxuu hal mar liqayaa 20 sawir oo isku xiga si uu u ogaado dhaqdhaqaaqa (Action-ka). half yar le couse ram ka ilalineyna vedioga images badan ka kobanyhe
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
NUM_CLASSES = 5  # Immisa shay ayaan rabaa inaan kala saaro?).Tirada ficillada kala duwan ee model-ka la barayo.

# 2. Dhisidda Model-ka (CNN + LSTM)
# Marka aad Model-ka dhisayso, wuxuu xogta u arkaa sidan: (20, 224, 224, 3) 20: Intee sawir baa is daba jooga? (Frames)
model = Sequential()

# A. QAYBTA CNN (Indhaha): Waxaan isticmaalaynaa MobileNetV2 oo horay u tababaran
# weights='imagenet' waxay la macno tahay: "Model-kan ha iga dhigin mid eber ah (madhan), ee iigu soo shub maskaxdii iyo khibraddii uu horay u soo bartay."Waxaan isticmaalaynaa khibradii model-ka uu horey ugu soo bartay 1.4 milyan oo sawir.
video_cnn = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
video_cnn.trainable = False # Ma rabno inaan dib u tababarno CNN-ka hadda

# TimeDistributed(video_cnn): Waxay CNN-ta (indhaha) u dirtay inay 20-ka frame mid-mid u soo eegto.
model.add(TimeDistributed(video_cnn, input_shape=(NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, CHANNELS)))
#Gudaha CNN-ta: Waxay xogtu ahayd: (20, 7, 7, 1280) Flatten: Waxay noqonaysaa: (20, 62720) 7*7*1280 (20 sawir, mid kastaa waa hal saf oo dheer).Frame kasta waxaan ka dhigaynaa hal saf oo dheer (Vector) si ay LSTM-tu u akhrin karto.
model.add(TimeDistributed(Flatten()))

# B. QAYBTA LSTM (Xasuusta): Si ay isku xirto 20-ka frame ee is daba jooga ah
# return_sequences=False (Gunaanad): Waxay la mid tahay adigoo yiri: "Dhammaan 20-kaas frame markaad eegto, ii sheeg hal ficil oo ay isku noqonayaan output Wuxuu soo saarayaa hal gunaanad oo video-ga oo dhan ah (Summary).
# True (Sequence): Waxay la mid tahay adigoo yiri: "Frame kasta oo aad akhrisidba, ii soo saar natiijo gooni ah." (Tan waxaa la isticmaalaa haddii aad rabto inaad LSTM kale ku xijiso). wuxuu soo saari lahaa 20 natiijo (mid kasta oo frame ka mid ah).
model.add(LSTM(64, return_sequences=False)) # 64 units oo xasuusta ah
model.add(Dropout(0.5)) # Si uusan model-ku u "khafifin" (Overfitting)

# C. QAYBTA GO'AANKA (Output):
# Dense (128): Waa maskax dheeraad ah oo si qoto dheer u dhex gasha xogta LSTM-ta.caqli dherad paterns complex sii barane
# Dense (NUM_CLASSES): Waa natiijada u dambaysa ee sheegaysa boqolayda ficil kasta.
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax')) # Softmax waxay sheegaysaa boqolayda ficil kasta

# 3. Compile (Isku dhiska model-ka)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# CNN + LSTM (Day 15)
# Qaabka uu u arko dhaqdhaqaaqa: Wuxuu u arkaa sidii "Sheeko taxane ah" (Sequence).
# Dhaqdhaqaaqa uu ku fiican yahay: Wuxuu aad ugu fiican yahay dhaqdhaqaaqa dheer ee wakhtiga badan qaata (Long-term dependency).
# Dhibka ka jira: Maadaama CNN-ku uu sawirada mid-mid u fiirinayo, waxaa laga yaabaa inuu seego dhaqdhaqaaqyada aadka u yaryar ee u dhexeeya labo frame oo is xiga.
# 2. 3D-CNN (Day 16)
# Qaabka uu u arko dhaqdhaqaaqa: Wuxuu u arkaa sidii "Hal xabbad oo isku duuban" (Spatiotemporal Cube).
# Dhaqdhaqaaqa uu ku fiican yahay: Wuxuu aad ugu fiican yahay dhaqdhaqaaqa dhow (Local Motion). Wuxuu si heer sare ah u fahmaa isbedelka yar ee u dhexeeya 3-5 frame oo is xiga (sidii gacanta oo wax yar dhaqaaqday).
# Dhibka ka jira: Wuxuu u baahan yahay computer aad u xoog badan (RAM/GPU weyn).
# 3. Two-Stream Network (Day 17)
# Qaabka uu u arko dhaqdhaqaaqa: Wuxuu u kala saaraa Muuqaalka (ma guurtada ah) iyo Dhaqdhaqaaqa (oo ah Optical Flow).
# Dhaqdhaqaaqa uu ku fiican yahay: Wuxuu u fiican yahay dhaqdhaqaaqa aadka u faahfaahsan (Fine-grained motion). Maadaama aad siinayso "Optical Flow" (oo ah xog ka hadlaysa dhaqdhaqaaqa kaliya), model-ku wuxuu arki karaa xataa haddii qofku faraha kaliya dhaqaajinayo.
# Dhibka ka jira: Waa in marka hore laga soo saaro Optical Flow-ga video-ga, taas oo waqti qaadata.