import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt

def nalozi_zvok(pot_do_datoteke):
    y, sr = sf.read(pot_do_datoteke)

    if y.ndim == 2:
        y = np.mean(y, axis=1) # mono zvok

    y = y.astype(np.float32)
    maks_vrednost = np.max(np.abs(y))
    if maks_vrednost > 1.0:
        y = y / maks_vrednost

    return y, int(sr)

def filtriraj_base(y, sr, mejna_frekvenca=150.0):
    nyq = sr / 2.0
    b, a = signal.butter(4, mejna_frekvenca / nyq, btype='low')
    filtriran_signal = signal.lfilter(b, a, y)
    return filtriran_signal.astype(np.float32)

def izracunaj_stft_rocno(y, n_fft=2048, hop=512):
    okno = np.hanning(n_fft)
    stevilo_oken = 1 + (len(y) - n_fft) // hop
    matrika_spektra = np.zeros((n_fft // 2 + 1, stevilo_oken), dtype=np.float32)

    for m in range(stevilo_oken):
        zacetek = m * hop
        konec = zacetek + n_fft
        kos_signala = y[zacetek:konec]
        kos_z_oknom = kos_signala * okno
        fft_rezultat = np.abs(np.fft.rfft(kos_z_oknom, n=n_fft))
        matrika_spektra[:, m] = fft_rezultat

    return matrika_spektra

def izracunaj_spektralni_pretok(matrika_spektra):
    razlika = np.diff(matrika_spektra, axis=1)
    samo_porast_energije = np.maximum(0.0, razlika)
    ovojnica_novosti = np.sum(samo_porast_energije, axis=0)
    if ovojnica_novosti.max() > 0:
        ovojnica_novosti = ovojnica_novosti / ovojnica_novosti.max()

    return ovojnica_novosti

def ugotovi_bpm(ovojnica_novosti, sr, hop):
    avtokorelacija = np.correlate(ovojnica_novosti, ovojnica_novosti, mode='full')
    sredina = len(ovojnica_novosti) - 1
    desna_stran = avtokorelacija[sredina:]
    okna_na_sekundo = sr / hop

    zamik_min = int(np.floor(okna_na_sekundo * 60.0 / 200.0))
    zamik_max = int(np.ceil(okna_na_sekundo * 60.0 / 60.0))

    izrezan_del = desna_stran[zamik_min:zamik_max]
    tau_optimalen = int(np.argmax(izrezan_del)) + zamik_min
    bpm = 60.0 * okna_na_sekundo / tau_optimalen
    return bpm, tau_optimalen

def poisci_točne_udarce(ovojnica_novosti, tau, sr, hop):
    dolzina = len(ovojnica_novosti)
    zacetek = int(np.argmax(ovojnica_novosti[:tau]))
    udarci_v_oknih = []
    trenutna_pozicija = zacetek

    okolica = int(tau * 0.25)

    while trenutna_pozicija < dolzina:
        levo = max(0, trenutna_pozicija - okolica)
        desno = min(dolzina - 1, trenutna_pozicija + okolica)
        dejanski_vrh = levo + int(np.argmax(ovojnica_novosti[levo:desno + 1]))
        udarci_v_oknih.append(dejanski_vrh)
        trenutna_pozicija += tau

    udarci_v_sekundah = np.array(udarci_v_oknih) * hop / sr
    return udarci_v_sekundah

def ustvari_zvok_metronoma(udarci_v_sekundah, sr, celotna_dolzina):
    kliki = np.zeros(celotna_dolzina, dtype=np.float32)
    trajanje_klika = 0.05
    st_vzorcev_klika = int(trajanje_klika * sr)
    t = np.arange(st_vzorcev_klika) / sr
    en_klik = np.sin(2 * np.pi * 1000.0 * t) * np.exp(-30.0 * t)

    for sekunda in udarci_v_sekundah:
        indeks = int(round(sekunda * sr))
        konec = min(indeks + st_vzorcev_klika, celotna_dolzina)

        if (konec - indeks) > 0:
            kliki[indeks:konec] += en_klik[:konec - indeks]

    return kliki

def narisi_grafe(y_orig, y_filtr, sr, ovojnica_novosti, hop, udarci_v_sekundah, bpm):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    čas_sekunde = np.arange(len(y_orig)) / sr

    axes[0].plot(čas_sekunde, y_orig, lw=0.4, color='steelblue')
    axes[0].set_title('1. Originalni zvočni signal (Mono)')
    axes[0].set_ylabel('Amplituda')
    axes[0].set_xlim([0, čas_sekunde[-1]])

    axes[1].plot(čas_sekunde, y_filtr, lw=0.4, color='seagreen')
    axes[1].set_title('2. Signal po nizkoprehodnem filtru (< 150 Hz) - Izolirani udarci')
    axes[1].set_ylabel('Amplituda')
    axes[1].set_xlim([0, čas_sekunde[-1]])

    čas_ovojnice = np.arange(len(ovojnica_novosti)) * hop / sr
    axes[2].plot(čas_ovojnice, ovojnica_novosti, lw=0.9, color='darkorange', label='Spektralni pretok')

    for udarec in udarci_v_sekundah:
        axes[2].axvline(x=udarec, color='crimson', alpha=0.6, lw=0.8)

    axes[2].set_title(f'3. Spektralni pretok in zaznani udarci (Rdeče črte) | Izračunan tempo = {bpm:.1f} BPM')
    axes[2].set_ylabel('Normirana energija')
    axes[2].set_xlabel('Čas (s)')
    axes[2].set_xlim([0, čas_ovojnice[-1]])
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('bpm_analiza_graf.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    vhodna_pesem = "song.mp3"
    izhodna_pesem = "rezultat_z_metronomom.wav"
    y, sr = nalozi_zvok(vhodna_pesem)
    print(f"Frekvenca vzorčenja: {sr} Hz | Trajanje: {len(y) / sr:.2f} sekund")

    y_filtriran = filtriraj_base(y, sr)
    S = izracunaj_stft_rocno(y_filtriran)
    ovojnica = izracunaj_spektralni_pretok(S)
    bpm, tau = ugotovi_bpm(ovojnica, sr, hop=512)

    print(f"\n Tempo pesmi: {bpm:.1f} BPM <<<\n")

    udarci = poisci_točne_udarce(ovojnica, tau, sr, hop=512)
    kliki = ustvari_zvok_metronoma(udarci, sr, len(y))
    končni_zvok = (y * 0.5) + kliki

    if np.max(np.abs(končni_zvok)) > 0:
        končni_zvok = končni_zvok / np.max(np.abs(končni_zvok)) * 0.95

    sf.write(izhodna_pesem, končni_zvok, sr)
    narisi_grafe(y, y_filtriran, sr, ovojnica, hop=512, udarci_v_sekundah=udarci, bpm=bpm)