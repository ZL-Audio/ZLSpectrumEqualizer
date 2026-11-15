// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

// This file is also dual licensed under the Apache License, Version 2.0. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>

#pragma once

#include <array>

namespace zlpanel::multilingual::it {
    inline constexpr std::array kTexts = {
        // band basic controls
        "Attiva/disattiva il bypass della banda.",
        "Attiva/disattiva l'ascolto in solo della banda.",
        "Seleziona il tipo di filtro.",
        "Seleziona la pendenza del filtro. Una pendenza maggiore produce una curva di risposta in frequenza più ripida.",
        "Seleziona la modalità del canale stereo.",
        "Regola la frequenza centrale o di taglio.",
        "Regola il guadagno di base e il guadagno target dinamico.",
        "Regola il fattore Q. Un valore di Q maggiore restringe la larghezza di banda.",
        "Attiva/disattiva il comportamento dinamico di equalizzazione per questa banda.",
        "Disattiva questa banda.",
        "Seleziona la banda precedente.",
        "Seleziona la banda successiva.",
        "Attiva/disattiva la scalatura passante per i filtri passa-basso e passa-alto.",
        "Regola la scala di ampiezza del filtro passante.",
        // band dynamic controls
        "Attiva/disattiva il bypass dell'elaborazione dinamica per questa banda.",
        "Attiva o disattiva l’uscita del segnale delta della dinamica spettrale.",
        "Seleziona la modalità dinamica spettrale.",
        "Utilizza una soglia statica per l'elaborazione dinamica.",
        "Utilizza una soglia dinamica basata sulla loudness della side-chain di banda.",
        "Utilizza una soglia dinamica basata sulla loudness della side-chain totale.",
        "Regola la soglia statica.",
        "Regola la soglia relativa della banda.",
        "Regola la soglia relativa totale.",
        "Regola l'ampiezza del ginocchio (Knee) dell'elaborazione dinamica.",
        "Regola il tempo di attacco dell'elaborazione dinamica.",
        "Regola il tempo di rilascio dell'elaborazione dinamica.",
        // top controls
        "Mostra/nasconde il pannello delle impostazioni dell'interfaccia.",
        "Mostra/nasconde il browser dei preset.",
        "Mostra/nasconde il pannello delle impostazioni dell'analizzatore.",
        "Mostra/nasconde il pannello delle impostazioni di uscita.",
        "Attiva/disattiva il bypass del plugin.",
        "Attiva/disattiva l'ingresso side-chain esterno.",
        // preset browser
        "Apre la cartella dei preset nel gestore file di sistema.",
        "Elimina il gruppo di preset selezionato.",
        "Elimina il preset selezionato.",
        "Chiude il browser dei preset.",
        // spectrum controls
        "Mostra o nasconde le impostazioni di elaborazione spettrale.",
        "Seleziona la risoluzione dell'elaborazione spettrale.",
        "Seleziona la modalità di smoothing in frequenza della side-chain per l'elaborazione spettrale.",
        "Regola la quantità di smoothing in frequenza della side-chain per l'elaborazione spettrale.",
        "Regola la pendenza di tilt della side-chain per l'elaborazione spettrale.",
        "Regola lo skew dell'attacco della side-chain per l'elaborazione spettrale. Uno skew più elevato accelera l'attacco alle alte frequenze.",
        "Regola lo skew del rilascio della side-chain per l'elaborazione spettrale. Uno skew più elevato accelera il rilascio alle alte frequenze.",
        "Regola il gate della loudness relativa per l'elaborazione spettrale.",
        // output controls
        "Regola il guadagno di uscita aggiuntivo.",
        "Regola la scala complessiva del guadagno di base e target di tutti i filtri.",
        "Attiva/disattiva la Compensazione Statica del Guadagno (SGC). La SGC è approssimata, ma non influisce sulla dinamica.",
        "Attiva/disattiva l'adattamento del guadagno basato sul loudness. Attivandolo misura il loudness di ingresso e uscita; disattivandolo imposta il guadagno di uscita sulla loro differenza.",
        // analyzer controls
        "Mostra/nasconde lo spettro del segnale in ingresso.",
        "Mostra/nasconde lo spettro del segnale in uscita.",
        "Mostra/nasconde lo spettro del segnale della side-chain.",
        "Seleziona la velocità di decadimento dell'analizzatore di spettro.",
        "Seleziona la pendenza di inclinazione (tilt) dell'analizzatore di spettro.",
        "Seleziona l'ampiezza di smussamento in frequenza dell'analizzatore di spettro.",
        "Seleziona la modalità di smussamento in frequenza.\nOCT: smussamento costante per frazioni di ottava.\nERB: smussamento per larghezza di banda rettangolare equivalente.",
        "Attiva/disattiva il congelamento dello spettro quando si tiene fermo il puntatore sull'analizzatore. Quando abilitato, puoi anche attivare/disattivare il congelamento tramite Shift.",
        "Seleziona la modalità del canale stereo visualizzata dall'analizzatore di spettro.",
        "Attiva/disattiva il rilevamento delle collisioni di frequenza.",
        "Regola l'intensità del rilevamento delle collisioni.",
        "Mostra/nasconde la lettura numerica dei valori.",
        "Mostra/nasconde il pannello dei misuratori di livello.",
        // curve scale & piano roll
        "Seleziona l'intervallo di visualizzazione in decibel per la curva di risposta dell'equalizzatore.",
        "Seleziona il limite superiore in decibel per la visualizzazione dell'analizzatore di spettro.",
        "Seleziona il livello minimo in decibel per la visualizzazione dell'analizzatore di spettro.",
        "Mostra/nasconde il piano roll.",
        " "
    };
}
