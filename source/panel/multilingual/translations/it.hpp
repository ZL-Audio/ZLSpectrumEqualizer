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
    static constexpr std::array kTexts = {
        "Rilascia: bypassa la banda.",
        "Premi: metti in solo la banda.",
        "Scegli il tipo di filtro.",
        "Scegli la pendenza del filtro. Una pendenza più elevata rende più ripida la variazione della curva di risposta del filtro.",
        "Scegli la modalità stereo.",
        "Regola la frequenza.",
        "Regola il guadagno base e il guadagno target.",
        "Regola il fattore di qualità. Un valore di Q più elevato rende più stretta la larghezza di banda.",
        "Premi: attiva il comportamento dinamico.",
        "Clicca: disattiva la banda.",

        "Rilascia: bypassa l'elaborazione dinamica.",
        "Premi: emetti il segnale delta dinamico.",
        "Scegli la modalità dinamica.",
        "L'elaborazione dinamica utilizza una soglia statica.",
        "L'elaborazione dinamica utilizza una soglia dinamica legata alla loudness della side-chain di banda.",
        "L'elaborazione dinamica utilizza una soglia dinamica legata alla loudness della side-chain totale.",
        "Regola la soglia statica.",
        "Regola la soglia relativa della banda.",
        "Regola la soglia relativa totale.",
        "Regola la larghezza del knee dell'elaborazione dinamica.",
        "Regola il tempo di attacco dell'elaborazione dinamica.",
        "Regola il tempo di rilascio dell'elaborazione dinamica.",

        "Rilascia: bypassa il plugin.",
        "Premi: usa la side-chain esterna.\nRilascia: usa la side-chain interna.",

        "Scegli la risoluzione dell'elaborazione spettrale.",
        "Scegli il tipo di smoothing della side-chain nell'elaborazione spettrale.",
        "Scegli il valore di smoothing della side-chain nell'elaborazione spettrale.",
        "Regola la pendenza di tilt della side-chain nell'elaborazione spettrale.",
        "Regola lo skew dell'attacco della side-chain nell'elaborazione spettrale. Uno skew più elevato rende l'attacco più veloce alle alte frequenze.",
        "Regola lo skew del rilascio della side-chain nell'elaborazione spettrale. Uno skew più elevato rende il rilascio più veloce alle alte frequenze.",
        "Regola il gate spettrale della loudness relativa.",

        "Regola il guadagno di uscita aggiuntivo.",
        "Regola la scala del guadagno base e target di tutti i filtri.",
        "Premi: attiva la compensazione statica del guadagno (SGC). L'SGC non è imprecisa, ma non influisce sulla dinamica.",
        "Premi: avvia la misurazione della loudness integrata del segnale di ingresso e di uscita\nRilascia: aggiorna il guadagno di uscita alla differenza tra i due valori di loudness.",

        "Premi: attiva l'analizzatore di spettro del segnale di ingresso",
        "Premi: attiva l'analizzatore di spettro del segnale di uscita",
        "Premi: attiva l'analizzatore di spettro del segnale side-chain",
        "Scegli la velocità di decadimento degli analizzatori di spettro.",
        "Scegli la pendenza di tilt degli analizzatori di spettro.",
        "Scegli il valore di smoothing degli analizzatori di spettro.",
        "Scegli il tipo di smoothing degli analizzatori di spettro.",
        "Premi: attiva la funzione di congelamento. Passa il cursore sull'analizzatore per congelare lo spettro.",
        "Premi: attiva il rilevamento delle collisioni.",
        "Regola l'intensità del rilevamento delle collisioni.",

        "Doppio clic: apri le impostazioni dell'interfaccia utente.",
    };
}
