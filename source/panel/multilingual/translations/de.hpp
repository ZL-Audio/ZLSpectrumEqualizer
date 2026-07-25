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

namespace zlpanel::multilingual::de {
    static constexpr std::array kTexts = {
        "Loslassen: Band bypassen.",
        "Drücken: Band solo schalten.",
        "Filtertyp auswählen.",
        "Filtersteilheit auswählen. Eine höhere Steilheit lässt die Frequenzgangkurve steiler abfallen.",
        "Stereomodus auswählen.",
        "Frequenz regeln.",
        "Basis-Gain und Ziel-Gain regeln.",
        "Gütefaktor (Q) regeln. Ein höherer Q-Wert verringert die Bandbreite.",
        "Drücken: Dynamikverhalten aktivieren.",
        "Klicken: Band ausschalten.",

        "Loslassen: Dynamikbearbeitung bypassen.",
        "Dynamikmodus auswählen.",
        "Dynamikbearbeitung nutzt einen statischen Schwellenwert.",
        "Dynamikbearbeitung nutzt einen dynamischen Schwellenwert, basierend auf der Band-Sidechain-Lautheit.",
        "Dynamikbearbeitung nutzt einen dynamischen Schwellenwert, basierend auf der Gesamt-Sidechain-Lautheit.",
        "Statischen Schwellenwert regeln.",
        "Relativen Band-Schwellenwert regeln.",
        "Relativen Gesamtschwellenwert regeln.",
        "Knee-Breite der Dynamikbearbeitung regeln.",
        "Attack-Zeit der Dynamikbearbeitung regeln.",
        "Release-Zeit der Dynamikbearbeitung regeln.",

        "Loslassen: Plugin bypassen.",
        "Drücken: Externe Sidechain verwenden.\nLoslassen: Interne Sidechain verwenden.",

        "Auflösung der Spektralverarbeitung auswählen.",
        "Glättungstyp der Spektral-Sidechain auswählen.",
        "Glättungswert der Spektral-Sidechain auswählen.",
        "Tilt-Neigung der Spektral-Sidechain regeln.",
        "Attack-Skew der Spektral-Sidechain regeln. Ein höherer Skew bewirkt einen schnelleren Attack bei hohen Frequenzen.",
        "Release-Skew der Spektral-Sidechain regeln. Ein höherer Skew bewirkt einen schnelleren Release bei hohen Frequenzen.",
        "Spektralgate der relativen Lautheit regeln.",

        "Zusätzlichen Ausgangsgain regeln.",
        "Skalierung von Basis- und Ziel-Gain aller Filter regeln.",
        "Drücken: Statische Gain-Kompensation (SGC) aktivieren. SGC ist ungenau, beeinflusst jedoch nicht die Dynamik.",
        "Drücken: Integrierte Lautheit von Ein- und Ausgangssignal messen.\nLoslassen: Ausgangsgain auf die Differenz der beiden Lautheitswerte aktualisieren.",

        "Drücken: Spektrumanalysator für Eingangssignal aktivieren.",
        "Drücken: Spektrumanalysator für Ausgangssignal aktivieren.",
        "Drücken: Spektrumanalysator für Sidechain-Signal aktivieren.",
        "Abfallgeschwindigkeit der Spektrumanalysatoren auswählen.",
        "Tilt-Neigung der Spektrumanalysatoren auswählen.",
        "Glättungswert der Spektrumanalysatoren auswählen.",
        "Glättungstyp der Spektrumanalysatoren auswählen.",
        "Drücken: Einfrierfunktion aktivieren. Bewege die Maus über den Analysator, um das Spektrum einzufrieren.",
        "Drücken: Kollisionserkennung aktivieren.",
        "Stärke der Kollisionserkennung regeln.",

        "Doppelklick: UI-Einstellungen öffnen.",
    };
}
